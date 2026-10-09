"""Chainable wrappers around polars frames of fisseq pipeline outputs.

A dataset wraps a `polars.LazyFrame`. Like a plot's layers, every chained method returns
a **new** dataset and leaves the original alone, so a base dataset can be branched
freely. Chaining only builds a query; it runs when the data is needed (`Dataset.df`,
`Dataset.save`, or passing the dataset to a plot, which accepts it in place of a
DataFrame).
"""

import copy
import pathlib
from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Self

import polars as pl

from . import _data, _remote, _variants, fisseq

if TYPE_CHECKING:
    from .calibration import Calibration
    from .summary import ClusterSummary

_META = "meta_"

# ClinVar calls made only of "Pathogenic" and "Likely pathogenic", e.g.
# "Pathogenic/Likely pathogenic" (not "Conflicting classifications of pathogenicity").
_PATHOGENIC_CALL = r"(?i)^(likely )?pathogenic(/(likely )?pathogenic)*$"
_CALL = "__clinvar_call"
_RANK = "__clinvar_rank"


class Dataset:
    """A lazily evaluated polars frame with chainable transforms and variant annotations.

    >>> ds = (fb.Dataset.read("profiles.parquet")
    ...         .variant_type()
    ...         .domain()
    ...         .filter(pl.col("meta_variant_type") != "WT"))
    >>> fb.BoxPlot(ds, x="meta_domain", y="meta_impact_score").save("vis/domain.png")

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame | Dataset
        The rows. Columns prefixed ``meta_`` are metadata; every other column is a feature.
    variant_col : str, default "meta_aa_changes"
        Column holding variant labels such as ``"A12V"``, used by the annotation methods.
    """

    #: The `Calibration` set by `calibrate` or `apply_calibration` (else ``None``).
    calibration: "Calibration | None" = None

    def __init__(
        self,
        data: "pl.DataFrame | pl.LazyFrame | Dataset",
        *,
        variant_col: str = "meta_aa_changes",
    ) -> None:
        if isinstance(data, Dataset):
            data = data.lazy()
        if isinstance(data, pl.DataFrame):
            data = data.lazy()
        if not isinstance(data, pl.LazyFrame):
            raise TypeError(
                f"data must be a polars DataFrame or LazyFrame, got {type(data).__name__}"
            )
        self._lf = data
        self._df: pl.DataFrame | None = None
        self.variant_col = variant_col

    @classmethod
    def read(
        cls,
        path: str | PathLike,
        *,
        download_dir: str | PathLike | None = None,
        refresh: bool = False,
        **kw: Any,
    ) -> Self:
        """Lazily read a parquet file (or glob). ``kw`` is passed to the constructor.

        ``path`` can also be a single remote file, ``"user@host:/path/file.parquet"``. It is
        copied with scp into ``download_dir`` (default: a temporary directory kept for the
        session) unless it is already there and ``refresh`` is false.
        """
        spec = _remote.parse(path)
        if spec is None:
            if download_dir is not None:
                raise ValueError(
                    f"download_dir is only used for remote files, but {path} is local"
                )
            return cls(pl.scan_parquet(path), **kw)
        host, remote_path = spec
        if any(ch in remote_path for ch in "*?["):
            raise ValueError("Remote paths can't be globs; pass a single file")
        remote_file = pathlib.PurePosixPath(remote_path)
        root = str(remote_file.parent)
        if download_dir is None:
            local = _remote.temp_dir(host, root)
        else:
            local = pathlib.Path(download_dir)
        [file] = _remote.Remote(host, root, local, refresh=refresh).fetch(
            [remote_file.name]
        )
        return cls(pl.scan_parquet(file), **kw)

    # ----- chaining -------------------------------------------------------------------

    def _replace(self, data: pl.LazyFrame | pl.DataFrame) -> Self:
        new = copy.copy(self)
        new._lf = data.lazy()
        new._df = data if isinstance(data, pl.DataFrame) else None
        return new

    def _require(self, *columns: str | None) -> None:
        _data.require_columns(pl.DataFrame(schema=self.schema), *columns)

    def pipe(
        self,
        function: Callable[..., pl.LazyFrame | pl.DataFrame],
        *args: Any,
        **kw: Any,
    ) -> Self:
        """Apply ``function(lazyframe, *args, **kw)`` and wrap the frame it returns."""
        return self._replace(function(self._lf, *args, **kw))

    def filter(self, *predicates: Any, **constraints: Any) -> Self:
        """Keep the rows matching ``predicates`` (see `polars.LazyFrame.filter`)."""
        return self._replace(self._lf.filter(*predicates, **constraints))

    def with_columns(self, *exprs: Any, **named_exprs: Any) -> Self:
        """Add or replace columns (see `polars.LazyFrame.with_columns`)."""
        return self._replace(self._lf.with_columns(*exprs, **named_exprs))

    def select(self, *exprs: Any, **named_exprs: Any) -> Self:
        """Keep only the given columns or expressions (see `polars.LazyFrame.select`)."""
        return self._replace(self._lf.select(*exprs, **named_exprs))

    def drop(self, *columns: Any, strict: bool = True) -> Self:
        """Remove columns (see `polars.LazyFrame.drop`)."""
        return self._replace(self._lf.drop(*columns, strict=strict))

    def rename(self, mapping: Mapping[str, str]) -> Self:
        """Rename columns (see `polars.LazyFrame.rename`)."""
        return self._replace(self._lf.rename(dict(mapping)))

    def sort(self, by: Any, *more_by: Any, **kw: Any) -> Self:
        """Sort rows (see `polars.LazyFrame.sort`)."""
        return self._replace(self._lf.sort(by, *more_by, **kw))

    def join(self, other: "Dataset | pl.DataFrame | pl.LazyFrame", **kw: Any) -> Self:
        """Join another dataset or frame. ``on`` defaults to ``variant_col``; see
        `polars.LazyFrame.join` for the other arguments."""
        if not any(k in kw for k in ("on", "left_on", "right_on")):
            kw["on"] = self.variant_col
        return self._replace(self._lf.join(other.lazy(), **kw))

    # ----- variant annotations --------------------------------------------------------

    def variant_type(
        self,
        *,
        output_col: str = "meta_variant_type",
        control_col: str | None = "meta_is_control",
    ) -> Self:
        """Classify each variant label and flag the controls.

        ``output_col`` gets one of "Synonymous", "Single Missense", "Frameshift",
        "Nonsense", "3nt Deletion", "WT" or "Other". ``control_col`` (skipped when
        ``None``) is ``True`` for synonymous variants without a ``:<tag>`` suffix, the
        pipeline's definition of a control.
        """
        self._require(self.variant_col)
        variant_type = _variants.variant_type_expr(self.variant_col)
        exprs = [variant_type.alias(output_col)]
        if control_col is not None:
            exprs.append(
                _variants.control_expr(variant_type, self.variant_col).alias(
                    control_col
                )
            )
        return self.with_columns(exprs)

    def position(
        self, *, strict: bool = False, output_col: str = "meta_position"
    ) -> Self:
        """Add the amino-acid position of each variant.

        By default the leading position of the first codon is used, so frameshifts,
        deletions and nonsense variants get one too. ``strict=True`` only accepts single
        substitutions such as ``"A12V"``; every other label gets null.
        """
        self._require(self.variant_col)
        return self.with_columns(
            _variants.position_expr(self.variant_col, strict=strict).alias(output_col)
        )

    def domain(
        self,
        regions: Mapping[str, tuple[int, int]] | None = None,
        *,
        output_col: str = "meta_domain",
    ) -> Self:
        """Name the protein region each single substitution falls in.

        ``regions`` maps a name to an inclusive ``(first, last)`` position range and
        defaults to `fisseq.LMNA_DOMAIN_REGIONS`. When ranges overlap, the first one
        listed wins. Labels other than single substitutions, and positions outside every
        range, get null.
        """
        self._require(self.variant_col)
        position = _variants.position_expr(self.variant_col, strict=True)
        regions = fisseq.LMNA_DOMAIN_REGIONS if regions is None else regions
        return self.with_columns(
            _variants.region_expr(position, regions).alias(output_col)
        )

    def tile(
        self,
        tiles: Sequence[tuple[str, int, int]] | None = None,
        *,
        allow_multiple: bool = False,
        output_col: str = "meta_tile",
    ) -> Self:
        """Name the tile each variant's position falls in.

        ``tiles`` is a list of ``(name, first, last)`` inclusive ranges and defaults to
        `fisseq.LMNA_TILES`. A position in the overlap between two tiles gets the first
        one, or both joined by ``","`` with ``allow_multiple=True``. Split the result by
        tile with ``ds.df.partition_by("meta_tile", as_dict=True)``.
        """
        self._require(self.variant_col)
        position = _variants.position_expr(self.variant_col)
        tiles = fisseq.LMNA_TILES if tiles is None else tiles
        return self.with_columns(
            _variants.tile_expr(position, tiles, allow_multiple=allow_multiple).alias(
                output_col
            )
        )

    def clinvar(
        self,
        clinvar: "str | PathLike | pl.DataFrame | pl.LazyFrame",
        *,
        variant_type_col: str = "meta_variant_type",
        output_col: str = "meta_clinvar_annotation",
    ) -> Self:
        """Join ClinVar clinical significance calls and build a combined annotation column.

        ``clinvar`` is the converted ClinVar table (a parquet path or a frame) with a
        ``variant`` column in the same ``"A12V"`` format and a
        ``clinvar_clinical_significance`` column. Every call is kept, one record per
        variant: the most severe one (pathogenic or likely pathogenic, then uncertain
        significance, then any other call), the first in the table among equals. Its
        columns are joined in with a ``meta_`` prefix.

        ``output_col`` is `fisseq.PATHOGENIC` for pathogenic and likely pathogenic calls,
        `fisseq.UNCERTAIN` for uncertain significance, and the variant type otherwise
        (including conflicting, benign and missing calls). Run `variant_type` first.
        """
        self._require(self.variant_col, variant_type_col)
        if isinstance(clinvar, (str, PathLike)):
            table = pl.scan_parquet(clinvar)
        else:
            table = clinvar.lazy()
        schema = table.collect_schema()
        _data.require_columns(
            pl.DataFrame(schema=schema), "variant", "clinvar_clinical_significance"
        )
        columns = schema.names()
        significance = pl.col("clinvar_clinical_significance")
        call = (
            pl.when(significance.str.contains(_PATHOGENIC_CALL))
            .then(pl.lit(fisseq.PATHOGENIC))
            .when(significance.str.starts_with(fisseq.UNCERTAIN))
            .then(pl.lit(fisseq.UNCERTAIN))
        )
        rank = (
            pl.when(call == fisseq.PATHOGENIC)
            .then(0)
            .when(call == fisseq.UNCERTAIN)
            .then(1)
            .otherwise(2)
        )
        table = (
            table.with_columns(call.alias(_CALL), rank.alias(_RANK))
            .sort(_RANK, maintain_order=True)
            .unique(subset=["variant"], keep="first", maintain_order=True)
            .drop(_RANK)
            .rename(
                {
                    c: f"{_META}{c}"
                    for c in columns
                    if c != "variant" and not c.startswith(_META)
                }
            )
        )
        return self._replace(
            self._lf.join(
                table,
                left_on=self.variant_col,
                right_on="variant",
                how="left",
                maintain_order="left",
            )
            .with_columns(
                pl.coalesce(_CALL, pl.col(variant_type_col)).alias(output_col)
            )
            .drop(_CALL)
        )

    # ----- calibration ----------------------------------------------------------------

    def calibrate(
        self,
        score: str,
        *,
        gnomad: "str | PathLike | pl.DataFrame",
        clinvar: "str | PathLike | pl.DataFrame | pl.LazyFrame | None" = None,
        direction: str = "auto",
        benign_method: str = "avg",
        n_components: int | str = "auto",
        n_bootstrap: int = 1000,
        n_restarts: int = 8,
        prior: float | None = None,
        splice_max: float | None = None,
        min_stars: int | None = 1,
        strict_constraint: bool = False,
        out_of_bag: bool = True,
        seed: int = 0,
        n_jobs: int | None = -1,
        prefix: str = "meta_excalibr",
    ) -> Self:
        """Calibrate a score into ACMG/AMP evidence points with ExCALIBR.

        Fits the score distributions of pathogenic (P/LP), benign (B/LB), gnomAD and
        synonymous variants as skew-normal mixtures over bootstrap resamples, estimates
        the prior probability of pathogenicity from gnomAD, and turns the local
        likelihood ratio into points from -8 (benign) to +8 (pathogenic). See the
        calibration guide in the docs for the method and its limits.

        Unlike the rest of the chain this is **eager**: it collects the data once, fits,
        and returns a dataset wrapping the result. The fitted `Calibration` is on the
        returned dataset's ``calibration`` attribute (carried along by later chained
        calls) and the variants get these columns:

        - ``<prefix>_group``: the control group a variant is in (``"P/LP"``,
          ``"B/LB"``, ``"Synonymous"``, ``"gnomAD"``, first match in that order), else
          null. ``<prefix>_groups`` lists every group it is in.
        - ``<prefix>_lr`` and ``<prefix>_posterior``: median bootstrap local likelihood
          ratio and the posterior at the calibration's prior.
        - ``<prefix>_points``: evidence points. Control variants get out-of-bag points
          (only bootstrap fits that held them out) when ``out_of_bag`` and
          ``<prefix>_oob`` is true; every other variant gets points from the thresholds.

        Parameters
        ----------
        score : str
            Numeric score column, e.g. ``"auroc_pooled_corrected"`` after
            `OvwtScores.per_variant`, or ``"meta_distinguishability_score"``.
        gnomad : path | pl.DataFrame
            The gnomAD browser CSV export for the gene (required: it is the population
            sample the prior comes from).
        clinvar : path | frame | None
            ClinVar controls, in the table format `clinvar` reads (``variant``,
            ``clinvar_clinical_significance``, optionally ``clinvar_review_status`` or
            ``clinvar_stars``). ``None`` uses the gnomAD export's ClinVar column, which
            only covers variants gnomAD observed and has no review-star filter.
        direction : {"auto", "lower_pathogenic", "higher_pathogenic", "both"}
            Which end of the score is pathogenic. ``"auto"`` compares the control means
            and detects bidirectional (both ends pathogenic) assays.
        benign_method : {"avg", "benign", "synonymous"}
            Benign reference: B/LB, synonymous, or the average of their mixture weights
            (the default; more stable when B/LB is small).
        n_components : {"auto", 2, 3}
            Skew-normal components. ``"auto"`` fits both and keeps 3 only if it wins on
            held-out likelihood in at least 95% of bootstraps (doubles the run time).
        n_bootstrap, n_restarts : int
            Bootstrap iterations (the published method uses 1000) and EM restarts per
            iteration. Run time scales with both; use ``n_bootstrap=100`` for a first
            look and 1000 for anything reported.
        prior : float | None
            Use this prior probability of pathogenicity instead of estimating it.
        splice_max : float | None
            Leave gnomAD variants with ``spliceai_ds_max`` above this out of the gnomAD
            sample (0.2 in Zeiberg et al. v2), for assays blind to splicing effects.
        min_stars : int | None
            Minimum ClinVar review stars for a ``clinvar`` table record to count.
        strict_constraint : bool
            Enforce the mixture density constraint over the whole score range, not just
            where the densities are non-negligible.
        out_of_bag : bool
            Give control variants out-of-bag points. Keep this on whenever the score was
            tuned or thresholded against ClinVar labels.
        seed : int
            Seeds every bootstrap; results do not depend on ``n_jobs``.
        n_jobs : int | None
            Parallel workers (joblib; -1 is every core).
        prefix : str
            Prefix of the added columns.

        Raises
        ------
        ValueError
            If a variant appears more than once (pool first with
            `OvwtScores.per_variant` or `Profiles.median_across_batches`), if gnomAD has
            fewer than 5 scored variants, or if no control group has at least 5. Smaller
            control groups (5 to 9) log a warning and give ``calibration.reliable ==
            False``.
        """
        from .calibration._dataset import calibrate

        return calibrate(
            self,
            score,
            gnomad=gnomad,
            clinvar=clinvar,
            direction=direction,
            benign_method=benign_method,
            n_components=n_components,
            n_bootstrap=n_bootstrap,
            n_restarts=n_restarts,
            prior=prior,
            splice_max=splice_max,
            min_stars=min_stars,
            strict_constraint=strict_constraint,
            out_of_bag=out_of_bag,
            seed=seed,
            n_jobs=n_jobs,
            prefix=prefix,
        )

    def apply_calibration(
        self,
        calibration: "Calibration | str | PathLike",
        score: str | None = None,
        *,
        prefix: str = "meta_excalibr",
    ) -> Self:
        """Score ``score`` with a fitted `Calibration` (or a path to one saved with
        `Calibration.to_json`) without refitting. Adds ``<prefix>_lr``,
        ``<prefix>_posterior`` and ``<prefix>_points``; ``score`` defaults to the column
        the calibration was fit on. Eager, like `calibrate`."""
        from .calibration._dataset import apply_calibration

        return apply_calibration(self, calibration, score, prefix=prefix)

    # ----- summaries ------------------------------------------------------------------

    def cluster_summary(
        self,
        by: str = "meta_cluster_idx",
        *,
        medians: Sequence[str] | Mapping[str, str] = (),
        zscore: bool | Sequence[str] = False,
        shares: "Mapping[str, str | Mapping[str, pl.Expr]] | None" = None,
        levels: Mapping[str, Sequence[Any]] | None = None,
        control_col: str = "meta_is_control",
        count_col: str = "n",
        label_col: str = "label",
    ) -> "ClusterSummary":
        """Summarize the rows (one per variant) per cluster, for a `ClusterMap`.

        >>> summary = spatial.cluster_summary(
        ...     medians=[*landmark_cols, "meta_distinguishability_score"],
        ...     zscore=landmark_cols,
        ...     shares={
        ...         "class": {fisseq.PATHOGENIC: is_pathogenic, "Synonymous": is_synonymous},
        ...         "domain": "meta_domain",
        ...     },
        ... )
        >>> summary.group("domain", "Share of domain", palette="tab10")  # a FeatureGroup

        Parameters
        ----------
        by : str, default "meta_cluster_idx"
            Cluster id column. Rows come out in natural order of it (``"2"`` before
            ``"10"``).
        medians : Sequence[str] | Mapping[str, str]
            Columns to take the per-cluster median of. A mapping renames them
            (column -> output name).
        zscore : bool | Sequence[str]
            Z-score these ``medians`` columns (``True``: all of them) against the control
            rows (``control_col``, see `variant_type`) over every row before taking the
            medians.
        shares : Mapping[str, str | Mapping[str, pl.Expr]] | None
            Share blocks by name. A column name gives one share per level of that column
            (null ignored; levels in the fisseq order, e.g. domains N- to C-terminal, else
            natural order). A mapping of level name -> boolean expression gives one share
            per expression, e.g. variant classes. Each share is the fraction of that
            level's variants that fall in the cluster, so every level column sums to 1
            (normalized per level, not per cluster). The level names become the columns.
        levels : Mapping[str, Sequence] | None
            For column share blocks, fix the levels (and their order) instead of reading
            them from the data; levels with no variants are kept (their share is NaN).
        control_col : str, default "meta_is_control"
        count_col : str, default "n"
            Number of rows per cluster.
        label_col : str, default "label"
            ``"<id> (n=<count>)"``, for ``ClusterMap(row_labels=...)``.

        Returns
        -------
        ClusterSummary
            Lazy like any dataset (only the levels of column share blocks are read up
            front, unless ``levels`` gives them). Its ``totals`` give each level's variant
            count and ``group(key, ...)`` builds a `FeatureGroup` labelled with them.
        """
        from .summary import cluster_summary

        return cluster_summary(
            self,
            by,
            medians=medians,
            zscore=zscore,
            shares=shares,
            levels=levels,
            control_col=control_col,
            count_col=count_col,
            label_col=label_col,
        )

    # ----- materializing --------------------------------------------------------------

    @property
    def df(self) -> pl.DataFrame:
        """The collected DataFrame (computed once per dataset, then reused)."""
        if self._df is None:
            self._df = self._lf.collect()
        return self._df

    def collect(self) -> pl.DataFrame:
        """Same as `df`."""
        return self.df

    def lazy(self) -> pl.LazyFrame:
        """The underlying query as a `polars.LazyFrame`."""
        return self._df.lazy() if self._df is not None else self._lf

    @property
    def schema(self) -> pl.Schema:
        return self._lf.collect_schema()

    @property
    def columns(self) -> list[str]:
        return self.schema.names()

    @property
    def features(self) -> list[str]:
        """Feature columns: every column not prefixed ``meta_``."""
        return [c for c in self.columns if not c.startswith(_META)]

    @property
    def meta(self) -> list[str]:
        """Metadata columns: every column prefixed ``meta_``."""
        return [c for c in self.columns if c.startswith(_META)]

    def save(self, path: str | PathLike, **write_kw: Any) -> Self:
        """Write the data to a parquet file, creating parent directories."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.write_parquet(path, **write_kw)
        return self

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({len(self.meta)} meta columns, "
            f"{len(self.features)} feature columns)"
        )
