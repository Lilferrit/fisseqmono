"""Per-variant feature profiles from the pipeline's feature-selection outputs."""

import fnmatch
import pathlib
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import polars as pl
import polars.selectors as cs

from . import _data, _pipeline, _transforms
from .dataset import Dataset

if TYPE_CHECKING:
    from .ovwt import OvwtScores

#: Aggregate statistics the pipeline appends to feature names (``<feature>_<statistic>``).
STATISTICS: tuple[str, ...] = (
    "median",
    "mean",
    "KS",
    "signedKS",
    "AUROC",
    "KSnegLogP",
    "AUROCnegLogP",
)

#: Feature categories, matched as substrings of the feature name in this order.
CATEGORIES: tuple[str, ...] = (
    "AreaShape",
    "Correlation",
    "Texture",
    "RadialDistribution",
    "Intensity",
    "Granularity",
    "Neighbors",
    "Zernike",
    "Location",
)

_COMPARTMENT_PREFIXES = ("Mean", "Median", "Children")
_CHANNEL_RE = re.compile(r"CH\d+")


class Profiles(Dataset):
    """Feature profiles: one row per variant, or per (variant, batch) before
    `median_across_batches`.

    Feature columns are named ``<feature>_<statistic>`` (e.g. ``AreaShape_Area_median``)
    as the pipeline writes them. Build them from a pipeline run with `from_pipeline`,
    then chain transforms. Every method returns a new `Profiles`:

    >>> profiles = (
    ...     fb.Profiles.from_pipeline(run_dir, types=["median"], passthrough=["KSnegLogP"])
    ...       .median_across_batches(paired={"_median": "_KSnegLogP"})
    ...       .variant_type()
    ...       .clinvar("clinvar_converted.parquet")
    ... )
    >>> fb.VolcanoPlot.from_wide(profiles).save("vis/volcano.png")

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame | Dataset
        The profiles.
    variant_col : str, default "meta_aa_changes"
        Variant label column.
    batch_col : str, default "meta_experiment"
        Column naming each row's batch, while there is one row per (variant, batch).
    passthrough : Sequence[str], default ()
        Statistics (column suffixes such as ``"KSnegLogP"``) that are carried along but
        are not profile values: `normalize` leaves them alone and `keep_features` keeps
        them next to their feature. `from_pipeline` sets this.
    """

    def __init__(
        self,
        data: "pl.DataFrame | pl.LazyFrame | Dataset",
        *,
        variant_col: str = "meta_aa_changes",
        batch_col: str = "meta_experiment",
        passthrough: Sequence[str] = (),
    ) -> None:
        super().__init__(data, variant_col=variant_col)
        self.batch_col = batch_col
        self.passthrough = tuple(s.lstrip("_") for s in passthrough)
        #: Loadings from the last `pca` / `pca_reduce` call: a ``component`` and an
        #: ``explained_variance_ratio`` column, then one column per input feature.
        self.pca_loadings: pl.DataFrame | None = None
        #: Explained variance of every fitted component from the last `pca` /
        #: `pca_reduce` / `impact_scores` call: ``component``, ``explained_variance_ratio``,
        #: ``cumulative_explained_variance`` and ``retained`` (kept as a feature). Pass it to
        #: `ExplainedVariancePlot`.
        self.pca_explained_variance: pl.DataFrame | None = None
        #: With ``pca_reduce(noise_floor=True)``: the variance ratio of the first component
        #: of the column-shuffled data.
        self.pca_noise_floor: float | None = None
        #: From the last `feature_select` call: one row per input feature, with
        #: ``feature`` and ``kept``.
        self.feature_selection: pl.DataFrame | None = None

    # ----- constructors ---------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        pipeline_dir: "str | PathLike | _pipeline.Source",
        types: Sequence[str] = ("median",),
        passthrough: Sequence[str] = (),
        *,
        batches: Sequence[str] | None = None,
        exclude: "_pipeline.Patterns | None" = None,
        features: Literal["union", "intersection"] = "union",
        metadata: bool | Sequence[str] = False,
        download_dir: str | PathLike | None = None,
        refresh: bool = False,
        variant_col: str = "meta_aa_changes",
        batch_col: str = "meta_experiment",
    ) -> Self:
        """Read ``feature_select_batchwise`` aggregates, one row per (variant, batch).

        ``pipeline_dir`` is a local run directory or a remote one, ``"user@host:/path"``.
        A remote run is listed over ssh and only the files read here are copied with scp,
        into ``download_dir`` (default: a temporary directory kept for the session), where
        later calls reuse them unless ``refresh=True``.

        For each batch, the ``aggregates/<type>.parquet`` files for ``types`` and the
        ``passthrough_aggregates/<type>.parquet`` files for ``passthrough`` (e.g.
        ``"KSnegLogP"``) are joined on the variant; variants missing from any of them are
        dropped. Each batch is tagged in ``batch_col`` with its directory name, and batches
        are stacked in natural order (``T2_R1`` before ``T10_R1``). ``batches`` selects a
        subset, and ``exclude`` drops the batches matching any of its patterns
        (``fnmatch`` globs such as ``"T10_*"``, or compiled regexes such as
        ``re.compile(r"_R3$")``).

        ``features="union"`` keeps every feature of any batch (null where a batch lacks
        it). ``"intersection"`` keeps only the features present in every batch, logs a
        warning per batch naming the columns it dropped, and raises if none is shared.

        ``metadata`` left-joins per-variant ``meta_`` columns (such as
        ``meta_num_cells``) from each batch's ``output.parquet``: ``True`` joins all of
        them, a list joins those named. Sum them across batches with
        ``median_across_batches(sum_cols=[...])``.

        The pipeline writes the aggregates already z-scored against each batch's
        synonymous controls, so `normalize` is not needed (and, up to rounding, does
        nothing). The passthrough values are raw.
        """
        if not types and not passthrough:
            raise ValueError("Pass at least one aggregate type or passthrough type")
        if features not in ("union", "intersection"):
            raise ValueError(f"features must be 'union' or 'intersection', got {features!r}")
        src = _pipeline.source(pipeline_dir, download_dir, refresh)
        stage = _pipeline.FEATURE_SELECT
        names = src.batches(stage, batches, exclude)
        files = [f"aggregates/{t}.parquet" for t in types]
        files += [f"passthrough_aggregates/{t}.parquet" for t in passthrough]
        if metadata is not False:
            files.append("output.parquet")
        local = iter(src.files([f"{stage}/{b}/{f}" for b in names for f in files]))
        paths = {b: dict(zip(files, local)) for b in names}
        frames: dict[str, pl.LazyFrame] = {}
        for b in names:
            parts = [
                _pipeline.scan(paths[b][f]).select(variant_col, ~cs.starts_with("meta_"))
                for f in files
                if f != "output.parquet"
            ]
            batch_lf = parts[0]
            for part in parts[1:]:
                batch_lf = batch_lf.join(part, on=variant_col)
            frames[b] = batch_lf
        if features == "intersection":
            present = {
                b: [c for c in lf.collect_schema().names() if c != variant_col]
                for b, lf in frames.items()
            }
            common = _transforms.intersect_features(present)
            frames = {b: lf.select(variant_col, *common) for b, lf in frames.items()}
        if metadata is not False:
            frames = {
                b: lf.join(
                    _output_metadata(paths[b]["output.parquet"], metadata, variant_col),
                    on=variant_col,
                    how="left",
                    maintain_order="left",
                )
                for b, lf in frames.items()
            }
        return cls(
            pl.concat(
                [_pipeline.tag(lf, b, batch_col) for b, lf in frames.items()],
                how="diagonal_relaxed",
            ),
            variant_col=variant_col,
            batch_col=batch_col,
            passthrough=passthrough,
        )

    @classmethod
    def from_global(cls, pipeline_dir: str | PathLike, channel: str, **kw: Any) -> Self:
        """Read ``global/<channel>/feature_select/aggregate.parquet`` from an older
        pipeline run (one row per variant, aggregated across batches).

        .. deprecated::
            The pipeline no longer writes ``global/``. Build the same table with the
            ``fisseqborn-global`` command (or `fisseqborn.write_global`) and read its
            ``<out>/feature_select/aggregate.parquet`` with `Profiles.read`.
        """
        warnings.warn(
            "Profiles.from_global reads the pipeline's global/ directory, which it no longer "
            "writes. Run `fisseqborn-global <pipeline_dir> --out <dir>` (or "
            "fisseqborn.write_global) and use "
            "Profiles.read('<dir>/feature_select/aggregate.parquet') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = (
            pathlib.Path(pipeline_dir) / "global" / channel / "feature_select" / "aggregate.parquet"
        )
        return cls(_pipeline.scan(path), **kw)

    def _replace(self, data: pl.LazyFrame | pl.DataFrame) -> Self:
        new = super()._replace(data)
        new.pca_loadings = None
        new.pca_explained_variance = None
        new.pca_noise_floor = None
        new.feature_selection = None
        return new

    # ----- helpers --------------------------------------------------------------------

    def _is_passthrough(self, column: str) -> bool:
        return any(column.endswith(f"_{s}") for s in self.passthrough)

    @property
    def values(self) -> list[str]:
        """Feature columns holding profile values (the features minus passthrough ones)."""
        return [c for c in self.features if not self._is_passthrough(c)]

    def _matrix(self) -> tuple[pl.DataFrame, np.ndarray]:
        df = self.df
        if not self.values:
            raise ValueError("No feature columns to embed")
        x = df.select(self.values).cast(pl.Float64).to_numpy()
        if not np.isfinite(x).all():
            raise ValueError(
                "Features contain null, NaN or infinite values; chain .drop_nonfinite() first"
            )
        return df, x

    # ----- transforms -----------------------------------------------------------------

    def normalize(
        self,
        *,
        by: str | Sequence[str] | None = None,
        types: Sequence[str] | None = None,
        control_col: str = "meta_is_control",
    ) -> Self:
        """Z-score each feature against the control rows: ``(x - mean) / std``.

        ``by`` computes the statistics within groups, e.g. ``by="meta_experiment"`` to
        normalize each batch against its own synonymous variants. ``types`` limits it to
        features with those statistic suffixes (default: every non-passthrough feature).
        NaN is treated as missing, and a feature whose control std is ~0 becomes null.
        Run `variant_type` first to create ``control_col``.
        """
        if control_col not in self.columns:
            raise ValueError(
                f"Column {control_col!r} not found; chain .variant_type() before .normalize()"
            )
        if by is not None:
            self._require(*([by] if isinstance(by, str) else by))
        features = self.values
        if types is not None:
            suffixes = tuple(f"_{t.lstrip('_')}" for t in types)
            features = [c for c in features if c.endswith(suffixes)]
        return self.with_columns(
            _transforms.normalize_exprs(features, control_col=control_col, by=by)
        )

    def median_across_batches(
        self,
        *,
        paired: Mapping[str, str] | None = None,
        n_col: str = "meta_n_experiments",
        features: Literal["union", "intersection"] = "union",
        sum_cols: Sequence[str] = (),
    ) -> Self:
        """Collapse to one row per variant: each feature's median across batches.

        ``paired`` maps a value suffix to a companion suffix, e.g. ``{"_median":
        "_KSnegLogP"}``: for each feature with both columns, both values come from the
        same batch (the one holding the lower-middle value), so every p-value belongs to
        the median it is shown with. Other ``meta_`` columns keep their first value, except
        ``sum_cols`` (e.g. ``["meta_num_cells"]``), which are summed. ``n_col`` counts the
        batches each variant was measured in.

        ``features="intersection"`` first drops every feature that has no value at all in
        some batch (as stacking batches with different features leaves it), logging a
        warning per batch naming what it dropped; it raises if nothing is left. This runs
        the query once to see which features each batch has.
        """
        self._require(self.variant_col, self.batch_col, *sum_cols)
        if features not in ("union", "intersection"):
            raise ValueError(f"features must be 'union' or 'intersection', got {features!r}")
        kept = self.features
        lf = self._lf
        if features == "intersection":
            present = _transforms.features_by_batch(lf, kept, batch_col=self.batch_col)
            common = set(_transforms.intersect_features(present))
            dropped = [c for c in kept if c not in common]
            kept = [c for c in kept if c in common]
            lf = lf.drop(dropped)
        return self._replace(
            _transforms.median_across_batches(
                lf,
                kept,
                variant_col=self.variant_col,
                batch_col=self.batch_col,
                paired=paired,
                n_col=n_col,
                sum_cols=sum_cols,
            )
        )

    def keep_features(self, patterns: str | Iterable[str]) -> Self:
        """Keep only the features matching ``patterns`` (names or ``fnmatch`` globs such
        as ``"*_CH1_*"``), plus every ``meta_`` column.

        Names that aren't present are ignored, so a blocklist consensus covering more
        statistics than these profiles can be passed directly. Passthrough columns (e.g.
        ``X_KSnegLogP``) are kept whenever their feature (``X_median``) is.
        """
        return self.select(self._keep_columns(self._match(patterns)))

    def _keep_columns(self, features: Iterable[str]) -> list[str]:
        """Every ``meta_`` column, ``features``, and the passthrough columns of those
        features, in their current order."""
        keep = set(features)
        bases = {c.rsplit("_", 1)[0] for c in keep}
        for col in self.features:
            if self._is_passthrough(col) and any(
                col.removesuffix(f"_{s}") in bases for s in self.passthrough
            ):
                keep.add(col)
        return [c for c in self.columns if c.startswith("meta_") or c in keep]

    def drop_features(self, patterns: str | Iterable[str]) -> Self:
        """Drop the features matching ``patterns`` (names or ``fnmatch`` globs such as
        ``"*CH2*"``)."""
        return self.drop(self._match(patterns))

    def _match(self, patterns: str | Iterable[str]) -> list[str]:
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        exact = {p for p in patterns if not any(ch in p for ch in "*?[")}
        globs = [p for p in patterns if p not in exact]
        return [
            c for c in self.features if c in exact or any(fnmatch.fnmatchcase(c, g) for g in globs)
        ]

    def drop_nonfinite(self) -> Self:
        """Drop every feature column that holds a null, NaN or infinite value."""
        df = self.df
        bad = df.select(
            (~_transforms.finite(pl.col(c)).is_not_null().all()).alias(c) for c in self.features
        )
        return self._replace(df.drop([c for c in bad.columns if bad[0, c]]))

    def feature_select(
        self,
        operations: Sequence[str] = ("variance_threshold", "blocklist", "correlation_threshold"),
        *,
        corr_threshold: float = 0.9,
        corr_method: Literal["pearson", "spearman", "kendall"] = "pearson",
        freq_cut: float = 0.05,
        unique_cut: float = 0.01,
        na_cutoff: float = 0.05,
        blocklist_file: str | PathLike | None = None,
        outlier_cutoff: float = 500.0,
        min_variance: float = 1e-6,
        **pycytominer_kw: Any,
    ) -> Self:
        """Drop redundant and uninformative features with `pycytominer.feature_select`.

        The ``operations`` run in order over the profile values (passthrough columns are
        not inputs, and are kept whenever their feature is). The defaults are the
        pipeline's: near-zero variance (``freq_cut``, ``unique_cut``, ``min_variance``),
        the blocklist, then one of every pair of features correlated above
        ``corr_threshold``. Other pycytominer operations are ``"drop_na_columns"``
        (``na_cutoff``), ``"drop_outliers"`` (``outlier_cutoff``) and ``"noise_removal"``
        (pass ``noise_removal_perturb_groups`` and ``noise_removal_stdev_cutoff`` through
        ``pycytominer_kw``). ``"blocklist"`` uses pycytominer's built-in Cell Painting
        blocklist unless ``blocklist_file`` is given.

        Which features were kept is stored on `feature_selection` (``feature``, ``kept``).
        Needs ``fisseqborn[select]``.

        >>> selected = (profiles.median_across_batches()
        ...                     .drop_nonfinite()
        ...                     .feature_select(corr_threshold=0.8)
        ...                     .pca_reduce(variance=0.9))
        """
        pycytominer = _transforms.require_extra("pycytominer", "select")
        df = self.df
        values = self.values
        if not values:
            raise ValueError("No feature columns to select from")
        selected = pycytominer.feature_select(
            profiles=df.select(values).to_pandas(),
            features=values,
            image_features=False,
            samples="all",
            operation=list(operations),
            corr_threshold=corr_threshold,
            corr_method=corr_method,
            freq_cut=freq_cut,
            unique_cut=unique_cut,
            na_cutoff=na_cutoff,
            blocklist_file=None if blocklist_file is None else str(blocklist_file),
            outlier_cutoff=outlier_cutoff,
            min_variance=min_variance,
            **pycytominer_kw,
        )
        kept = set(selected.columns)
        new = self._replace(df.select(self._keep_columns(kept)))
        new.feature_selection = pl.DataFrame(
            {"feature": values, "kept": [c in kept for c in values]},
            schema={"feature": pl.String, "kept": pl.Boolean},
        )
        return new

    def impact_score(
        self, *, control_col: str = "meta_is_control", output_col: str = "meta_impact_score"
    ) -> Self:
        """Add the impact score: the cosine distance between each profile and the median
        control profile, halved so it runs from 0 (same direction) to 1 (opposite).

        Missing values are skipped per row and feature. Run `variant_type` first.
        """
        if control_col not in self.columns:
            raise ValueError(
                f"Column {control_col!r} not found; chain .variant_type() before .impact_score()"
            )
        return self.with_columns(
            _transforms.impact_score_expr(self.values, control_col=control_col).alias(output_col)
        )

    # ----- embeddings -----------------------------------------------------------------

    def pca(self, n_components: int = 5, *, prefix: str = "meta_pc_") -> Self:
        """Add the first ``n_components`` principal component scores as
        ``meta_pc_1 .. meta_pc_<n>``; loadings are kept on `pca_loadings`."""
        from sklearn.decomposition import PCA

        df, x = self._matrix()
        model = PCA(n_components=n_components)
        scores = model.fit_transform(x)
        names = [f"{prefix}{i + 1}" for i in range(scores.shape[1])]
        new = self._replace(
            df.drop(names, strict=False).with_columns(
                pl.Series(name, scores[:, i]) for i, name in enumerate(names)
            )
        )
        new.pca_loadings = _loadings(model, names, self.values)
        new.pca_explained_variance = _explained_variance(
            model.explained_variance_ratio_, len(names), prefix, start=1
        )
        return new

    def pca_reduce(
        self,
        variance: float = 0.9,
        *,
        noise_floor: bool = False,
        seed: int = 0,
        prefix: str = "X_",
    ) -> Self:
        """Replace the features with their principal components ``X_0 .. X_<k-1>``.

        ``k`` is the fewest components that together explain ``variance`` of the total
        variance, or with ``noise_floor=True``, the number of components explaining more
        variance than the first component of the data with each column shuffled
        independently (seeded by ``seed``). Loadings of the kept components are stored on
        `pca_loadings`, the explained variance of every component on
        `pca_explained_variance` and, with ``noise_floor=True``, the floor ratio on
        `pca_noise_floor`:

        >>> reduced = profiles.pca_reduce(variance=0.9)
        >>> fb.ExplainedVariancePlot(reduced.pca_explained_variance, thresholds=[0.9])
        """
        df, x = self._matrix()
        model, scores = _fit_pca(x)
        ratios = model.explained_variance_ratio_
        floor = None
        if noise_floor:
            k, floor = _transforms.pca_noise_floor(x, ratios, seed)
            k = max(1, min(k, len(ratios)))
        else:
            k = _n_components(ratios, variance)
        names = [f"{prefix}{i}" for i in range(k)]
        new = self._replace(
            df.drop(self.features).with_columns(
                pl.Series(name, scores[:, i]) for i, name in enumerate(names)
            )
        )
        new.passthrough = ()
        new.pca_loadings = _loadings(model, names, self.values, k)
        new.pca_explained_variance = _explained_variance(ratios, k, prefix)
        new.pca_noise_floor = floor
        return new

    def impact_scores(
        self,
        variances: Sequence[float] = (0.7, 0.8, 0.9, 1.0),
        *,
        control_col: str = "meta_is_control",
        prefix: str = "meta_impact_score_",
    ) -> Self:
        """Add the impact score computed on the principal components, once per
        cumulative-variance threshold.

        PCA is fitted once. For each ratio in ``variances``, the impact score (see
        `impact_score`) is computed on the fewest components explaining that share of the
        variance, as ``pca_reduce(variance=v).impact_score()`` would, and added as
        ``<prefix><v>`` (e.g. ``meta_impact_score_0.9``). The features themselves are kept,
        and `pca_explained_variance` is set (``retained`` marks the largest threshold's
        components).

        >>> scored = profiles.impact_scores([0.7, 0.9])
        >>> fb.RocPlot(scored, label="meta_variant_type", positive="Single Missense",
        ...            score=["meta_impact_score_0.7", "meta_impact_score_0.9"])
        """
        if control_col not in self.columns:
            raise ValueError(
                f"Column {control_col!r} not found; chain .variant_type() before .impact_scores()"
            )
        if not variances:
            raise ValueError("Pass at least one variance ratio")
        df, x = self._matrix()
        model, scores = _fit_pca(x)
        ratios = model.explained_variance_ratio_
        ks = {v: _n_components(ratios, v) for v in variances}
        pcs = [f"__pc_{i}" for i in range(max(ks.values()))]
        tmp = df.select(control_col).with_columns(
            pl.Series(name, scores[:, i]) for i, name in enumerate(pcs)
        )
        outputs = [f"{prefix}{v}" for v in variances]
        impact = tmp.select(
            _transforms.impact_score_expr(pcs[: ks[v]], control_col=control_col).alias(out)
            for v, out in zip(variances, outputs)
        )
        new = self._replace(df.drop(outputs, strict=False).hstack(impact))
        new.pca_explained_variance = _explained_variance(ratios, max(ks.values()), "X_")
        return new

    def save_pca_loadings(self, path: str | PathLike, **write_kw: Any) -> Self:
        """Write `pca_loadings` (from the last `pca` / `pca_reduce` call) to a parquet
        file, creating parent directories. Returns the profiles unchanged, so it can sit
        in a chain."""
        if self.pca_loadings is None:
            raise ValueError("No PCA loadings; chain .pca() or .pca_reduce() first")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pca_loadings.write_parquet(path, **write_kw)
        return self

    def umap(
        self,
        *,
        n_components: int = 2,
        n_neighbors: int = 30,
        min_dist: float = 0.1,
        metric: str = "cosine",
        seed: int = 42,
        output_cols: Sequence[str] | None = None,
        **umap_kw: Any,
    ) -> Self:
        """Add an ``n_components``-dimensional UMAP embedding of the features, as
        ``output_cols`` (default ``meta_notebook_umap_1 .. meta_notebook_umap_<n>``).
        Needs ``fisseqborn[umap]``."""
        if output_cols is None:
            output_cols = [f"meta_notebook_umap_{i + 1}" for i in range(n_components)]
        elif len(output_cols) != n_components:
            raise ValueError(
                f"output_cols has {len(output_cols)} names but n_components={n_components}"
            )
        umap = _transforms.require_extra("umap", "umap")
        df, x = self._matrix()
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=seed,
            **umap_kw,
        )
        coords = reducer.fit_transform(x)
        return self._replace(
            df.with_columns(pl.Series(name, coords[:, i]) for i, name in enumerate(output_cols))
        )

    def cluster(
        self,
        method: Literal["leiden", "kmeans"] = "leiden",
        *,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        n_clusters: int | None = None,
        seed: int = 0,
        output_col: str = "meta_cluster_idx",
    ) -> Self:
        """Cluster the profiles and add the cluster ids as strings (``"0"``, ``"1"``, ...).

        ``"leiden"`` runs Leiden community detection on the ``n_neighbors``-nearest
        neighbor graph at the given ``resolution`` and needs ``fisseqborn[cluster]``.
        ``"kmeans"`` needs ``n_clusters``.
        """
        df, x = self._matrix()
        labels = _transforms.cluster_labels(
            x,
            method=method,
            n_neighbors=n_neighbors,
            resolution=resolution,
            n_clusters=n_clusters,
            seed=seed,
        )
        return self._replace(
            df.with_columns(pl.Series(output_col, labels.astype(str), dtype=pl.String))
        )

    def distinguishability(
        self,
        ovwt: "OvwtScores",
        *,
        score: str = "auroc_pooled",
        reference: Literal["synonymous", "all"] | None = "synonymous",
        rescale: bool = True,
        output_col: str = "meta_distinguishability_score",
    ) -> Self:
        """Join a per-variant distinguishability score from OvWT results.

        Per-batch results (`OvwtScores.from_pipeline`) are batch-corrected against
        ``reference`` (see `OvwtScores.correct`; ``None`` skips the correction, and
        ``rescale=False`` uses a plain z-score, as the old global stage did) and then
        take the median across batches. Results that are already per variant (after
        `OvwtScores.per_variant`, or the ``global_scores.parquet`` that
        ``fisseqborn-global`` writes, e.g. ``score="meta_median_auroc_pooled"``) are
        joined as they are, reading ``score``.
        """
        if score not in ovwt.columns and "test_auroc" in ovwt.columns:
            raise ValueError(
                f"Score column {score!r} not found; these look like legacy OvWT results, "
                "pass score='test_auroc'"
            )
        per_variant = ovwt
        if ovwt.batch_col in ovwt.columns:
            if reference is not None:
                per_variant = per_variant.correct(score, reference=reference, rescale=rescale)
                score = f"{score}_corrected"
            per_variant = per_variant.per_variant(score)
        per_variant._require(per_variant.variant_col, score)
        right = per_variant.lazy().select(
            pl.col(per_variant.variant_col).alias(self.variant_col),
            pl.col(score).alias(output_col),
        )
        return self._replace(
            self._lf.drop(output_col, strict=False).join(
                right, on=self.variant_col, how="left", maintain_order="left"
            )
        )

    # ----- inspection -----------------------------------------------------------------

    def feature_info(self) -> pl.DataFrame:
        """One row per feature column, parsed from its CellProfiler-style name.

        Columns: ``feature`` (the column name), ``base`` (without the statistic suffix),
        ``statistic`` (e.g. ``"median"``, null if unrecognized), ``compartment`` (e.g.
        ``"Nuclei"``, ``"Cytoplasm"``, null for whole-cell features), ``category`` (the
        first of `CATEGORIES` in the name) and ``channels`` (e.g. ``"CH1+CH3"``).
        """
        return feature_info(self.features)


def feature_info(columns: Iterable[str]) -> pl.DataFrame:
    """Parse feature column names; see `Profiles.feature_info`."""
    rows = []
    for col in columns:
        base, _, last = col.rpartition("_")
        statistic = last if base and last in STATISTICS else None
        base = base if statistic else col
        tokens = base.split("_")
        compartment = tokens[1] if len(tokens) > 2 and tokens[0] in _COMPARTMENT_PREFIXES else None
        category = next((c for c in CATEGORIES if c in base), None)
        channels = sorted(set(_CHANNEL_RE.findall(base)), key=lambda c: int(c[2:]))
        rows.append(
            {
                "feature": col,
                "base": base,
                "statistic": statistic,
                "compartment": compartment,
                "category": category,
                "channels": "+".join(channels) or None,
            }
        )
    schema = {
        k: pl.String
        for k in ("feature", "base", "statistic", "compartment", "category", "channels")
    }
    return pl.DataFrame(rows, schema=schema)


def _output_metadata(
    path: pathlib.Path, metadata: bool | Sequence[str], variant_col: str
) -> pl.LazyFrame:
    """The per-variant ``meta_`` columns of a batch's ``output.parquet``."""
    lf = _pipeline.scan(path)
    if metadata is True:
        columns = [c for c in lf.collect_schema().names() if c.startswith("meta_")]
        columns = [c for c in columns if c != variant_col]
    else:
        columns = [metadata] if isinstance(metadata, str) else list(metadata)
        _data.require_columns(pl.DataFrame(schema=lf.collect_schema()), variant_col, *columns)
    return lf.select(variant_col, *columns).unique(variant_col, keep="first", maintain_order=True)


def _fit_pca(x: np.ndarray) -> tuple[Any, np.ndarray]:
    """Every principal component of ``x``: the fitted model and the scores."""
    from sklearn.decomposition import PCA

    model = PCA(n_components=None)
    return model, model.fit_transform(x)


def _n_components(ratios: np.ndarray, variance: float) -> int:
    """The fewest components that together explain ``variance`` of the total."""
    if not 0 < variance <= 1:
        raise ValueError(f"variance must be in (0, 1], got {variance}")
    k = int(np.searchsorted(np.cumsum(ratios), variance - 1e-12)) + 1
    return max(1, min(k, len(ratios)))


def _explained_variance(
    ratios: np.ndarray, k: int, prefix: str, *, start: int = 0
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "component": [f"{prefix}{i + start}" for i in range(len(ratios))],
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": np.cumsum(ratios),
            "retained": np.arange(len(ratios)) < k,
        }
    )


def _loadings(
    model: Any, names: list[str], features: list[str], k: int | None = None
) -> pl.DataFrame:
    k = len(names) if k is None else k
    return pl.DataFrame(
        {
            "component": names,
            "explained_variance_ratio": model.explained_variance_ratio_[:k],
            **{f: model.components_[:k, j] for j, f in enumerate(features)},
        }
    )
