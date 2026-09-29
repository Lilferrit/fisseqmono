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
from typing import Any, Self

import polars as pl

from . import _data, _variants, fisseq

_META = "meta_"


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
    def read(cls, path: str | PathLike, **kw: Any) -> Self:
        """Lazily read a parquet file (or glob). ``kw`` is passed to the constructor."""
        return cls(pl.scan_parquet(path), **kw)

    # ----- chaining -------------------------------------------------------------------

    def _replace(self, data: pl.LazyFrame | pl.DataFrame) -> Self:
        new = copy.copy(self)
        new._lf = data.lazy()
        new._df = data if isinstance(data, pl.DataFrame) else None
        return new

    def _require(self, *columns: str | None) -> None:
        _data.require_columns(pl.DataFrame(schema=self.schema), *columns)

    def pipe(
        self, function: Callable[..., pl.LazyFrame | pl.DataFrame], *args: Any, **kw: Any
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
            exprs.append(_variants.control_expr(variant_type, self.variant_col).alias(control_col))
        return self.with_columns(exprs)

    def position(self, *, strict: bool = False, output_col: str = "meta_position") -> Self:
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
        return self.with_columns(_variants.region_expr(position, regions).alias(output_col))

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
            _variants.tile_expr(position, tiles, allow_multiple=allow_multiple).alias(output_col)
        )

    def clinvar(
        self,
        clinvar: "str | PathLike | pl.DataFrame | pl.LazyFrame",
        *,
        variant_type_col: str = "meta_variant_type",
        output_col: str = "meta_clinvar_annotation",
    ) -> Self:
        """Join ClinVar pathogenicity calls and build a combined annotation column.

        ``clinvar`` is the converted ClinVar table (a parquet path or a frame) with a
        ``variant`` column in the same ``"A12V"`` format and a
        ``clinvar_clinical_significance`` column. Only the first record per variant is
        used, and only when it contains "Pathogenic". Its columns are joined in with a
        ``meta_`` prefix.

        ``output_col`` holds the ClinVar call where there is one and the variant type
        otherwise, with a lone "Pathogenic" merged into `fisseq.PATHOGENIC`. Run
        `variant_type` first.
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
        table = (
            table.unique(subset=["variant"], keep="first", maintain_order=True)
            .filter(pl.col("clinvar_clinical_significance").str.contains("Pathogenic"))
            .rename(
                {c: f"{_META}{c}" for c in columns if c != "variant" and not c.startswith(_META)}
            )
        )
        significance = pl.col("meta_clinvar_clinical_significance")
        annotation = pl.coalesce(significance, pl.col(variant_type_col))
        return self._replace(
            self._lf.join(
                table,
                left_on=self.variant_col,
                right_on="variant",
                how="left",
                maintain_order="left",
            ).with_columns(
                pl.when(annotation == "Pathogenic")
                .then(pl.lit(fisseq.PATHOGENIC))
                .otherwise(annotation)
                .alias(output_col)
            )
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
