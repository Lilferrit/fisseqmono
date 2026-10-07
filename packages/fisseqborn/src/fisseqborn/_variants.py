"""Polars expressions that derive per-variant annotations from ``meta_aa_changes`` labels.

Variant classification itself (``classify_variant``, ``variant_type_expr``, ``control_expr``)
is shared with the pipelines and lives in :mod:`fisseq_common.variant`; this module adds the
position, region and tile annotations the plots use.

Labels look like ``"A12V"`` (substitution), ``"A12A"`` (synonymous), ``"A12fs"``
(frameshift), ``"A12-"`` / ``"A12-|L13-"`` (codon deletions), ``"A12*"`` / ``"A12X"``
(nonsense) and ``"WT"``. Multi-codon changes are joined with ``|``; a ``:<tag>`` suffix
(e.g. ``"A12A:downsampled"``) marks a pseudo-variant resampled from another one.
"""

from collections.abc import Mapping, Sequence

import polars as pl

from fisseq_common.variant import (  # noqa: F401 (re-exported for the plots)
    VARIANT_TYPES,
    classify_variant,
    control_expr,
    variant_type_expr,
)


def position_expr(column: str, *, strict: bool = False) -> pl.Expr:
    """Amino-acid position of a label.

    ``strict=False`` reads the leading position of the first ``|`` codon, so
    frameshifts, deletions and nonsense variants get one too. ``strict=True`` only
    accepts single substitutions like ``"A12V"`` and gives null for everything else.
    """
    pattern = r"^[A-Za-z](\d+)[A-Za-z]$" if strict else r"^.(\d+)"
    source = pl.col(column) if strict else pl.col(column).str.split("|").list.first()
    return source.str.extract(pattern, 1).cast(pl.Int64)


def substitution_exprs(column: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """``(wt, position, mut)`` of a single substitution like ``"A12V"`` (``"A12A"`` for a
    synonymous one). Every other label (frameshifts, deletions, ``|`` multi-codon changes,
    ``:<tag>`` pseudo-variants, ``"WT"``) gets null in all three."""
    pattern = r"^([A-Z])(\d+)([A-Z])$"
    label = pl.col(column)
    return (
        label.str.extract(pattern, 1),
        label.str.extract(pattern, 2).cast(pl.Int64),
        label.str.extract(pattern, 3),
    )


def region_expr(position: pl.Expr, regions: Mapping[str, tuple[int, int]]) -> pl.Expr:
    """Name of the first region whose inclusive ``(first, last)`` range holds ``position``."""
    return pl.coalesce(
        [
            pl.when(position.is_between(first, last)).then(pl.lit(name))
            for name, (first, last) in regions.items()
        ]
        + [pl.lit(None, dtype=pl.String)]
    )


def tile_expr(
    position: pl.Expr, tiles: Sequence[tuple[str, int, int]], *, allow_multiple: bool
) -> pl.Expr:
    """Tile(s) covering ``position``; overlaps give the first tile, or all of them
    joined with ``","`` when ``allow_multiple``. Positions outside every tile are null."""
    matches = pl.concat_list(
        [
            pl.when(position.is_between(first, last)).then(pl.lit(name))
            for name, first, last in tiles
        ]
    ).list.drop_nulls()
    joined = matches.list.join(",") if allow_multiple else matches.list.first()
    return pl.when(matches.list.len() > 0).then(joined)
