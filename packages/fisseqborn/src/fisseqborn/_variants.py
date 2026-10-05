"""Polars expressions that derive per-variant annotations from ``meta_aa_changes`` labels.

Labels look like ``"A12V"`` (substitution), ``"A12A"`` (synonymous), ``"A12fs"``
(frameshift), ``"A12-"`` / ``"A12-|L13-"`` (codon deletions), ``"A12*"`` / ``"A12X"``
(nonsense) and ``"WT"``. Multi-codon changes are joined with ``|``; a ``:<tag>`` suffix
(e.g. ``"A12A:downsampled"``) marks a pseudo-variant resampled from another one.
"""

import re
from collections.abc import Mapping, Sequence

import polars as pl

VARIANT_TYPES: tuple[str, ...] = (
    "Frameshift",
    "3nt Deletion",
    "Nonsense",
    "WT",
    "Synonymous",
    "Single Missense",
    "Other",
)

_SUBSTITUTION_RE = re.compile(r"([A-Z])(\d+)([A-Z])")


def classify_variant(label: str) -> str:
    """Classify a single variant label; the reference that `variant_type_expr` vectorizes.

    Returns one of `VARIANT_TYPES`. Frameshifts win over everything else, then codon
    deletions (one codon, or two adjacent codons, count as a "3nt Deletion"; anything
    longer is "Other"), then nonsense (``X`` or ``*``), then ``WT``; a remaining label
    that starts with a substitution is "Synonymous" when the amino acid is unchanged and
    "Single Missense" otherwise.
    """
    if "fs" in label:
        return "Frameshift"
    if label.endswith("-"):
        codons = label.split("|")
        if len(codons) == 1:
            return "3nt Deletion"
        if len(codons) == 2 and int(codons[0][1:-1]) == int(codons[1][1:-1]) - 1:
            return "3nt Deletion"
        return "Other"
    if "X" in label or "*" in label:
        return "Nonsense"
    if "WT" in label:
        return "WT"
    match = _SUBSTITUTION_RE.match(label)
    if match is None:
        return "Other"
    return "Synonymous" if match.group(1) == match.group(3) else "Single Missense"


def variant_type_expr(column: str) -> pl.Expr:
    """Vectorized `classify_variant` over ``column`` (null labels stay null)."""
    label = pl.col(column)
    codons = label.str.split("|")

    def codon_position(i: int) -> pl.Expr:
        # "A12-" -> 12, the same slice as ``codon[1:-1]``
        codon = codons.list.get(i, null_on_oob=True)
        return codon.str.slice(1, codon.str.len_chars() - 2).cast(
            pl.Int64, strict=False
        )

    adjacent_pair = (codons.list.len() == 2) & (
        codon_position(0) == codon_position(1) - 1
    )
    ref = label.str.extract(r"^([A-Z])\d+[A-Z]", 1)
    alt = label.str.extract(r"^[A-Z]\d+([A-Z])", 1)
    return (
        pl.when(label.is_null())
        .then(pl.lit(None, dtype=pl.String))
        .when(label.str.contains("fs", literal=True))
        .then(pl.lit("Frameshift"))
        .when(label.str.ends_with("-"))
        .then(
            pl.when((codons.list.len() == 1) | adjacent_pair.fill_null(False))
            .then(pl.lit("3nt Deletion"))
            .otherwise(pl.lit("Other"))
        )
        .when(
            label.str.contains("X", literal=True)
            | label.str.contains("*", literal=True)
        )
        .then(pl.lit("Nonsense"))
        .when(label.str.contains("WT", literal=True))
        .then(pl.lit("WT"))
        .when(ref.is_null())
        .then(pl.lit("Other"))
        .when(ref == alt)
        .then(pl.lit("Synonymous"))
        .otherwise(pl.lit("Single Missense"))
    )


def control_expr(variant_type: pl.Expr, column: str) -> pl.Expr:
    """The pipeline's control rule: a synonymous variant without a ``:<tag>`` suffix."""
    return (variant_type == "Synonymous") & ~pl.col(column).str.contains(
        ":", literal=True
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
