"""Variant classification from ``meta_aa_changes`` labels, in Python and as Polars expressions.

Labels look like ``"A12V"`` (substitution), ``"A12A"`` (synonymous), ``"A12fs"``
(frameshift), ``"A12-"`` / ``"A12-|L13-"`` (codon deletions), ``"A12*"`` / ``"A12X"``
(nonsense) and ``"WT"``. Multi-codon changes are joined with ``|``. A ``:<tag>`` suffix (e.g.
the ``:downsampled-half`` pseudo-variants the data pipeline's QC_FILTER adds) marks a variant
resampled from another one; it is stripped before classification.

:func:`classify_variant` is the definition. :func:`variant_type_expr` is the same rule
vectorized for Polars, and :func:`control_expr` the control rule both pipelines and fisseqborn
use: a synonymous variant without a tag.
"""

import re

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


def classify_variant(v: str) -> str:
    """
    Classify a variant label string into a biological category.

    Any trailing ``:<tag>`` metadata suffix (e.g. ``"M1K:downsampled-half"``)
    is stripped before classification.

    Parameters
    ----------
    v : str
        Variant label string (e.g. ``"A123G"``, ``"A123fs"``, ``"WT"``).

    Returns
    -------
    str
        One of: ``"Frameshift"``, ``"3nt Deletion"``, ``"Nonsense"``,
        ``"WT"``, ``"Synonymous"``, ``"Single Missense"``, or ``"Other"``.
    """
    v = v.split(":", 1)[0]
    if "fs" in v:
        return "Frameshift"
    if v.endswith("-"):
        parts = v.split("|")
        n = len(parts)
        if n == 1:
            return "3nt Deletion"
        if n == 2 and int(parts[0][1:-1]) == int(parts[1][1:-1]) - 1:
            return "3nt Deletion"
        return "Other"
    if "X" in v or "*" in v:
        return "Nonsense"
    if "WT" in v:
        return "WT"
    m = _SUBSTITUTION_RE.match(v)
    if m is None:
        return "Other"
    return "Synonymous" if m.group(1) == m.group(3) else "Single Missense"


def variant_type_expr(column: str) -> pl.Expr:
    """:func:`classify_variant` over ``column`` as a Polars expression.

    Differs from it only where the Python version raises: a null label gives null, and a
    two-codon deletion with a non-integer position gives ``"Other"``.
    """
    label = pl.col(column).str.split(":").list.first()
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
    """The control rule: a synonymous variant without a ``:<tag>`` suffix."""
    return (variant_type == "Synonymous") & ~pl.col(column).str.contains(
        ":", literal=True
    )
