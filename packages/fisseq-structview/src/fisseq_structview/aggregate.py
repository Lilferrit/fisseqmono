"""Per-variant scores to one value per structure residue.

The steps run in this order: `filter_types` keeps the variant classes shown, `apply_offset`
maps score positions onto the structure's numbering, and `per_position` aggregates each
residue's variants with one of `METHODS`.
"""

from collections.abc import Sequence

import polars as pl

from .data import POSITION, SCORE, VARIANT, VARIANT_TYPE, WT

#: Aggregation methods, by name, with their dropdown labels.
METHODS: dict[str, str] = {
    "median": "Median",
    "mean": "Mean",
    "min": "Min",
    "max": "Max",
    "frac_deleterious": "Fraction deleterious",
}
DEFAULT_METHOD = "median"

#: Variant classes with a position that can be shown (labels of `fisseqborn`'s
#: ``variant_type``); multi-codon changes have no single position.
VARIANT_TYPES: tuple[str, ...] = ("Single Missense", "Nonsense", "Synonymous")
DEFAULT_TYPES: tuple[str, ...] = ("Single Missense",)

#: Default score above which a variant counts as deleterious: an AUROC this far above 0.5.
DEFAULT_THRESHOLD = 0.6

#: Columns of `per_position`'s output.
RESIDUE = "residue"
VALUE = "value"
N_VARIANTS = "n_variants"
DETAILS = "details"


def filter_types(df: pl.DataFrame, types: Sequence[str]) -> pl.DataFrame:
    """The rows whose variant class is one of ``types``."""
    return df.filter(pl.col(VARIANT_TYPE).is_in(list(types)))


def apply_offset(df: pl.DataFrame, offset: int) -> pl.DataFrame:
    """Add `RESIDUE`, the structure residue number: the score position plus ``offset``.

    Use a positive offset when the structure numbers residues higher than the scores (e.g. a
    construct whose position 1 is residue 25 of the model: ``offset=24``).
    """
    return df.with_columns((pl.col(POSITION) + int(offset)).alias(RESIDUE))


def _method_expr(method: str, threshold: float) -> pl.Expr:
    score = pl.col(SCORE)
    match method:
        case "median":
            return score.median()
        case "mean":
            return score.mean()
        case "min":
            return score.min()
        case "max":
            return score.max()
        case "frac_deleterious":
            return (score > threshold).mean()
    raise ValueError(f"Unknown aggregation {method!r}; choose from {list(METHODS)}")


def per_position(
    df: pl.DataFrame, method: str = DEFAULT_METHOD, threshold: float = DEFAULT_THRESHOLD
) -> pl.DataFrame:
    """One row per residue with a score, sorted by residue.

    Parameters
    ----------
    df : pl.DataFrame
        Per-variant rows with `RESIDUE` (see `apply_offset`). Rows without a position or a
        score are ignored.
    method : str
        One of `METHODS`. ``"frac_deleterious"`` is the fraction of the residue's variants
        scoring above ``threshold`` (a higher AUROC means more distinguishable from wild
        type).
    threshold : float
        The deleterious threshold.

    Returns
    -------
    pl.DataFrame
        `RESIDUE`, `WT` (the wild-type residue the labels name, the first one if they
        disagree), `VALUE`, `N_VARIANTS` and `DETAILS` (``"A12V 0.731"`` per variant, best
        first).
    """
    agg = _method_expr(method, threshold)
    return (
        df.filter(pl.col(RESIDUE).is_not_null() & pl.col(SCORE).is_not_null())
        .sort(SCORE, descending=True)
        .group_by(RESIDUE)
        .agg(
            pl.col(WT).drop_nulls().first().alias(WT),
            agg.cast(pl.Float64).alias(VALUE),
            pl.len().cast(pl.UInt32).alias(N_VARIANTS),
            pl.format("{} {}", pl.col(VARIANT), pl.col(SCORE).round(3)).alias(DETAILS),
        )
        .sort(RESIDUE)
    )
