"""Match aggregated positions to a structure's residues, and check that they line up.

The two numberings differ when the structure is a fragment, a construct, or numbered from a
different start; `match_positions` catches a wrong ``--offset`` two ways: no scored residue
in the structure at all, or wild-type residues in the variant labels that disagree with the
structure's sequence.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from .aggregate import RESIDUE, VALUE
from .data import WT

logger = logging.getLogger(__name__)

#: The largest share of scored residues whose wild type may disagree with the structure.
MAX_WT_MISMATCH = 0.1


class MappingError(ValueError):
    """Scores that don't line up with the structure's residue numbering."""


@dataclass(frozen=True)
class PositionMatch:
    """How a per-position table lines up with a structure chain.

    Attributes
    ----------
    values : dict[int, float]
        Structure residue -> aggregated value, for scored residues in the structure.
    unscored : tuple[int, ...]
        Structure residues without a score (drawn gray).
    outside : tuple[int, ...]
        Scored residues the structure doesn't have (not drawn).
    wt_mismatches : tuple[tuple[int, str, str], ...]
        ``(residue, label wild type, structure residue)`` where they differ.
    """

    values: dict[int, float]
    unscored: tuple[int, ...]
    outside: tuple[int, ...]
    wt_mismatches: tuple[tuple[int, str, str], ...]


def _span(numbers: list[int]) -> str:
    return f"{min(numbers)}-{max(numbers)}" if numbers else "none"


def match_positions(
    per_pos: pl.DataFrame,
    residues: Mapping[int, str],
    *,
    offset: int = 0,
    check_wt: bool = True,
    max_mismatch: float = MAX_WT_MISMATCH,
) -> PositionMatch:
    """Split ``per_pos`` (from `aggregate.per_position`) by whether the structure has each
    residue.

    Parameters
    ----------
    per_pos : pl.DataFrame
        `RESIDUE`, `VALUE` and optionally `WT`.
    residues : Mapping[int, str]
        The structure chain's residue number -> one-letter code (`Structure.residues`).
    offset : int
        The offset already applied, only for the error messages.
    check_wt : bool
        Compare the labels' wild-type residues with the structure's.
    max_mismatch : float
        Largest allowed share of compared residues that disagree.

    Raises
    ------
    MappingError
        If no scored residue is in the structure, or more than ``max_mismatch`` of the
        compared wild-type residues disagree.
    """
    rows = per_pos.select(
        RESIDUE, VALUE, WT if WT in per_pos.columns else pl.lit(None).alias(WT)
    ).rows()
    scored = [r for r, _, _ in rows]
    inside = [(r, v, wt) for r, v, wt in rows if r in residues]
    structure_span = _span(list(residues))
    if rows and not inside:
        raise MappingError(
            f"None of the {len(rows)} scored positions (residues {_span(scored)} after "
            f"offset {offset}) is in the structure chain, whose residues are "
            f"{structure_span}. Set the numbering offset (--offset: structure residue = "
            "score position + offset)."
        )
    outside = tuple(r for r in scored if r not in residues)
    if outside:
        logger.warning(
            "%d of %d scored positions (%s) are not in the structure chain (residues %s) "
            "and are not shown",
            len(outside),
            len(scored),
            _span(list(outside)),
            structure_span,
        )
    mismatches = tuple(
        (r, wt, residues[r])
        for r, _, wt in inside
        if wt is not None and residues[r] != "X" and wt != residues[r]
    )
    compared = sum(1 for _, _, wt in inside if wt is not None)
    if check_wt and compared and len(mismatches) / compared > max_mismatch:
        examples = ", ".join(
            f"{r}: label {wt}, structure {s}" for r, wt, s in mismatches[:5]
        )
        raise MappingError(
            f"The wild-type residue of {len(mismatches)} of {compared} scored positions "
            f"disagrees with the structure ({examples}, ...). The numbering offset "
            f"({offset}) is probably wrong, or the structure is of a different protein or "
            "isoform. Skip this check with --no-check-wt."
        )
    values = {r: v for r, v, _ in inside if v is not None}
    unscored = tuple(r for r in residues if r not in values)
    return PositionMatch(
        values=values, unscored=unscored, outside=outside, wt_mismatches=mismatches
    )
