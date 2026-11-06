"""What the info popup says about the loaded data: the run, its batches, the structures and
the annotations."""

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass

from .data import CLINVAR_CLASS, CLINVAR_CLASSES, VARIANT_TYPE, ClinvarCalls, RunScores
from .regions import Track
from .structure import Structure

#: Columns of `DataInfo.structures`.
STRUCTURE_COLUMNS = ("Structure", "File", "Chain", "Residues", "pLDDT", "Offset")


@dataclass(frozen=True)
class DataInfo:
    """The info popup's content, as plain values."""

    #: Section title -> (label, value) rows.
    sections: dict[str, list[tuple[str, str]]]
    #: (batch, OvWT result rows), in the run's order.
    batches: list[tuple[str, int | None]]
    #: One row per structure, `STRUCTURE_COLUMNS`.
    structures: list[tuple[str, ...]]


def _residues(structure: Structure) -> str:
    residues = sorted(structure.residues)
    return f"{residues[0]}–{residues[-1]} ({len(residues)})" if residues else "none"


def data_info(
    run: RunScores | None,
    structures: Sequence[Structure],
    *,
    plddt: Sequence[bool],
    plddt_forced: bool,
    labels: Sequence[str] | None = None,
    offsets: Sequence[int] | None = None,
    params_source: pathlib.Path | None = None,
    clinvar: ClinvarCalls | None = None,
    tracks: Sequence[Track] = (),
) -> DataInfo:
    """Describe ``run`` (``None``: no scores loaded), the structures and the annotations.

    Parameters
    ----------
    plddt : Sequence[bool]
        Per structure, whether the B-factors are treated as pLDDT.
    plddt_forced : bool
        Whether that was set by ``--plddt`` rather than detected.
    """
    labels = labels or [s.path.name for s in structures]
    offsets = offsets or [0] * len(structures)
    how = " (--plddt)" if plddt_forced else ""
    structure_rows = [
        (
            label,
            str(s.path),
            s.chain + (f" (of {', '.join(s.chains)})" if len(s.chains) > 1 else ""),
            _residues(s),
            ("yes" if p else "no") + how,
            str(offset),
        )
        for s, label, p, offset in zip(structures, labels, plddt, offsets)
    ]
    sections: dict[str, list[tuple[str, str]]] = {}
    if params_source is not None:
        sections["Launch"] = [("Parameters file", str(params_source))]
    if run is None:
        sections["Run"] = [("Scores", "none loaded")]
        batches: list[tuple[str, int | None]] = []
    else:
        by_type = (
            run.table(run.scores[0], True)
            .group_by(VARIANT_TYPE)
            .len()
            .sort("len", descending=True)
            .rows()
        )
        sections["Run"] = [
            ("Source", run.source),
            ("Layout", run.layout or "detected"),
            ("Track", run.track or "—"),
            ("Variant column", run.variant_col or "—"),
            ("Batches", str(len(run.batches))),
            ("Variants", f"{sum(n for _, n in by_type):,}"),
            *[(f"  {kind}", f"{n:,}") for kind, n in by_type if kind is not None],
            ("Scores", ", ".join(run.scores)),
            ("Pooling", "median across batches, batch-corrected or raw"),
        ]
        batches = [(b, run.batch_rows.get(b)) for b in run.batches]
    annotations = []
    if clinvar is not None:
        counts = dict(clinvar.table.group_by(CLINVAR_CLASS).len().rows())
        annotations += [
            ("ClinVar", clinvar.source),
            ("ClinVar release", clinvar.release or "—"),
            *[
                (f"  {name}", f"{counts.get(key, 0):,}")
                for key, (name, _) in CLINVAR_CLASSES.items()
            ],
        ]
    annotations += [
        (f"Regions: {t.label or 'unnamed'}", ", ".join(t.regions)) for t in tracks
    ]
    if annotations:
        sections["Annotations"] = annotations
    return DataInfo(sections, batches, structure_rows)
