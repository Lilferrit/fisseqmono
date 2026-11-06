"""Small synthetic structures and pipeline runs for the unit tests."""

import pathlib
from collections.abc import Mapping, Sequence

import polars as pl

THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
    "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
    "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}  # fmt: skip

#: The sequence of the test structure, residues 1-8.
SEQUENCE = "MKTAYIAK"


def pdb_text(
    sequence: str = SEQUENCE,
    *,
    start: int = 1,
    chain: str = "A",
    bfactors: Sequence[float] | None = None,
    per_atom_noise: bool = False,
    title: str | None = None,
) -> str:
    """Backbone atoms (N, CA, C, O) of ``sequence`` numbered from ``start``."""
    lines = [f"TITLE     {title}"] if title else []
    serial = 1
    for i, aa in enumerate(sequence):
        b = bfactors[i] if bfactors is not None else 90.0
        for j, atom in enumerate(("N", "CA", "C", "O")):
            bj = b + (j * 3.0 if per_atom_noise else 0.0)
            x, y, z = 3.8 * i + 0.5 * j, 0.3 * j, 0.0
            lines.append(
                f"ATOM  {serial:5d}  {atom:<3s} {THREE[aa]} {chain}{start + i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bj:6.2f}           {atom[0]}"
            )
            serial += 1
    lines += ["TER", "END"]
    return "\n".join(lines) + "\n"


def write_pdb(path: pathlib.Path, **kw) -> pathlib.Path:
    path.write_text(pdb_text(**kw))
    return path


def write_run(
    root: pathlib.Path, batches: Mapping[str, Mapping[str, float]]
) -> pathlib.Path:
    """``ovwt_batchwise/<batch>/results.parquet`` with every score column set to the
    batch's ``{label: auroc}`` value (``auroc_pooled`` 0.01 lower)."""
    for batch, scores in batches.items():
        d = root / "ovwt_batchwise" / batch
        d.mkdir(parents=True)
        labels = list(scores)
        values = [scores[v] for v in labels]
        pl.DataFrame(
            {
                "meta_aa_changes": labels,
                "auroc_pooled": [v - 0.01 for v in values],
                "auroc_median_barcode": values,
                "auroc_folds": [[v, v] for v in values],
                "auroc_median_fold": values,
                "meta_n_barcodes": [5] * len(labels),
                "meta_n_cells": [100] * len(labels),
            }
        ).write_parquet(d / "results.parquet")
    return root


#: Two batches over SEQUENCE: synonymous controls at 1-3 and missense variants at 2, 4, 5.
RUN = {
    "batch1": {
        "M1M": 0.50, "K2K": 0.52, "T3T": 0.48,
        "K2E": 0.70, "K2R": 0.60, "A4V": 0.90, "Y5F": 0.40, "A4*": 0.95,
    },
    "batch2": {
        "M1M": 0.49, "K2K": 0.51, "T3T": 0.50,
        "K2E": 0.72, "K2R": 0.58, "A4V": 0.86, "Y5F": 0.42, "A4*": 0.97,
    },
}  # fmt: skip
