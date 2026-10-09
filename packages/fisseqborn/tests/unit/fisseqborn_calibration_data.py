"""Synthetic control groups with a known answer, for the calibration tests.

Scores come from two normal components (skew 0): "abnormal" at -2 and "normal" at +1. The
pathogenic group is mostly abnormal, benign and synonymous mostly normal, and gnomAD is
``prior * pathogenic + (1 - prior) * benign``, so the true prior is ``prior``.
"""

import numpy as np
import polars as pl

ABNORMAL = (-2.0, 0.7)
NORMAL = (1.0, 0.7)
W_PATHOGENIC, W_BENIGN, W_SYNONYMOUS = 0.9, 0.05, 0.03


def _draw(rng: np.random.Generator, n: int, w_abnormal: float) -> np.ndarray:
    abnormal = rng.random(n) < w_abnormal
    return np.where(abnormal, rng.normal(*ABNORMAL, n), rng.normal(*NORMAL, n))


def true_log_lr(x: np.ndarray, benign_weight: float = W_BENIGN) -> np.ndarray:
    def pdf(mu: float, sd: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sd) ** 2) / sd

    a, n = pdf(*ABNORMAL), pdf(*NORMAL)
    f_p = W_PATHOGENIC * a + (1 - W_PATHOGENIC) * n
    f_b = benign_weight * a + (1 - benign_weight) * n
    return np.log(f_p) - np.log(f_b)


def groups(
    seed: int = 0,
    *,
    prior: float = 0.3,
    n_pathogenic: int = 120,
    n_benign: int = 120,
    n_gnomad: int = 500,
    n_synonymous: int = 0,
    n_unlabeled: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """``scores`` and ``member`` (pathogenic, benign, gnomAD, synonymous) arrays."""
    rng = np.random.default_rng(seed)
    w_gnomad = prior * W_PATHOGENIC + (1 - prior) * W_BENIGN
    parts = [
        (_draw(rng, n_pathogenic, W_PATHOGENIC), 0),
        (_draw(rng, n_benign, W_BENIGN), 1),
        (_draw(rng, n_gnomad, w_gnomad), 2),
        (_draw(rng, n_synonymous, W_SYNONYMOUS), 3),
        (_draw(rng, n_unlabeled, 0.5), None),
    ]
    scores = np.concatenate([p for p, _ in parts])
    member = np.zeros((len(scores), 4), dtype=bool)
    start = 0
    for values, group in parts:
        if group is not None:
            member[start : start + len(values), group] = True
        start += len(values)
    return scores, member


AA3 = {"A": "Ala", "V": "Val", "R": "Arg", "W": "Trp"}


def dataset_frames(
    seed: int = 0, *, prior: float = 0.3, n_each: int = 60, n_gnomad: int = 250
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """A scored dataset (one row per ``meta_aa_changes``), a gnomAD browser export and a
    ClinVar table for it.

    Variants ``A<i>V`` are missense; ``A<i>A`` synonymous. The first ``n_each`` missense
    variants are P/LP, the next ``n_each`` B/LB (in ClinVar and in the gnomAD export's
    ClinVar column); the next ``n_gnomad`` are in gnomAD only.
    """
    rng = np.random.default_rng(seed)
    w_gnomad = prior * W_PATHOGENIC + (1 - prior) * W_BENIGN
    n_missense = 2 * n_each + n_gnomad + 40
    missense = [f"A{i}V" for i in range(1, n_missense + 1)]
    scores = np.concatenate(
        [
            _draw(rng, n_each, W_PATHOGENIC),
            _draw(rng, n_each, W_BENIGN),
            _draw(rng, n_gnomad, w_gnomad),
            _draw(rng, 40, 0.5),
        ]
    )
    synonymous = [f"A{i}A" for i in range(1, n_each + 1)]
    data = pl.DataFrame(
        {
            "meta_aa_changes": missense + synonymous,
            "score": np.concatenate([scores, _draw(rng, n_each, W_SYNONYMOUS)]),
        }
    )
    labelled = missense[: 2 * n_each]
    classes = ["Pathogenic"] * n_each + ["Likely benign"] * n_each
    in_gnomad = missense[
        n_each : 2 * n_each + n_gnomad
    ]  # benign controls + gnomAD-only
    gnomad_rows = []
    for v in sorted(set(labelled) | set(in_gnomad), key=missense.index):
        pos = v[1:-1]
        gnomad_rows.append(
            {
                "Protein Consequence": f"p.Ala{pos}Val",
                "Filters - joint": "PASS" if v in in_gnomad else "AC0",
                "Allele Count": "3",
                "Allele Number": "1000",
                "ClinVar Germline Classification": classes[labelled.index(v)]
                if v in labelled
                else "",
                "spliceai_ds_max": "0.01",
            }
        )
    gnomad = pl.DataFrame(gnomad_rows)
    clinvar = pl.DataFrame(
        {
            "variant": labelled,
            "clinvar_clinical_significance": classes,
            "clinvar_review_status": ["criteria provided, single submitter"]
            * len(labelled),
        }
    )
    return data, gnomad, clinvar
