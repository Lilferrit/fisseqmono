"""Regression against upstream ExCALIBR's published BRCA1 (Findlay et al. 2018) calibration.

An exact match across implementations is not expected (independent random streams, the
log-likelihood fix described in `fisseqborn.calibration._mixture`, fewer bootstraps here
than upstream's run), so the checks have tolerances: the prior within 0.05, every score's
points within one of upstream's, and out-of-bag evidence agreeing with the ClinVar controls
about as well as Zeiberg et al. report for BRCA1 (Supp. Table S2).
"""

import csv
import json
import pathlib

import numpy as np
import pytest

from fisseqborn.calibration import _core
from fisseqborn.calibration._result import assign_points

DATA = pathlib.Path(__file__).parent / "data" / "excalibr"
N_BOOTSTRAP = 100


def _read(name: str) -> tuple[np.ndarray, np.ndarray]:
    with open(DATA / name, newline="") as f:
        rows = list(csv.DictReader(f))
    scores = np.array([float(r["score"]) for r in rows])
    member = np.zeros((len(rows), 4), dtype=bool)
    for i, r in enumerate(rows):
        for code in r["sample_assignments"].split(","):
            if code.strip():
                member[i, int(code)] = True
    return scores, member


@pytest.fixture(scope="module")
def brca1():
    scores, member = _read("brca_findlay_example.csv")
    cal, oob = _core.fit_calibration(
        scores, member, n_components=2, n_bootstrap=N_BOOTSTRAP, seed=0
    )
    return scores, member, cal, oob


def test_matches_upstream_calibration(brca1):
    scores, _, cal, _ = brca1
    reference = json.loads((DATA / "BRCA1_Findlay_2018.json").read_text())
    ref_ranges = {int(k): v for k, v in reference["point_ranges"].items()}
    assert cal.direction == "lower_pathogenic"
    assert not reference["scoreset_flipped"]
    assert cal.prior == pytest.approx(reference["prior"], abs=0.05)
    grid = np.linspace(scores.min(), scores.max(), 2000)
    ours, theirs = cal.points(grid), assign_points(grid, ref_ranges)
    assert np.abs(ours - theirs).max() <= 1
    assert ours.max() >= 6 and ours.min() <= -5


def test_out_of_bag_agrees_with_clinvar(brca1):
    _, member, _, oob = brca1
    pathogenic, benign = oob[member[:, 0]], oob[member[:, 1]]
    pathogenic, benign = pathogenic[~np.isnan(pathogenic)], benign[~np.isnan(benign)]
    correct = (pathogenic > 0).sum() + (benign < 0).sum()
    determinate = (pathogenic != 0).sum() + (benign != 0).sum()
    # Zeiberg et al. report 97.9% across genes; BRCA1's controls are well separated.
    assert correct / determinate >= 0.95
    assert determinate / (len(pathogenic) + len(benign)) >= 0.75


@pytest.mark.parametrize(
    ("name", "mode"),
    [
        ("brca_findlay_PU_example.csv", "positive_unlabeled"),
        ("brca_findlay_NU_example.csv", "negative_unlabeled"),
    ],
)
def test_one_class_examples(name, mode):
    scores, member = _read(name)
    cal, _ = _core.fit_calibration(
        scores, member, n_components=2, n_bootstrap=40, out_of_bag=False, seed=0
    )
    assert cal.mode == mode
    assert cal.direction == "lower_pathogenic"
    assert 0.01 < cal.prior < 0.6
    assert cal.points(np.array([scores.min()]))[0] > 0
    assert cal.points(np.array([scores.max()]))[0] < 0
