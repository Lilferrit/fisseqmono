"""Unit tests for the k-fold cross-validated OVWT_BATCHWISE stage.

Fixture sizing note: the inner 80/20 train/calibration split
(``split_indices_stratified``) runs *inside* each outer fold, so a
``(barcode, is_wt)`` stratum wants comfortably more than ``n_folds`` members
for the folds to carry a real signal. Undersized fixtures used to send every
variant into ``ovwt_batchwise``'s per-variant ``except`` branch, and the test
would then pass against empty output while asserting nothing. These fixtures
use ``n_folds=3`` with ~15 cells per barcode for that reason;
``test_normal_variant_survives`` guards the failure mode directly.
"""

import pathlib
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from fisseq_data_pipeline.ovwt import (
    CV_MODE_BARCODE_HOLDOUT,
)

LABEL = "meta_aa_changes"


WT = "WT"


def _cells(
    variants: dict[str, int] | None = None,
    wt_barcodes: int = 3,
    cells_per_wt_barcode: int = 15,
    barcodes_per_variant: int = 2,
    seed: int = 0,
) -> pl.DataFrame:
    """
    Cell-level frame with a real signal: variants sit away from WT on
    Intensity_Mean, so the classifier has something to find.

    ``variants`` maps a variant label to its per-barcode cell count.
    """
    variants = variants or {"M1K": 15, "A1A": 15}
    rng = np.random.default_rng(seed)
    rows = {LABEL: [], "meta_barcode": [], "Intensity_Mean": [], "Texture_Var": []}

    for b in range(wt_barcodes):
        n = cells_per_wt_barcode
        rows[LABEL] += [WT] * n
        rows["meta_barcode"] += [f"wt_bc{b}"] * n
        rows["Intensity_Mean"] += rng.normal(0.0, 0.3, n).tolist()
        rows["Texture_Var"] += rng.random(n).tolist()

    for offset, (variant, per_barcode) in enumerate(variants.items(), start=1):
        for b in range(barcodes_per_variant):
            rows[LABEL] += [variant] * per_barcode
            rows["meta_barcode"] += [f"{variant}_bc{b}"] * per_barcode
            rows["Intensity_Mean"] += rng.normal(
                3.0 * offset, 0.3, per_barcode
            ).tolist()
            rows["Texture_Var"] += rng.random(per_barcode).tolist()

    return pl.DataFrame(rows)


def test_cli_end_to_end(tmp_path: pathlib.Path):
    cells_path = tmp_path / "normalized.parquet"
    _cells().write_parquet(cells_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_data_pipeline.ovwt",
            "output_dir=.",
            f"input_file={cells_path}",
            f"label_column={LABEL}",
            "n_folds=3",
            "min_cells=null",
            "downsample_wt=false",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    for name in ("results.parquet", "cell_scores.parquet", "models.pkl"):
        assert (tmp_path / name).exists(), f"{name} missing"

    results = pl.read_parquet(tmp_path / "results.parquet")
    assert set(results.get_column(LABEL).to_list()) == {"M1K", "A1A"}


@pytest.mark.parametrize("n_folds", [2, 3])
def test_cli_respects_n_folds(tmp_path: pathlib.Path, n_folds: int):
    import pickle

    cells_path = tmp_path / "normalized.parquet"
    _cells().write_parquet(cells_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_data_pipeline.ovwt",
            "output_dir=.",
            f"input_file={cells_path}",
            f"label_column={LABEL}",
            f"n_folds={n_folds}",
            "min_cells=null",
            "downsample_wt=false",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    with open(tmp_path / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert all(len(folds) == n_folds for folds in models.values())


@pytest.mark.parametrize(
    "n_folds_arg,expected_folds", [("99", 3), ("null", 3), ("2", 2)]
)
def test_cli_barcode_holdout(
    tmp_path: pathlib.Path, n_folds_arg: str, expected_folds: int
):
    """cv_mode and n_folds both reach the CLI.

    ``n_folds=null`` is the spelling ``modules/local/ovwt_batchwise.nf`` emits
    for a null ``params.ovwt_n_folds`` (Groovy renders null as the literal
    ``null``), so this case pins the Hydra parse that module depends on.
    """
    import pickle

    cells_path = tmp_path / "normalized.parquet"
    _cells(barcodes_per_variant=3).write_parquet(cells_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_data_pipeline.ovwt",
            "output_dir=.",
            f"input_file={cells_path}",
            f"label_column={LABEL}",
            f"cv_mode={CV_MODE_BARCODE_HOLDOUT}",
            f"n_folds={n_folds_arg}",
            "min_cells=null",
            "downsample_wt=false",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr

    with open(tmp_path / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert models
    assert all(len(folds) == expected_folds for folds in models.values())

    cell_scores = pl.read_parquet(tmp_path / "cell_scores.parquet")
    assert cell_scores.get_column("score").null_count() == 0
