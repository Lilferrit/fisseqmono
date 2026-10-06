"""OVWT_BATCHWISE's entry point (``python -m fisseq_common.stages.ovwt``): it rebuilds the
normalized cells from the filter stage's outputs and scores them. The command-line tests moved
from the data pipeline's ``ovwt`` wrapper.

Fixture sizing note: the inner 80/20 train/calibration split runs *inside* each outer fold, so
a ``(barcode, is_wt)`` stratum wants comfortably more than ``n_folds`` members for the folds to
carry a real signal (see ``test_ovwt.py``). These fixtures use ``n_folds`` <= 3 with 15 cells
per barcode for that reason.
"""

import pathlib
import pickle
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_common.stages.ovwt as m
from fisseq_common.schema import EMBEDDING_SELECTOR
from fisseq_common.stages.filter import FilterParams, run_filter
from fisseq_common.stages.ovwt import CV_MODE_BARCODE_HOLDOUT

LABEL = "meta_aa_changes"
WT = "WT"


def _cells(
    barcodes_per_variant: int = 2, per_barcode: int = 15, seed: int = 0
) -> pl.DataFrame:
    """
    A QC_FILTER-shaped cell table with a real signal: variants sit away from WT on
    Intensity_Mean, so the classifier has something to find.
    """
    rng = np.random.default_rng(seed)
    rows = {LABEL: [], "meta_barcode": [], "Intensity_Mean": [], "Texture_Var": []}
    groups = [(WT, f"wt_bc{b}", 0.0) for b in range(3)] + [
        (variant, f"{variant}_bc{b}", 3.0 * offset)
        for offset, variant in enumerate(["M1K", "A1A"], start=1)
        for b in range(barcodes_per_variant)
    ]
    for label, barcode, centre in groups:
        rows[LABEL] += [label] * per_barcode
        rows["meta_barcode"] += [barcode] * per_barcode
        rows["Intensity_Mean"] += rng.normal(centre, 0.3, per_barcode).tolist()
        rows["Texture_Var"] += rng.random(per_barcode).tolist()
    return pl.DataFrame(rows).with_columns(
        pl.int_range(pl.len(), dtype=pl.Int64).alias("meta_cell_index"),
        pl.lit(None, dtype=pl.String).alias("meta_variant_tag"),
    )


def _write_inputs(tmp_path: pathlib.Path, cells: pl.DataFrame) -> list[str]:
    """Write ``cells`` and run the filter stage on it; return the three input overrides."""
    cells_path = tmp_path / "filtered_cells.parquet"
    cells.write_parquet(cells_path)
    norm_dir = tmp_path / "normalization"
    norm_dir.mkdir()
    run_filter(
        FilterParams(
            output_dir=str(norm_dir),
            cells_file=str(cells_path),
            qc_passed_file=str(cells_path),
        )
    )
    return [
        f"cells_file={cells_path}",
        f"filtered_keys_file={norm_dir / 'filtered_keys.parquet'}",
        f"normalizer_file={norm_dir / 'normalizer.parquet'}",
    ]


def _run_cli(tmp_path: pathlib.Path, inputs: list[str], *args: str) -> None:
    out = tmp_path / "out"
    out.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_common.stages.ovwt",
            "output_dir=.",
            *inputs,
            f"label_column={LABEL}",
            "min_cells=null",
            "downsample_wt=false",
            "random_seed=0",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=out,
    )
    assert result.returncode == 0, result.stderr


def _models(tmp_path: pathlib.Path) -> dict:
    with open(tmp_path / "out" / "models.pkl", "rb") as f:
        return pickle.load(f)


def test_cli_end_to_end(tmp_path: pathlib.Path):
    _run_cli(tmp_path, _write_inputs(tmp_path, _cells()), "n_folds=3")

    for name in ("results.parquet", "cell_scores.parquet", "models.pkl"):
        assert (tmp_path / "out" / name).exists(), f"{name} missing"
    results = pl.read_parquet(tmp_path / "out" / "results.parquet")
    assert set(results[LABEL]) == {"M1K", "A1A"}


@pytest.mark.parametrize("n_folds", [2, 3])
def test_cli_respects_n_folds(tmp_path: pathlib.Path, n_folds: int):
    _run_cli(tmp_path, _write_inputs(tmp_path, _cells()), f"n_folds={n_folds}")
    assert all(len(folds) == n_folds for folds in _models(tmp_path).values())


@pytest.mark.parametrize(
    "n_folds_arg,expected_folds", [("99", 3), ("null", 3), ("2", 2)]
)
def test_cli_barcode_holdout(
    tmp_path: pathlib.Path, n_folds_arg: str, expected_folds: int
):
    """cv_mode and n_folds both reach the CLI.

    ``n_folds=null`` is the spelling the OVWT_BATCHWISE module emits for a null
    ``params.ovwt_n_folds`` (Groovy renders null as the literal ``null``), so this case pins
    the Hydra parse that module depends on.
    """
    _run_cli(
        tmp_path,
        _write_inputs(tmp_path, _cells(barcodes_per_variant=3)),
        f"cv_mode={CV_MODE_BARCODE_HOLDOUT}",
        f"n_folds={n_folds_arg}",
    )
    models = _models(tmp_path)
    assert models
    assert all(len(folds) == expected_folds for folds in models.values())
    cell_scores = pl.read_parquet(tmp_path / "out" / "cell_scores.parquet")
    assert cell_scores["score"].null_count() == 0


def test_main_passes_the_named_feature_selector(tmp_path: pathlib.Path):
    """``feature_selector="embeddings"`` (the embeddings track) reaches run_ovwt."""
    inputs = dict(arg.split("=", 1) for arg in _write_inputs(tmp_path, _cells()))
    cfg = m.OvwtConfig(
        output_dir=str(tmp_path / "out"), feature_selector="embeddings", **inputs
    )
    with (
        patch("fisseq_common.stages.config.setup_logging"),
        patch("fisseq_common.stages.ovwt.run_ovwt") as run_ovwt,
    ):
        m.main.__wrapped__(OmegaConf.structured(cfg))
    cells_lf, _, selector = run_ovwt.call_args.args
    assert selector.meta.eq(EMBEDDING_SELECTOR)
    # the rebuilt cells: wildtype-normalized, every QC-passed cell once
    cells = cells_lf.collect()
    assert cells.height == _cells().height
    wt = cells.filter(pl.col(LABEL) == WT)["Intensity_Mean"]
    assert wt.mean() == pytest.approx(0.0, abs=1e-9)
