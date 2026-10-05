"""Tests for OVWT_BATCHWISE.

Fixture sizing note: the inner 80/20 train/calibration split
(``split_indices_stratified``, utils/xgbparams.py) runs *inside* each outer
fold, so a ``(barcode, is_wt)`` stratum wants comfortably more than
``n_folds`` members for the folds to carry a real signal. It no longer
raises on a singleton stratum (those rows go to the train half), but an
undersized fixture still trains on almost nothing. Fixtures further down use
``n_folds=3`` with ~15 cells per barcode for that reason.

The cv_mode / barcode-holdout / per-fold-AUROC / logging tests near the end
are ported from fisseq-data-pipeline's tests/unit/test_ovwt.py (its #75-#77,
#80), adapted to ``emb_*`` feature columns and ``OvwtEmbeddingConfig``.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer
from fisseq_embeddings_pipeline.ovwt import (
    CV_MODE_BARCODE_HOLDOUT,
    OvwtEmbeddingConfig,
    main,
)


def _cfg(**overrides) -> OvwtEmbeddingConfig:
    defaults = dict(
        output_dir="/tmp/out",
        embeddings_file="embeddings.parquet",
        filtered_keys_file="filtered_keys.parquet",
        normalizer_file="normalizer.parquet",
    )
    defaults.update(overrides)
    return OvwtEmbeddingConfig(**defaults)


def test_default_label_column():
    assert _cfg().label_column == "meta_aa_changes"


def test_default_wt_label():
    assert _cfg().wt_label == "WT"


def test_default_n_folds():
    assert _cfg().n_folds == 5


def test_default_calibrate():
    assert _cfg().calibrate is True


def test_default_min_cells():
    assert _cfg().min_cells == 250


def test_default_downsample_wt():
    assert _cfg().downsample_wt is True


def test_inherits_random_seed_default():
    assert _cfg().random_seed == 0


def test_no_random_state_field():
    """No stage-local random_state -- every stochastic step reads the
    shared AppConfig.random_seed instead."""
    assert not hasattr(_cfg(), "random_state")


def test_xgboost_sub_config_has_defaults():
    cfg = _cfg()
    assert cfg.xgboost.num_boost_round == 100
    assert cfg.xgboost.early_stopping_rounds == 5


def test_min_cells_can_be_disabled():
    assert _cfg(min_cells=None).min_cells is None


def _kfold_fixture_lf(
    wt_n: int = 15,
    variant_n: int = 15,
    n_variant_barcodes: int = 2,
    variant_label: str = "M1K",
    separable: bool = True,
) -> pl.LazyFrame:
    rng = np.random.default_rng(0)
    rows = []
    wt_center = 1.0
    for i in range(wt_n):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "emb_0000": wt_center + rng.normal(scale=0.05),
            }
        )
    variant_center = 0.0 if separable else wt_center
    for b in range(n_variant_barcodes):
        for i in range(variant_n):
            rows.append(
                {
                    "meta_aa_changes": variant_label,
                    "meta_barcode": f"bc_v{b}",
                    "emb_0000": variant_center + rng.normal(scale=0.05),
                }
            )
    return pl.DataFrame(rows).lazy()


WT = "WT"


def _write_cli_fixture(tmp_path: Path) -> "tuple[Path, Path, Path]":
    """Build embeddings.parquet/filtered_keys.parquet/normalizer.parquet the
    way FILTER_EMBEDDINGS would. Sized per the same rationale as
    _kfold_fixture_lf above: a handful of synonymous+untagged control cells
    (A1A) to fit the Normalizer, a normal-sized WT barcode, and 2 M1K
    barcodes each large enough to survive the k-fold + inner-split path at
    n_folds=3."""
    n_control = 10
    wt_n = 15
    variant_n = 15
    n_variant_barcodes = 2
    total = n_control + wt_n + n_variant_barcodes * variant_n

    rng = np.random.default_rng(3)
    aa_changes = ["A1A"] * n_control + ["WT"] * wt_n
    barcodes = [f"bc_ctrl{i}" for i in range(n_control)] + ["bc_wt"] * wt_n
    emb = list(rng.normal(loc=5.0, scale=0.3, size=n_control)) + list(
        rng.normal(loc=6.0, scale=0.3, size=wt_n)
    )
    for b in range(n_variant_barcodes):
        aa_changes += ["M1K"] * variant_n
        barcodes += [f"bc_v{b}"] * variant_n
        emb += list(rng.normal(loc=4.0, scale=0.3, size=variant_n))

    embeddings_df = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * total,
            "meta_well": ["well1"] * total,
            "meta_tile": ["tile0x0y"] * total,
            "meta_cell_index": list(range(total)),
            "meta_barcode": barcodes,
            "meta_aa_changes": aa_changes,
            "meta_edit_distance": [0] * total,
            "emb_0000": emb,
        }
    )
    qc_passed_df = embeddings_df.select(JOIN_KEYS)

    embeddings_path = tmp_path / "embeddings.parquet"
    embeddings_df.write_parquet(embeddings_path)

    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        embeddings_df.lazy(), qc_passed_df.lazy(), "meta_aa_changes"
    )
    filtered_keys_path = tmp_path / "filtered_keys.parquet"
    filtered_keys_lf.collect().write_parquet(filtered_keys_path)
    normalizer_path = tmp_path / "normalizer.parquet"
    normalizer.save(normalizer_path)

    return embeddings_path, filtered_keys_path, normalizer_path


def _run_ovwt(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.ovwt", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_ovwt(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        "n_folds=3",
        "min_cells=1",
    )
    assert result.returncode == 0, result.stderr

    results = pl.read_parquet(output_dir / "results.parquet")
    assert "M1K" in results["meta_aa_changes"].to_list()
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert row["meta_n_barcodes"] == 2
    assert 0.0 <= row["auroc_pooled"] <= 1.0

    cell_scores = pl.read_parquet(output_dir / "cell_scores.parquet")
    assert cell_scores["score"].null_count() == 0

    with open(output_dir / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert "M1K" in models
    assert len(models["M1K"]) == 3


def test_main_is_hydra_entry_point() -> None:
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(main)


def test_main_barcode_holdout_null_n_folds_via_cli(tmp_path: Path) -> None:
    """cv_mode and n_folds both reach the CLI. ``n_folds=null`` is what
    modules/local/ovwt_batchwise/main.nf emits for a null
    ``params.ovwt_n_folds`` (Groovy renders null as the literal ``null``),
    so this pins the Hydra parse that module depends on."""
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_ovwt(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        f"cv_mode={CV_MODE_BARCODE_HOLDOUT}",
        "n_folds=null",
        "min_cells=1",
    )
    assert result.returncode == 0, result.stderr

    with open(output_dir / "models.pkl", "rb") as f:
        models = pickle.load(f)
    assert len(models["M1K"]) == 2  # one fold per M1K barcode
    results = pl.read_parquet(output_dir / "results.parquet")
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert len(row["auroc_folds"]) == 2
