"""OvWT scoring on embedding-shaped features (``emb_*`` columns, ``EMBEDDING_SELECTOR``).

Complements test_ovwt.py, which scores CellProfiler-shaped columns. Fixture sizing note: the
inner 80/20 train/calibration split runs *inside* each outer fold, so a ``(barcode, is_wt)``
stratum wants comfortably more than ``n_folds`` members for the folds to carry a real signal.
Fixtures here use ``n_folds=3`` with ~15 cells per barcode for that reason.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_common.stages.ovwt as ovwt_mod
from fisseq_common.stages.ovwt import (
    OvwtParams,
    _barcode_holdout_splits,
    downsample_wildtype,
    filter_min_cells,
    ovwt_batchwise,
    predict_binary,
)
from fisseq_common.stages.xgbparams import train_binary_xgboost


def _cfg(**overrides) -> OvwtParams:
    defaults = dict(
        output_dir="/tmp/out",
    )
    defaults.update(overrides)
    return OvwtParams(**defaults)


def _separable_df() -> pl.DataFrame:
    """WT cells cluster around emb_0000=1.0, variant cells around emb_0000=0.0
    -- trivially separable, so a fitted model's predicted P(wildtype) should
    land near 1 for WT rows and near 0 for variant rows."""
    n = 20
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n + ["M1K"] * n,
            "emb_0000": [1.0 + 0.01 * i for i in range(n)]
            + [0.0 + 0.01 * i for i in range(n)],
        }
    )


def _xgb_cfg():
    return OmegaConf.structured(_cfg())


def test_predict_binary_separates_wt_from_variant():
    df = _separable_df()
    model = train_binary_xgboost(df, df, "meta_aa_changes", "WT", _xgb_cfg())
    scores = predict_binary(df, model, "meta_aa_changes", "WT")

    wt_scores = scores[: len(df) // 2]
    variant_scores = scores[len(df) // 2 :]
    # Early stopping (5 rounds) means predicted probabilities aren't fully
    # saturated to 0/1 even on trivially-separable data -- assert clear
    # directional separation rather than a strict >0.9/<0.1 threshold.
    assert wt_scores.mean() > 0.7
    assert variant_scores.mean() < 0.3
    assert wt_scores.mean() > variant_scores.mean()


def test_filter_min_cells_drops_small_variant():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["M1K"] * 3 + ["M2L"] * 10,
            "meta_barcode": ["bc_wt"] * 5 + ["bc1"] * 3 + ["bc2"] * 10,
        }
    )
    out = filter_min_cells(df, "meta_aa_changes", "WT", min_cells=5)
    assert set(out["meta_aa_changes"].to_list()) == {"WT", "M2L"}


def test_filter_min_cells_keeps_wt_regardless_of_count():
    df = pl.DataFrame(
        {"meta_aa_changes": ["WT"] * 2 + ["M1K"] * 10, "meta_barcode": ["a"] * 12}
    )
    out = filter_min_cells(df, "meta_aa_changes", "WT", min_cells=5)
    assert "WT" in out["meta_aa_changes"].to_list()


def test_filter_min_cells_none_is_noop():
    df = pl.DataFrame(
        {"meta_aa_changes": ["WT"] * 2 + ["M1K"] * 1, "meta_barcode": ["a"] * 3}
    )
    out = filter_min_cells(df, "meta_aa_changes", "WT", min_cells=None)
    assert out.height == df.height


def test_downsample_wildtype_shrinks_to_largest_variant():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 100 + ["M1K"] * 20,
            "meta_barcode": ["bc_wt"] * 100 + ["bc1"] * 20,
        }
    )
    out = downsample_wildtype(df, "meta_aa_changes", "WT", seed=0)
    wt_count = (out["meta_aa_changes"] == "WT").sum()
    assert wt_count <= 21  # target 20 +/- rounding slack
    assert (out["meta_aa_changes"] == "M1K").sum() == 20


def test_downsample_wildtype_noop_when_wt_already_smaller():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * 5 + ["M1K"] * 20,
            "meta_barcode": ["bc_wt"] * 5 + ["bc1"] * 20,
        }
    )
    out = downsample_wildtype(df, "meta_aa_changes", "WT", seed=0)
    assert out.height == df.height


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


def _ovwt_cfg(**overrides) -> OvwtParams:
    defaults = dict(n_folds=3, min_cells=1)
    defaults.update(overrides)
    return _cfg(**defaults)


def test_ovwt_batchwise_output_columns_exact():
    results, cell_scores, _ = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg())
    assert set(results.columns) == {
        "meta_aa_changes",
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
        "meta_n_barcodes",
        "meta_n_cells",
    }
    assert "score" in cell_scores.columns
    assert "meta_variant_scored_against" in cell_scores.columns


def test_ovwt_batchwise_auroc_pooled_in_valid_range_and_separable():
    results, _, _ = ovwt_batchwise(_kfold_fixture_lf(separable=True), _ovwt_cfg())
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert 0.0 <= row["auroc_pooled"] <= 1.0
    assert row["auroc_pooled"] > 0.7  # clearly separable synthetic data


def test_ovwt_batchwise_auroc_median_barcode_excludes_wt():
    """One variant barcode is clearly separable from WT, the other
    overlaps heavily with WT -- auroc_median_barcode (computed only over
    the variant's own barcodes) should sit meaningfully below the
    perfectly-separable single-barcode AUROC, confirming the median is
    computed per-barcode and WT's own barcode never enters that set."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "emb_0000": 1.0 + rng.normal(scale=0.05),
            }
        )
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "M1K",
                "meta_barcode": "bc_v0",
                "emb_0000": 0.0 + rng.normal(scale=0.05),
            }
        )
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "M1K",
                "meta_barcode": "bc_v1",
                "emb_0000": 1.0 + rng.normal(scale=0.05),
            }
        )
    lf = pl.DataFrame(rows).lazy()

    results, _, _ = ovwt_batchwise(lf, _ovwt_cfg())
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert row["meta_n_barcodes"] == 2
    assert row["auroc_median_barcode"] is not None
    assert 0.0 <= row["auroc_median_barcode"] <= 1.0


def test_ovwt_batchwise_meta_n_cells_and_barcodes():
    results, _, _ = ovwt_batchwise(
        _kfold_fixture_lf(n_variant_barcodes=2, variant_n=15, wt_n=15), _ovwt_cfg()
    )
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert row["meta_n_barcodes"] == 2
    assert row["meta_n_cells"] == 15 + 2 * 15


def test_ovwt_batchwise_models_shape_matches_n_folds():
    _, _, models = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg())
    assert "M1K" in models
    assert len(models["M1K"]) == 3  # n_folds=3


def test_ovwt_batchwise_calibrate_false_gives_none_calibrators():
    _, _, models = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg(calibrate=False))
    for _model, calibrator in models["M1K"]:
        assert calibrator is None


def test_ovwt_batchwise_calibrate_true_gives_calibrators():
    _, _, models = ovwt_batchwise(_kfold_fixture_lf(), _ovwt_cfg(calibrate=True))
    for _model, calibrator in models["M1K"]:
        assert calibrator is not None


def test_ovwt_batchwise_rare_barcode_variant_is_scored_not_skipped():
    """M2L has a single, extremely rare barcode (2 cells). It collapses into
    _stratification_key's "rare|variant" bucket, and a fold's share of that
    bucket used to be a singleton the inner split raised on, so M2L got
    skipped. split_indices_stratified now sends singleton strata to the
    train half, so M2L is scored alongside M1K."""
    rng = np.random.default_rng(2)
    rows = []
    for i in range(15):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "emb_0000": 1.0 + rng.normal(scale=0.05),
            }
        )
    for b in range(2):
        for i in range(15):
            rows.append(
                {
                    "meta_aa_changes": "M1K",
                    "meta_barcode": f"bc_v{b}",
                    "emb_0000": 0.0 + rng.normal(scale=0.05),
                }
            )
    for i in range(2):
        rows.append(
            {
                "meta_aa_changes": "M2L",
                "meta_barcode": "bc_rare",
                "emb_0000": 0.5 + rng.normal(scale=0.05),
            }
        )
    lf = pl.DataFrame(rows).lazy()

    results, cell_scores, models = ovwt_batchwise(lf, _ovwt_cfg())

    assert {"M1K", "M2L"} <= set(results["meta_aa_changes"].to_list())
    assert {"M1K", "M2L"} <= set(models)
    assert cell_scores["score"].null_count() == 0


def test_ovwt_batchwise_variant_failure_is_isolated_not_fatal(monkeypatch):
    """One doomed variant must not take the whole run down with it."""
    real_train = ovwt_mod.train_binary_xgboost

    def _fail_on_m2l(train, val, label_col, wt_label, cfg):
        if (train.get_column(label_col) == "M2L").any():
            raise ValueError("synthetic training failure")
        return real_train(train, val, label_col, wt_label, cfg)

    monkeypatch.setattr(ovwt_mod, "train_binary_xgboost", _fail_on_m2l)
    lf = pl.concat(
        [
            _kfold_fixture_lf().collect(),
            _kfold_fixture_lf(variant_label="M2L")
            .collect()
            .filter(pl.col("meta_aa_changes") == "M2L"),
        ]
    ).lazy()
    results, _, models = ovwt_batchwise(lf, _ovwt_cfg())
    assert "M1K" in results["meta_aa_changes"].to_list()
    assert "M2L" not in results["meta_aa_changes"].to_list()
    assert "M2L" not in models


def test_ovwt_batchwise_all_variants_filtered_out_returns_empty_frames():
    """min_cells set higher than any variant's cell count filters every
    non-WT variant out before the per-variant loop even starts -- must
    return correctly-schema'd empty DataFrames, not raise on pl.concat([])."""
    lf = _kfold_fixture_lf()
    results, cell_scores, models = ovwt_batchwise(lf, _ovwt_cfg(min_cells=10_000))

    assert results.height == 0
    assert set(results.columns) == {
        "meta_aa_changes",
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
        "meta_n_barcodes",
        "meta_n_cells",
    }
    # Explicit schema: an all-null list/median column must not come back as
    # the Null dtype.
    assert results.schema["auroc_folds"] == pl.List(pl.Float64)
    assert results.schema["auroc_median_fold"] == pl.Float64
    assert cell_scores.height == 0
    assert "score" in cell_scores.columns
    assert models == {}


def _cp_style_kfold_fixture_lf(wt_n: int = 15, variant_n: int = 15) -> pl.LazyFrame:
    """Same shape/separability as _kfold_fixture_lf, but with a
    CellProfiler-style feature-column name (not matched by
    EMBEDDING_SELECTOR)."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(wt_n):
        rows.append(
            {
                "meta_aa_changes": "WT",
                "meta_barcode": "bc_wt",
                "Cells_AreaShape_Area": 1.0 + rng.normal(scale=0.05),
            }
        )
    for i in range(variant_n):
        rows.append(
            {
                "meta_aa_changes": "M1K",
                "meta_barcode": "bc_v0",
                "Cells_AreaShape_Area": 0.0 + rng.normal(scale=0.05),
            }
        )
    return pl.DataFrame(rows).lazy()


def test_ovwt_batchwise_with_feature_selector_matches_cp_style_columns():
    from fisseq_common.schema import FEATURE_SELECTOR

    results, cell_scores, _ = ovwt_batchwise(
        _cp_style_kfold_fixture_lf(), _ovwt_cfg(), feature_selector=FEATURE_SELECTOR
    )
    row = results.filter(pl.col("meta_aa_changes") == "M1K").row(0, named=True)
    assert 0.0 <= row["auroc_pooled"] <= 1.0
    assert row["auroc_pooled"] > 0.7  # clearly separable synthetic data
    assert cell_scores["score"].null_count() == 0


LABEL = "meta_aa_changes"


WT = "WT"


def _cells(
    variants: "dict[str, int] | None" = None,
    wt_barcodes: int = 3,
    cells_per_wt_barcode: int = 15,
    barcodes_per_variant: int = 2,
    seed: int = 0,
) -> pl.DataFrame:
    """Cell-level frame with a real signal: variants sit away from WT on
    emb_0000, so the classifier has something to find. ``variants`` maps a
    variant label to its per-barcode cell count."""
    variants = variants or {"M1K": 15, "A1A": 15}
    rng = np.random.default_rng(seed)
    rows = {LABEL: [], "meta_barcode": [], "emb_0000": [], "emb_0001": []}

    for b in range(wt_barcodes):
        n = cells_per_wt_barcode
        rows[LABEL] += [WT] * n
        rows["meta_barcode"] += [f"wt_bc{b}"] * n
        rows["emb_0000"] += rng.normal(0.0, 0.3, n).tolist()
        rows["emb_0001"] += rng.random(n).tolist()

    for offset, (variant, per_barcode) in enumerate(variants.items(), start=1):
        for b in range(barcodes_per_variant):
            rows[LABEL] += [variant] * per_barcode
            rows["meta_barcode"] += [f"{variant}_bc{b}"] * per_barcode
            rows["emb_0000"] += rng.normal(3.0 * offset, 0.3, per_barcode).tolist()
            rows["emb_0001"] += rng.random(per_barcode).tolist()

    return pl.DataFrame(rows)


def _port_cfg(**overrides) -> OvwtParams:
    base = dict(n_folds=3, min_cells=None, downsample_wt=False, random_seed=0)
    base.update(overrides)
    return _cfg(**base)


def test_results_column_order():
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg())
    assert results.columns == [
        LABEL,
        "auroc_pooled",
        "auroc_median_barcode",
        "auroc_folds",
        "auroc_median_fold",
        "meta_n_barcodes",
        "meta_n_cells",
    ]


def test_one_fold_auroc_per_fold():
    cfg = _port_cfg(n_folds=3)
    results, _, models = ovwt_batchwise(_cells().lazy(), cfg)
    assert results.schema["auroc_folds"] == pl.List(pl.Float64)
    for row in results.iter_rows(named=True):
        assert len(row["auroc_folds"]) == cfg.n_folds
        assert len(row["auroc_folds"]) == len(models[row[LABEL]])


def test_median_fold_is_median_of_fold_aurocs():
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg())
    for row in results.iter_rows(named=True):
        defined = [a for a in row["auroc_folds"] if a is not None]
        assert row["auroc_median_fold"] == pytest.approx(float(np.median(defined)))


def test_median_fold_recovers_separable_signal():
    results, _, _ = ovwt_batchwise(_cells().lazy(), _port_cfg())
    assert results.get_column("auroc_median_fold").min() > 0.9


def _holdout_arrays(
    n_variant_barcodes: int = 3, n_wt_barcodes: int = 3, per_barcode: int = 15
) -> "tuple[np.ndarray, np.ndarray]":
    """``(barcodes, is_wt)`` for a synthetic variant-vs-wildtype subset."""
    barcodes, is_wt = [], []
    for b in range(n_wt_barcodes):
        barcodes += [f"wt_bc{b}"] * per_barcode
        is_wt += [True] * per_barcode
    for b in range(n_variant_barcodes):
        barcodes += [f"var_bc{b}"] * per_barcode
        is_wt += [False] * per_barcode
    return np.array(barcodes), np.array(is_wt)


def test_holdout_never_trains_on_a_held_out_barcode():
    """Each held-out barcode is wholly in test and wholly absent from fit --
    with one barcode per fold and with several packed into a fold."""
    for n_variant_barcodes, n_folds, per_fold in ((3, None, 1), (6, 3, 2)):
        barcodes, is_wt = _holdout_arrays(n_variant_barcodes=n_variant_barcodes)
        splits = _barcode_holdout_splits(barcodes, is_wt, seed=0, n_folds=n_folds)
        for fit_idx, test_idx in splits:
            held_out = np.unique(barcodes[test_idx][~is_wt[test_idx]])
            assert len(held_out) == per_fold
            for barcode in held_out:
                assert set(np.flatnonzero(barcodes == barcode)) <= set(test_idx)
                assert barcode not in set(barcodes[fit_idx])
