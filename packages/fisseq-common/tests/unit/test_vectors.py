from __future__ import annotations

import polars as pl
import pytest

from fisseq_common.schema import CONTROL_COLUMN_NAME, IMPACT_SCORE_COL
from fisseq_common.utils.vectors import (
    COSINE_DIST_COL,
    compute_cosine_distance,
    compute_impact_score,
)


def test_compute_impact_score_control_is_zero() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert result[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_impact_score_orthogonal_is_half() -> None:
    # Control along f1; test row along f2 — cosine angle = 90° → score = 0.5
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, 0.0],
            "f2": [0.0, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(0.5, abs=1e-9)


def test_compute_impact_score_opposite_is_one() -> None:
    # Opposite direction to the control median → max impact score
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, -1.0],
            "f2": [0.0, 0.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_impact_score_drops_temp_columns() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert not [c for c in result.columns if c.startswith("tmp_")]


def test_compute_impact_score_output_columns() -> None:
    lf = pl.LazyFrame({CONTROL_COLUMN_NAME: [True], "f1": [1.0], "f2": [0.0]})
    result = compute_impact_score(lf).collect()
    assert IMPACT_SCORE_COL in result.columns
    assert CONTROL_COLUMN_NAME in result.columns
    assert "f1" in result.columns


def test_compute_impact_score_null_columns_excluded_from_calc() -> None:
    # f2 is null on the sole control row, so the control median for f2 is
    # itself null -> f2 is excluded from this row's calc (per-row, via
    # compute_cosine_distance), leaving f1 only. Control median (f1=1);
    # non-control (f1=-1) -> opposite direction -> score = 1.
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, False],
            "f1": [1.0, -1.0],
            "f2": [None, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_impact_score_null_in_one_row_does_not_affect_other_rows() -> None:
    # f2 is fully populated on both control rows and on one non-control row,
    # and only null on a *different* non-control row. A column-wide drop
    # (the old, buggy behavior) would exclude f2 from every row's score;
    # the fix must only exclude it for the row where it's actually null.
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, True, False, False],
            "f1": [0.0, 0.0, 0.0, 0.0],
            "f2": [1.0, 3.0, 5.0, None],  # control median f2 = 2.0
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())

    # Row with f2=5.0: same direction as control median (f1=0, f2=2) -> f2>0
    # both sides -> cosine_sim=1 -> score=0. This is only true if f2 was
    # actually used; if f2 had been dropped table-wide, f1 alone (all
    # zeros) would make the score 0.5 (zero-norm fallback) instead.
    populated_row = non_ctrl.filter(pl.col("f2") == 5.0)
    assert populated_row[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)

    # Row with f2=None still gets a finite score: f2 is excluded for just
    # this row, leaving f1 alone (zero on both sides -> zero-norm fallback
    # -> distance 1 -> score 0.5), matching compute_cosine_distance's
    # documented all-features-excluded behavior.
    null_row = non_ctrl.filter(pl.col("f2").is_null())
    assert null_row[IMPACT_SCORE_COL][0] == pytest.approx(0.5, abs=1e-9)


def test_compute_impact_score_null_columns_kept_in_output() -> None:
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True],
            "f1": [1.0],
            "f2": [None],
        }
    )
    result = compute_impact_score(lf).collect()
    assert "f2" in result.columns


def test_compute_impact_score_uses_control_median() -> None:
    # Two control rows whose median is (1, 0); verify the non-control score
    lf = pl.LazyFrame(
        {
            CONTROL_COLUMN_NAME: [True, True, False],
            "f1": [0.0, 2.0, 0.0],  # median f1 = 1.0
            "f2": [0.0, 0.0, 1.0],
        }
    )
    result = compute_impact_score(lf).collect()
    non_ctrl = result.filter(pl.col(CONTROL_COLUMN_NAME).not_())
    # control median = (1, 0); test = (0, 1) → orthogonal → 0.5
    assert non_ctrl[IMPACT_SCORE_COL][0] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_cosine_distance
# ---------------------------------------------------------------------------


def test_compute_cosine_distance_identical_vectors_is_zero() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [2.0], "f1_b": [1.0], "f2_b": [2.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_orthogonal_vectors_is_one() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0], "f1_b": [0.0], "f2_b": [1.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(1.0, abs=1e-9)


def test_compute_cosine_distance_opposite_vectors_is_two() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [0.0], "f1_b": [-1.0], "f2_b": [0.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(2.0, abs=1e-9)


def test_compute_cosine_distance_zero_norm_no_nan() -> None:
    lf = pl.LazyFrame({"f1": [0.0], "f2": [0.0], "f1_b": [1.0], "f2_b": [0.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert not result[COSINE_DIST_COL].is_nan().any()


def test_compute_cosine_distance_keeps_suffixed_columns() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f1_b": [1.0]})
    result = compute_cosine_distance(lf, ["f1"], suffix="_b").collect()
    assert "f1_b" in result.columns


def test_compute_cosine_distance_null_feature_excluded_one_side() -> None:
    # f2 is null on the unsuffixed side, so only f1 is used -> identical -> 0.0
    lf = pl.LazyFrame({"f1": [1.0], "f2": [None], "f1_b": [1.0], "f2_b": [5.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_null_feature_excluded_other_side() -> None:
    # f2 is null on the suffixed side instead -> same exclusion, same result
    lf = pl.LazyFrame({"f1": [1.0], "f2": [5.0], "f1_b": [1.0], "f2_b": [None]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_nan_feature_excluded() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [float("nan")], "f1_b": [1.0], "f2_b": [5.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_inf_feature_excluded() -> None:
    lf = pl.LazyFrame({"f1": [1.0], "f2": [float("inf")], "f1_b": [1.0], "f2_b": [5.0]})
    result = compute_cosine_distance(lf, ["f1", "f2"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_compute_cosine_distance_all_features_excluded_no_nan() -> None:
    lf = pl.LazyFrame({"f1": [None]}, schema={"f1": pl.Float64})
    lf = lf.with_columns(f1_b=pl.lit(None, dtype=pl.Float64))
    result = compute_cosine_distance(lf, ["f1"], suffix="_b").collect()
    assert result[COSINE_DIST_COL][0] == pytest.approx(1.0, abs=1e-9)
    assert not result[COSINE_DIST_COL].is_nan().any()
