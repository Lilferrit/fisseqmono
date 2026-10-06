from __future__ import annotations

import logging

import polars as pl
import pytest

from fisseq_common.schema import (
    COMPONENT_IDX_COL,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    VARIANCE_EXPLAINED_COL,
)
from fisseq_common.stages.dimreduction import compute_pca

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_df() -> pl.DataFrame:
    """6 rows, 3 informative feature columns -- enough for a 2-component fit
    and a 2-neighbor UMAP fit."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C", "A1D"],
            "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [0.0, 2.0, 1.0, 5.0, 3.0, 6.0],
            "f3": [1.0, 0.0, 3.0, 2.0, 5.0, 4.0],
        }
    )


@pytest.fixture
def feature_df_with_null_column(feature_df: pl.DataFrame) -> pl.DataFrame:
    return feature_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("f_null"))


# ---------------------------------------------------------------------------
# compute_pca
# ---------------------------------------------------------------------------


def test_compute_pca_returns_expected_score_columns(feature_df: pl.DataFrame) -> None:
    scores_df, _ = compute_pca(feature_df, "meta_aa_changes", 2)
    assert scores_df.columns == ["meta_aa_changes", "meta_pc_1", "meta_pc_2"]


def test_compute_pca_scores_row_count_matches_input(feature_df: pl.DataFrame) -> None:
    scores_df, _ = compute_pca(feature_df, "meta_aa_changes", 2)
    assert scores_df.height == feature_df.height


def test_compute_pca_components_df_schema(feature_df: pl.DataFrame) -> None:
    _, components_df = compute_pca(feature_df, "meta_aa_changes", 2)
    assert components_df[COMPONENT_IDX_COL].to_list() == [1, 2]
    assert VARIANCE_EXPLAINED_COL in components_df.columns
    assert CUMULATIVE_VARIANCE_EXPLAINED_COL in components_df.columns
    other_cols = set(components_df.columns) - {
        COMPONENT_IDX_COL,
        VARIANCE_EXPLAINED_COL,
        CUMULATIVE_VARIANCE_EXPLAINED_COL,
    }
    assert other_cols == {"f1", "f2", "f3"}


def test_compute_pca_cumulative_variance_nondecreasing_and_bounded(
    feature_df: pl.DataFrame,
) -> None:
    _, components_df = compute_pca(feature_df, "meta_aa_changes", 3)
    cum = components_df[CUMULATIVE_VARIANCE_EXPLAINED_COL].to_list()
    assert cum == sorted(cum)
    assert cum[-1] <= 1.0 + 1e-6


def test_compute_pca_drops_all_null_column_and_warns(
    feature_df_with_null_column: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        _, components_df = compute_pca(
            feature_df_with_null_column, "meta_aa_changes", 2
        )
    assert "f_null" in caplog.text
    assert "f_null" not in components_df.columns


def test_compute_pca_raises_when_all_features_all_null() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A", "B", "C"],
            "f1": [None, None, None],
        },
        schema={"meta_aa_changes": pl.Utf8, "f1": pl.Float64},
    )
    with pytest.raises(ValueError):
        compute_pca(df, "meta_aa_changes", 1)


def test_compute_pca_row_order_does_not_change_label_score_pairing(
    feature_df: pl.DataFrame,
) -> None:
    # sklearn's PCA applies svd_flip for a canonical, sample-order-independent
    # sign convention, so re-fitting the identical point set in a different
    # row order should reproduce identical loadings/scores per label -- not
    # just per row-position.
    scores_a, _ = compute_pca(feature_df, "meta_aa_changes", 2)
    scores_b, _ = compute_pca(feature_df.reverse(), "meta_aa_changes", 2)
    joined = scores_a.join(scores_b, on="meta_aa_changes", suffix="_reversed")
    assert joined["meta_pc_1"].to_list() == pytest.approx(
        joined["meta_pc_1_reversed"].to_list()
    )
    assert joined["meta_pc_2"].to_list() == pytest.approx(
        joined["meta_pc_2_reversed"].to_list()
    )
