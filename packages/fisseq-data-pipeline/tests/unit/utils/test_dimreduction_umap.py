"""compute_umap (fisseq_common.stages.dimreduction), tested here because umap-learn is a
dependency of this pipeline, its only caller, and not of fisseq-common."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from fisseq_common.stages.dimreduction import compute_umap

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
# compute_umap
# ---------------------------------------------------------------------------


def test_compute_umap_returns_expected_score_columns(feature_df: pl.DataFrame) -> None:
    result = compute_umap(
        feature_df,
        "meta_aa_changes",
        n_components=2,
        n_neighbors=2,
        metric="cosine",
        min_dist=0.1,
        random_state=42,
    )
    assert result.columns == ["meta_aa_changes", "meta_umap_1", "meta_umap_2"]


def test_compute_umap_scores_row_count_matches_input(feature_df: pl.DataFrame) -> None:
    result = compute_umap(
        feature_df,
        "meta_aa_changes",
        n_components=2,
        n_neighbors=2,
        metric="cosine",
        min_dist=0.1,
        random_state=42,
    )
    assert result.height == feature_df.height


def test_compute_umap_drops_all_null_column_and_warns(
    feature_df_with_null_column: pl.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        result = compute_umap(
            feature_df_with_null_column,
            "meta_aa_changes",
            n_components=2,
            n_neighbors=2,
            metric="cosine",
            min_dist=0.1,
            random_state=42,
        )
    assert "f_null" in caplog.text
    assert result.height == feature_df_with_null_column.height


def test_compute_umap_raises_when_all_features_all_null() -> None:
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A", "B", "C"],
            "f1": [None, None, None],
        },
        schema={"meta_aa_changes": pl.Utf8, "f1": pl.Float64},
    )
    with pytest.raises(ValueError):
        compute_umap(
            df,
            "meta_aa_changes",
            n_components=1,
            n_neighbors=1,
            metric="cosine",
            min_dist=0.1,
            random_state=42,
        )


def test_compute_umap_params_passed_to_umap_constructor(
    feature_df: pl.DataFrame,
) -> None:
    mock_reducer = MagicMock()
    mock_reducer.fit_transform.return_value = np.zeros((feature_df.height, 2))
    with patch("umap.UMAP", return_value=mock_reducer) as mock_umap_cls:
        compute_umap(
            feature_df,
            "meta_aa_changes",
            n_components=2,
            n_neighbors=3,
            metric="cosine",
            min_dist=0.25,
            random_state=7,
        )
    _, kwargs = mock_umap_cls.call_args
    assert kwargs["metric"] == "cosine"
    assert kwargs["n_neighbors"] == 3
    assert kwargs["min_dist"] == 0.25
    assert kwargs["random_state"] == 7
    assert kwargs["n_components"] == 2
