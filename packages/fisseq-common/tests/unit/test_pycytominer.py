"""pycytominer feature selection."""

from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest

import fisseq_common.stages.pycytominer as m


@pytest.fixture
def agg_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "B", "C"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        }
    )


def test_pyc_feature_select_returns_polars_dataframe(
    agg_df: pl.DataFrame,
) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        result = m.pyc_feature_select(agg_df)
    assert isinstance(result, pl.DataFrame)


def test_pyc_feature_select_passes_feature_columns(agg_df: pl.DataFrame) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        m.pyc_feature_select(agg_df)
    features_arg = mock_fs.call_args.kwargs["features"]
    assert "f1" in features_arg
    assert "f2" in features_arg
    assert "meta_aa_changes" not in features_arg


def test_pyc_feature_select_passes_correct_operations(agg_df: pl.DataFrame) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        m.pyc_feature_select(agg_df)
    ops = mock_fs.call_args.kwargs["operation"]
    assert "variance_threshold" in ops
    assert "blocklist" in ops
    assert "correlation_threshold" in ops


def test_pyc_feature_select_dropped_features_absent_from_output(
    agg_df: pl.DataFrame,
) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.drop("f2").to_pandas()
        result = m.pyc_feature_select(agg_df)
    assert "f1" in result.columns
    assert "f2" not in result.columns


def test_pyc_feature_select_meta_columns_preserved(agg_df: pl.DataFrame) -> None:
    with patch("pycytominer.feature_select") as mock_fs:
        mock_fs.return_value = agg_df.to_pandas()
        result = m.pyc_feature_select(agg_df)
    assert "meta_aa_changes" in result.columns


def test_pyc_feature_select_with_no_operations_is_a_no_op(agg_df: pl.DataFrame) -> None:
    assert m.pyc_feature_select(agg_df, operations=[]).equals(agg_df)
