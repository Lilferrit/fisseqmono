from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.correlatefeatures as m


def make_corr_cfg(
    tmp_path, half1_file, half2_file, *, label_column="meta_aa_changes"
) -> OmegaConf:
    return OmegaConf.structured(
        m.CorrelateFeaturesConfig(
            output_dir=str(tmp_path / "corr_out"),
            half1_file=str(half1_file),
            half2_file=str(half2_file),
            label_column=label_column,
        )
    )


def test_main_writes_correlations_file(tmp_path) -> None:
    df1 = pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.0, 2.0]})
    df2 = pl.DataFrame({"meta_aa_changes": ["A", "B"], "f1_mean": [1.1, 2.1]})
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)

    with patch("fisseq_data_pipeline.correlatefeatures.setup_logging"):
        m.main.__wrapped__(make_corr_cfg(tmp_path, p1, p2))

    result = pl.read_parquet(tmp_path / "corr_out" / "correlations.parquet")
    assert set(result.columns) == {"feature", "r", "r_squared"}


def test_main_matches_compute_feature_correlations(tmp_path) -> None:
    df1 = pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_mean": [1.0, 2.0, 4.0]})
    df2 = pl.DataFrame({"meta_aa_changes": ["A", "B", "C"], "f1_mean": [2.0, 5.0, 1.0]})
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)

    with patch("fisseq_data_pipeline.correlatefeatures.setup_logging"):
        m.main.__wrapped__(make_corr_cfg(tmp_path, p1, p2))

    result = pl.read_parquet(tmp_path / "corr_out" / "correlations.parquet")
    expected = m.compute_feature_correlations(df1, df2, "meta_aa_changes")
    assert result["r"][0] == pytest.approx(expected["r"][0])
