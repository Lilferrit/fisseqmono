"""COMBINE_BLOCKLISTS (``python -m fisseq_common.stages.combineblocklists``): one experiment's
per-method blocklists in a single table."""

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_common.stages.combineblocklists as m


def _run_main(tmp_path, blocklist_files: str) -> pl.DataFrame:
    cfg = m.CombineBlocklistsParams(
        output_dir=str(tmp_path / "out"), blocklist_files=blocklist_files
    )
    with patch("fisseq_common.stages.config.setup_logging"):
        m.main.__wrapped__(OmegaConf.structured(cfg))
    return pl.read_parquet(tmp_path / "out" / "blocklist.parquet")


def test_main_concatenates_the_methods_sorted_by_feature(tmp_path) -> None:
    bl_dir = tmp_path / "bl"
    bl_dir.mkdir()
    pl.DataFrame(
        {
            "feature": ["f2_mean", "f1_mean"],
            "median_r": [0.9, 0.8],
            "feature_ok": [True] * 2,
        }
    ).write_parquet(bl_dir / "mean.parquet")
    pl.DataFrame(
        {"feature": ["f1_std"], "median_r": [0.3], "feature_ok": [False]}
    ).write_parquet(bl_dir / "std.parquet")

    result = _run_main(tmp_path, str(bl_dir / "*.parquet"))

    assert result["feature"].to_list() == ["f1_mean", "f1_std", "f2_mean"]
    assert result["feature_ok"].to_list() == [True, False, True]


def test_main_raises_on_empty_glob(tmp_path) -> None:
    with pytest.raises(ValueError, match="No files matched"):
        _run_main(tmp_path, str(tmp_path / "nonexistent" / "*.parquet"))
