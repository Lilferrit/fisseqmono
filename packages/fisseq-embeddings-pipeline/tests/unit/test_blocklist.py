"""BLOCKLIST -- median r across replicates decides what is reproducible."""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import fisseq_embeddings_pipeline.blocklist as m


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    corr_dir = tmp_path / "correlations"
    for rep, r in enumerate([0.9, 0.8, 0.1], start=1):
        d = corr_dir / f"rep{rep}"
        d.mkdir(parents=True)
        pl.DataFrame(
            {"feature": ["keep", "drop"], "r": [r, 0.0], "r_squared": [r**2, 0.0]}
        ).write_parquet(d / "KS.parquet")

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.blocklist",
            f"output_dir={output_dir}",
            f"correlation_files={corr_dir}/rep*/KS.parquet",
            "minimum_correlation=0.5",
            "output_name=KS",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "KS.parquet").sort("feature")
    assert out["feature"].to_list() == ["drop", "keep"]
    assert out["feature_ok"].to_list() == [False, True]
    assert out["median_r"].to_list() == pytest.approx([0.0, 0.8])


def test_empty_glob_raises(tmp_path: Path) -> None:
    """An empty match here is always a wiring bug -- the correlation files
    are a declared rule input."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.blocklist",
            f"output_dir={tmp_path / 'out'}",
            f"correlation_files={tmp_path}/nothing/*.parquet",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode != 0
    assert "No files matched glob pattern" in result.stderr


def test_config_defaults() -> None:
    cfg = m.BlocklistConfig(correlation_files="x")
    assert cfg.minimum_correlation == 0.5
    assert cfg.output_name == "blocklist"


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
