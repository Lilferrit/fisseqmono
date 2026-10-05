"""CORRELATE_FEATURES -- per-dimension Pearson r between two split halves."""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import fisseq_embeddings_pipeline.correlatefeatures as m

LABEL = "meta_aa_changes"


def _half(values: "dict[str, list[float]]", labels: "list[str]") -> pl.DataFrame:
    return pl.DataFrame({LABEL: labels, **values})


def test_main_runs_end_to_end_via_cli(tmp_path: Path) -> None:
    labels = ["A", "B", "C", "D"]
    _half({"emb_0000_KS": [1.0, 2.0, 3.0, 4.0]}, labels).write_parquet(
        tmp_path / "h1.parquet"
    )
    _half({"emb_0000_KS": [1.0, 2.0, 3.0, 4.0]}, labels).write_parquet(
        tmp_path / "h2.parquet"
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.correlatefeatures",
            f"output_dir={output_dir}",
            f"half1_file={tmp_path / 'h1.parquet'}",
            f"half2_file={tmp_path / 'h2.parquet'}",
            "output_name=KS",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "KS.parquet")
    assert out["feature"].to_list() == ["emb_0000_KS"]
    assert out["r"][0] == pytest.approx(1.0)


def test_main_is_hydra_entry_point() -> None:
    assert callable(m.main)
