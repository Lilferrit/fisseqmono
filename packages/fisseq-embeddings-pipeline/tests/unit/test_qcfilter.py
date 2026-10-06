"""Tests for QC_FILTER."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import polars as pl

from fisseq_embeddings_pipeline.qcfilter import (
    QcFilterConfig,
    main,
)


def _cfg(**overrides) -> QcFilterConfig:
    base = dict(
        output_dir="/tmp/out",
        cell_files=[],
        bc_threshold=3,
        variant_bc_threshold=2,
        edit_distance_threshold=1,
    )
    base.update(overrides)
    return QcFilterConfig(**base)


def test_qc_filter_config_leaves_pseudo_variants_off():
    """Pseudo-variant downsampling is a data-pipeline calibration; off here by default."""
    cfg = _cfg()
    assert cfg.downsample_amounts is None
    assert not hasattr(cfg, "downsample_seed")


def test_qc_filter_config_reads_metadata_parquet_column_names():
    cfg = _cfg()
    assert cfg.barcode_col_name == "meta_barcode"
    assert cfg.aa_changes_col_name == "meta_aa_changes"
    assert cfg.edit_distance_col_name == "meta_edit_distance"


def _write_cells(path: Path, barcodes: List[str], aa_changes: List[str]) -> None:
    n = len(barcodes)
    pl.DataFrame(
        {
            "meta_batch": ["batch1"] * n,
            "meta_well": ["well1"] * n,
            "meta_tile": ["tile0x0y"] * n,
            "meta_cell_index": list(range(n)),
            "meta_barcode": barcodes,
            "meta_aa_changes": aa_changes,
            "meta_edit_distance": [0] * n,
        }
    ).write_parquet(path)


def _run_qcfilter(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.qcfilter", *args],
        capture_output=True,
        text=True,
        # See test_tile_shard.py's identical comment: keeps Hydra's own
        # outputs/<date>/<time>/ dir out of the repo tree.
        cwd=tmp_path,
    )


def test_main_composite_join_key_survives_end_to_end(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(3)], ["A1A"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    for col in ("meta_batch", "meta_well", "meta_tile", "meta_cell_index"):
        assert col in filtered.columns
    assert filtered.height == 3


def test_main_sorts_on_the_cell_keys_and_keeps_meta_cell_index(tmp_path: Path):
    """filtered_cells.parquet is sorted on JOIN_KEYS (a stable row order for every seeded
    step downstream), and meta_cell_index keeps its per-tile values: QC doesn't reassign
    it, unlike in the data pipeline."""
    source = tmp_path / "metadata.parquet"
    n = 6
    pl.DataFrame(
        {
            "meta_batch": ["batch1"] * n,
            "meta_well": ["well1"] * n,
            "meta_tile": ["tile1", "tile0", "tile1", "tile0", "tile1", "tile0"],
            "meta_cell_index": [2, 7, 0, 3, 1, 5],
            "meta_barcode": [f"bc{i}" for i in range(n)],
            "meta_aa_changes": ["A1A"] * n,
            "meta_edit_distance": [0] * n,
        }
    ).write_parquet(source)
    output_dir = tmp_path / "out"
    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr
    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert filtered.select("meta_tile", "meta_cell_index").rows() == [
        ("tile0", 3),
        ("tile0", 5),
        ("tile0", 7),
        ("tile1", 0),
        ("tile1", 1),
        ("tile1", 2),
    ]


def test_main_n_variants_none_matches_no_restriction(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(6)], ["M1K"] * 3 + ["M2L"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert set(filtered["meta_aa_changes"].to_list()) == {"M1K", "M2L"}


def test_main_n_variants_restricts_before_qc_thresholds(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    # M2L has 3 barcodes (passes variant_bc_threshold=2); M1K has only 1
    # barcode so it would fail variant_bc_threshold on its own -- but
    # n_variants=1 in "top" mode should drop M1K (fewer cells) before QC
    # thresholding even runs, so variants_per_barcode never sees it either.
    _write_cells(source, ["bc0"] + ["bc1", "bc2", "bc3"], ["M1K"] + ["M2L"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=2",
        "n_variants=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert set(filtered["meta_aa_changes"].to_list()) == {"M2L"}

    variants_per_barcode = pl.read_parquet(output_dir / "variants_per_barcode.parquet")
    assert "M1K" not in variants_per_barcode["meta_aa_changes"].to_list()


def test_main_variant_allow_list_file_bypasses_cap(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(
        source,
        [f"bc{i}" for i in range(6)],
        ["M1K", "M2L", "M2L", "M3Q", "M3Q", "M3Q"],
    )
    allow_list_path = tmp_path / "allow_list.parquet"
    pl.DataFrame({"meta_aa_changes": ["M1K"]}).write_parquet(allow_list_path)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "n_variants=1",
        f"variant_allow_list_file={allow_list_path}",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    # M1K is allow-listed and passes through despite having the fewest
    # cells; top-1 among the remaining {M2L, M3Q} keeps M3Q.
    assert set(filtered["meta_aa_changes"].to_list()) == {"M1K", "M3Q"}


def test_main_variant_allow_list_ignored_when_n_variants_none(tmp_path: Path):
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(6)], ["M1K"] * 3 + ["M2L"] * 3)
    allow_list_path = tmp_path / "allow_list.parquet"
    pl.DataFrame({"meta_aa_changes": ["M1K"]}).write_parquet(allow_list_path)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        f"variant_allow_list_file={allow_list_path}",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr

    filtered = pl.read_parquet(output_dir / "filtered_cells.parquet")
    assert set(filtered["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
    assert "variant_allow_list_file" in result.stdout + result.stderr


def test_main_single_cell_files_path_not_a_list(tmp_path: Path):
    """cell_files accepts a single (non-list) metadata.parquet-shaped path,
    matching how BUILD_DATASET's output is wired in via Nextflow."""
    source = tmp_path / "metadata.parquet"
    _write_cells(source, [f"bc{i}" for i in range(3)], ["A1A"] * 3)
    output_dir = tmp_path / "out"

    result = _run_qcfilter(
        tmp_path,
        f"output_dir={output_dir}",
        f"cell_files={source}",
        "bc_threshold=1",
        "variant_bc_threshold=1",
        "random_seed=0",
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "filtered_cells.parquet").exists()
    assert (output_dir / "barcode_counts.parquet").exists()
    assert (output_dir / "variants_per_barcode.parquet").exists()


def test_main_is_hydra_entry_point():
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(main)
