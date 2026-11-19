"""Tests for concat_embeddings -- CONCAT_EMBEDDINGS, which stacks EMBED_CELLS'
per-well tables into the experiment's embeddings.parquet."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from fisseq_embeddings_pipeline.concat_embeddings import (
    concat_embeddings,
    well_tables,
)
from fisseq_embeddings_pipeline.utils.cell_table import CELL_METADATA_SCHEMA


def _well_table(path: Path, well: str, n_cells: int, n_dims: int = 3) -> Path:
    """One well's EMBED_CELLS output: CELL_METADATA_SCHEMA, then emb_*."""
    if n_cells == 0:
        frame = pl.DataFrame(schema=CELL_METADATA_SCHEMA)
    else:
        frame = pl.DataFrame(
            {
                "meta_batch": ["batch1"] * n_cells,
                "meta_well": [well] * n_cells,
                "meta_tile": ["tile00x00y"] * n_cells,
                "meta_cell_index": list(range(1, n_cells + 1)),
                "meta_barcode": [f"bc{i}" for i in range(n_cells)],
                "meta_aa_changes": ["WT"] * n_cells,
                "meta_edit_distance": [0] * n_cells,
                **{
                    f"emb_{d:04d}": [float(i + d) for i in range(n_cells)]
                    for d in range(n_dims)
                },
            },
            schema_overrides=CELL_METADATA_SCHEMA,
        )
    frame.write_parquet(path)
    return path


def test_well_tables_are_in_natural_order(tmp_path: Path):
    for well in ("well10", "well2", "well1"):
        _well_table(tmp_path / f"embeddings_{well}.parquet", well, 1)

    assert [p.name for p in well_tables(str(tmp_path))] == [
        "embeddings_well1.parquet",
        "embeddings_well2.parquet",
        "embeddings_well10.parquet",
    ]


def test_well_tables_requires_at_least_one(tmp_path: Path):
    with pytest.raises(ValueError, match="no \\*.parquet"):
        well_tables(str(tmp_path))


def test_concat_stacks_every_well_in_order(tmp_path: Path):
    paths = [
        _well_table(tmp_path / "embeddings_well1.parquet", "well1", 2),
        _well_table(tmp_path / "embeddings_well2.parquet", "well2", 3),
    ]

    out = concat_embeddings(paths).collect()

    assert out.columns == [*CELL_METADATA_SCHEMA, "emb_0000", "emb_0001", "emb_0002"]
    assert out["meta_well"].to_list() == ["well1"] * 2 + ["well2"] * 3
    assert out.schema["meta_cell_index"] == CELL_METADATA_SCHEMA["meta_cell_index"]


def test_concat_keeps_an_empty_wells_place(tmp_path: Path):
    """A well with no cells has no emb_* columns; it adds no rows, and the
    other wells' columns come through whichever well is first."""
    paths = [
        _well_table(tmp_path / "embeddings_well1.parquet", "well1", 0),
        _well_table(tmp_path / "embeddings_well2.parquet", "well2", 2),
    ]

    out = concat_embeddings(paths).collect()

    assert out.height == 2
    assert out.columns == [*CELL_METADATA_SCHEMA, "emb_0000", "emb_0001", "emb_0002"]


def test_concat_of_only_empty_wells_is_the_bare_schema(tmp_path: Path):
    paths = [_well_table(tmp_path / "embeddings_well1.parquet", "well1", 0)]

    out = concat_embeddings(paths).collect()

    assert out.height == 0
    assert out.schema == pl.Schema(CELL_METADATA_SCHEMA)


def test_main_runs_via_cli(tmp_path: Path):
    wells = tmp_path / "wells"
    wells.mkdir()
    _well_table(wells / "embeddings_well1.parquet", "well1", 2)
    _well_table(wells / "embeddings_well2.parquet", "well2", 1)
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.concat_embeddings",
            f"output_dir={output_dir}",
            f"inputs_dir={wells}",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "embeddings.parquet")
    assert out["meta_well"].to_list() == ["well1", "well1", "well2"]
