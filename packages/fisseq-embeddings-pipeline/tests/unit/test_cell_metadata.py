"""Tests for BUILD_CELL_METADATA.

Covers build_cell_metadata() -- the flat meta_* projection of
BUILD_CELL_IMAGES' cell_table.parquet that feeds QC_FILTER -- plus
empty-table handling and the Hydra main() CLI end-to-end.

The projection itself is shared with BUILD_CP_FEATURES via
utils/cell_table.py, so the two stages can't drift on the four columns
that become the shared stages' EMBEDDINGS_JOIN_KEYS; test_projection_matches_cp_features
below pins that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from fisseq_common.schema import (
    META_BARCODE_COL,
    META_BATCH_COL,
    META_EDIT_DISTANCE_COL,
)
from fisseq_common.stages.config import EMBEDDINGS_JOIN_KEYS as JOIN_KEYS
from fisseq_embeddings_pipeline.cell_metadata import (
    CellMetadataConfig,
    build_cell_metadata,
    main,
)
from fisseq_embeddings_pipeline.cp_features import CpFeaturesConfig, build_cp_features
from fisseq_embeddings_pipeline.utils.cell_table import (
    CELL_META_SCHEMA,
    CELL_METADATA_SCHEMA,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _cfg(cell_table: Path, **overrides) -> CellMetadataConfig:
    defaults = dict(
        output_dir="/tmp/out",
        cell_table=str(cell_table),
        batch_stem="test_batch",
    )
    defaults.update(overrides)
    return CellMetadataConfig(**defaults)


def _row(
    well: str,
    tile: str,
    tile_cell_index: int,
    barcode: str,
    aa_changes: str,
    edit_distance: int,
    **cp_features,
) -> dict:
    """One row of BUILD_CELL_IMAGES' cell_table.parquet: its meta_*
    columns, then CellProfiler columns under their own names."""
    row = {
        "meta_well": well,
        "meta_tile": tile,
        "meta_cell_index": tile_cell_index,
        "meta_barcode": barcode,
        "meta_aa_changes": aa_changes,
        "meta_edit_distance": edit_distance,
        "meta_variant_class": "Synonymous",
    }
    row.update(cp_features)
    return row


def _write_cell_table(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# build_cell_metadata() -- the meta_* projection
# ---------------------------------------------------------------------------


def test_build_cell_metadata_projects_meta_columns(tmp_path: Path):
    cell_table = _write_cell_table(
        tmp_path / "cell_table.parquet",
        [
            _row("well1", "tile0x0y", 0, "bc1", "A1A", 0),
            _row("well1", "tile0x0y", 1, "bc2", "A1B:tagged", 1),
            _row("well2", "tile1x0y", 0, "bc3", "synonymous", 0),
        ],
    )

    result = build_cell_metadata(_cfg(cell_table))

    assert result.columns == list(CELL_METADATA_SCHEMA)
    assert result.height == 3
    assert result[META_BATCH_COL].unique().to_list() == ["test_batch"]
    assert result["meta_well"].to_list() == ["well1", "well1", "well2"]
    assert result["meta_cell_index"].to_list() == [0, 1, 0]
    assert result[META_BARCODE_COL].to_list() == ["bc1", "bc2", "bc3"]
    assert result["meta_aa_changes"].to_list() == ["A1A", "A1B:tagged", "synonymous"]
    assert result[META_EDIT_DISTANCE_COL].to_list() == [0, 1, 0]


def test_build_cell_metadata_carries_join_keys(tmp_path: Path):
    """The four columns FILTER_EMBEDDINGS/FILTER_CP_FEATURES join QC's
    output back on -- the reason this stage exists rather than pointing
    QC_FILTER straight at cell_table.parquet."""
    cell_table = _write_cell_table(
        tmp_path / "cell_table.parquet",
        [_row("well1", "tile0x0y", 0, "bc1", "A1A", 0)],
    )

    result = build_cell_metadata(_cfg(cell_table))

    assert set(JOIN_KEYS).issubset(result.columns)


def test_build_cell_metadata_drops_features_and_the_variant_class(tmp_path: Path):
    """No feature columns -- QC only ever needs the seven meta_* fields --
    and no meta_variant_class: QC and every stage after it derive the
    class from meta_aa_changes themselves."""
    cell_table = _write_cell_table(
        tmp_path / "cell_table.parquet",
        [_row("well1", "tile0x0y", 0, "bc1", "A1A", 0, Cells_AreaShape_Area=1.0)],
    )

    result = build_cell_metadata(_cfg(cell_table))

    assert "Cells_AreaShape_Area" not in result.columns
    assert "meta_variant_class" not in result.columns


def test_build_cell_metadata_empty_table_keeps_schema(tmp_path: Path):
    cell_table = tmp_path / "cell_table.parquet"
    cell_table.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(schema=CELL_META_SCHEMA).write_parquet(cell_table)

    result = build_cell_metadata(_cfg(cell_table))

    assert result.height == 0
    assert result.schema == CELL_METADATA_SCHEMA


def test_projection_matches_cp_features(tmp_path: Path):
    """BUILD_CELL_METADATA and BUILD_CP_FEATURES must agree cell-for-cell
    on the meta_* columns -- FILTER_CP_FEATURES inner-joins one against
    QC's filtering of the other."""
    cell_images_dir = tmp_path / "cell_images"
    cell_table = _write_cell_table(
        cell_images_dir / "cell_table.parquet",
        [
            _row("well1", "tile0x0y", 0, "bc1", "A1A", 0, Cells_AreaShape_Area=1.0),
            _row("well1", "tile0x0y", 1, "bc2", "A1B", 1, Cells_AreaShape_Area=2.0),
        ],
    )

    metadata = build_cell_metadata(_cfg(cell_table, batch_stem="b"))
    cp_features = build_cp_features(
        CpFeaturesConfig(
            output_dir="/tmp/out",
            cell_images_dir=str(cell_images_dir),
            batch_stem="b",
        )
    )

    meta_cols = list(CELL_METADATA_SCHEMA)
    assert metadata.equals(cp_features.select(meta_cols))


# ---------------------------------------------------------------------------
# Hydra main()
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    cell_table = _write_cell_table(
        tmp_path / "cell_images" / "cell_table.parquet",
        [
            _row("well1", "tile0x0y", 0, "bc1", "A1A", 0),
            _row("well1", "tile0x0y", 1, "bc2", "A1B", 0),
        ],
    )
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.cell_metadata",
            f"output_dir={output_dir}",
            f"cell_table={cell_table}",
            "batch_stem=cli_batch",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out_path = output_dir / "metadata.parquet"
    assert out_path.exists()
    metadata = pl.read_parquet(out_path)
    assert metadata.height == 2
    assert metadata[META_BATCH_COL].unique().to_list() == ["cli_batch"]
    assert metadata.columns == list(CELL_METADATA_SCHEMA)


def test_main_is_hydra_entry_point():
    assert callable(main)
