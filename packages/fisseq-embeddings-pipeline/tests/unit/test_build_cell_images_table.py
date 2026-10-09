"""Tests for BUILD_CELL_IMAGES' build-table phase
(fisseq_embeddings_pipeline.build_cell_images_table).

Covers `tile_cell_meta` -- the per-cell meta columns both
`cell_table.parquet` and every shard sample's `meta.json` come from: the
index-value join between a tile's segmentation and reads tables (the
highest-correctness-risk logic in this stage), the genotype renames and the
variant class -- then the row-position CellProfiler join and the cross-tile
`diagonal_relaxed` concat.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import polars as pl
import pytest
import yaml

from fisseq_embeddings_pipeline import build_cell_images_table as mod
from fisseq_embeddings_pipeline.utils.cell_table import CELL_META_SCHEMA

_COLUMNS = mod.GenotypeColumns()


def _write_segmentation_csv(
    path: Path, index: Sequence[int], bbox_x1: Sequence[int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "bbox_x1": list(bbox_x1),
            "bbox_y1": [0] * len(bbox_x1),
            "bbox_x2": list(bbox_x1),
            "bbox_y2": [0] * len(bbox_x1),
            "orig_index": list(index),
            "mask8": [0] * len(bbox_x1),
        },
        index=list(index),
    ).to_csv(path)


def _write_reads_csv(
    path: Path,
    index: Sequence[int],
    edit_distance: Sequence[int],
    barcode: Sequence[str],
    aa_changes: Sequence[str | None] | None = None,
    **extra: Sequence,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "editDistance": list(edit_distance),
            "upBarcode": list(barcode),
            "aaChanges": list(aa_changes or ["WT"] * len(barcode)),
            **{k: list(v) for k, v in extra.items()},
        },
        index=list(index),
    ).to_csv(path)


def _tile(tmp_path: Path, name: str = "t") -> tuple[Path, Path]:
    return tmp_path / name / "cells.csv", tmp_path / name / "cells_reads.csv"


# ---------------------------------------------------------------------------
# tile_cell_meta -- the index-value join, renames and variant class
# ---------------------------------------------------------------------------


def test_tile_cell_meta_joins_by_index_value(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[10, 20, 30])
    _write_reads_csv(
        reads_csv,
        index=[1, 2, 3],
        edit_distance=[0, 1, 2],
        barcode=["bcA", "bcB", "bcC"],
        aa_changes=["A12V", "A12A", "WT"],
    )

    meta = mod.tile_cell_meta(
        str(seg_csv), str(reads_csv), "well1", "tile0x0y", _COLUMNS
    )

    assert meta.schema == pl.Schema(CELL_META_SCHEMA)
    assert meta.to_dicts() == [
        {
            "meta_well": "well1",
            "meta_tile": "tile0x0y",
            "meta_cell_index": 1,
            "meta_barcode": "bcA",
            "meta_aa_changes": "A12V",
            "meta_edit_distance": 0,
            "meta_variant_class": "Single Missense",
        },
        {
            "meta_well": "well1",
            "meta_tile": "tile0x0y",
            "meta_cell_index": 2,
            "meta_barcode": "bcB",
            "meta_aa_changes": "A12A",
            "meta_edit_distance": 1,
            "meta_variant_class": "Synonymous",
        },
        {
            "meta_well": "well1",
            "meta_tile": "tile0x0y",
            "meta_cell_index": 3,
            "meta_barcode": "bcC",
            "meta_aa_changes": "WT",
            "meta_edit_distance": 2,
            "meta_variant_class": "WT",
        },
    ]


def test_tile_cell_meta_keeps_the_segmentation_row_order(tmp_path: Path):
    """The reads table may be in another order on disk; the rows still
    follow the segmentation table's, which is the mask-label order."""
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[10, 20, 30])
    _write_reads_csv(
        reads_csv,
        index=[3, 2, 1],
        edit_distance=[2, 1, 0],
        barcode=["bcC", "bcB", "bcA"],
    )

    meta = mod.tile_cell_meta(str(seg_csv), str(reads_csv), "w", "t", _COLUMNS)

    assert meta["meta_cell_index"].to_list() == [1, 2, 3]
    assert meta["meta_barcode"].to_list() == ["bcA", "bcB", "bcC"]
    assert meta["meta_edit_distance"].to_list() == [0, 1, 2]


def test_tile_cell_meta_raises_on_index_set_mismatch(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[10, 20, 30])
    _write_reads_csv(
        reads_csv, index=[1, 2, 4], edit_distance=[0, 1, 2], barcode=["a", "b", "c"]
    )

    with pytest.raises(ValueError, match="different tile_cell_index sets"):
        mod.tile_cell_meta(str(seg_csv), str(reads_csv), "w", "t", _COLUMNS)


def test_tile_cell_meta_renames_per_experiment_genotype_columns(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[1], bbox_x1=[1])
    reads_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bc": ["x"], "variant": ["A1V"], "ed": [3]}, index=[1]).to_csv(
        reads_csv
    )

    meta = mod.tile_cell_meta(
        str(seg_csv),
        str(reads_csv),
        "w",
        "t",
        mod.GenotypeColumns(barcode="bc", aa_changes="variant", edit_distance="ed"),
    )

    assert meta.select("meta_barcode", "meta_aa_changes", "meta_edit_distance").row(
        0
    ) == ("x", "A1V", 3)


def test_tile_cell_meta_names_a_missing_genotype_column(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[1], bbox_x1=[1])
    _write_reads_csv(reads_csv, index=[1], edit_distance=[0], barcode=["a"])

    with pytest.raises(ValueError, match="no column.*'barcode_x'"):
        mod.tile_cell_meta(
            str(seg_csv),
            str(reads_csv),
            "w",
            "t",
            mod.GenotypeColumns(barcode="barcode_x"),
        )


def test_tile_cell_meta_keeps_a_missing_genotype_null(tmp_path: Path):
    """A cell with no reads has no barcode, variant or class -- null, never a
    made-up value (architecture decision 19)."""
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[1, 2], bbox_x1=[1, 2])
    reads_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "editDistance": [0, None],
            "upBarcode": ["a", None],
            "aaChanges": ["WT", None],
        },
        index=[1, 2],
    ).to_csv(reads_csv)

    meta = mod.tile_cell_meta(str(seg_csv), str(reads_csv), "w", "t", _COLUMNS)

    assert meta.row(1, named=True) == {
        "meta_well": "w",
        "meta_tile": "t",
        "meta_cell_index": 2,
        "meta_barcode": None,
        "meta_aa_changes": None,
        "meta_edit_distance": None,
        "meta_variant_class": None,
    }


def test_tile_cell_meta_empty_tile(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[], bbox_x1=[])
    _write_reads_csv(reads_csv, index=[], edit_distance=[], barcode=[])

    meta = mod.tile_cell_meta(str(seg_csv), str(reads_csv), "w", "t", _COLUMNS)

    assert meta.height == 0
    assert meta.schema == pl.Schema(CELL_META_SCHEMA)


# ---------------------------------------------------------------------------
# build_tile_table -- the CellProfiler row-position join
# ---------------------------------------------------------------------------


def test_build_tile_table_is_the_meta_alone_without_cellprofiler(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    _write_segmentation_csv(seg_csv, index=[1, 2], bbox_x1=[1, 2])
    _write_reads_csv(
        reads_csv,
        index=[1, 2],
        edit_distance=[0, 0],
        barcode=["a", "b"],
        otherAuxCol=["x", "y"],
    )

    table = mod.build_tile_table(
        str(seg_csv), str(reads_csv), None, "well1", "tile0x0y", _COLUMNS
    )

    assert table.columns == list(CELL_META_SCHEMA)


def test_build_tile_table_folds_in_cellprofiler_by_row_position(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    cp_csv = tmp_path / "t" / "cellprofiler_my_pipeline.csv"
    _write_segmentation_csv(seg_csv, index=[10, 11, 12], bbox_x1=[1, 2, 3])
    _write_reads_csv(
        reads_csv, index=[10, 11, 12], edit_distance=[0, 0, 0], barcode=["a", "b", "c"]
    )
    # CellProfiler's own ObjectNumber-style index (1, 2, 3), deliberately
    # not matching tile_cell_index (10, 11, 12) -- the join must be by row
    # position, not index value.
    pd.DataFrame(
        {
            "Cells_AreaShape_Area": [100.0, 200.0, 300.0],
            "Nuclei_AreaShape_Area": [1.0, 2.0, 3.0],
        },
        index=[1, 2, 3],
    ).to_csv(cp_csv)

    table = mod.build_tile_table(
        str(seg_csv), str(reads_csv), str(cp_csv), "well1", "tile0x0y", _COLUMNS
    )

    # The CellProfiler columns keep their own names, after the meta columns.
    assert table.columns == [
        *CELL_META_SCHEMA,
        "Cells_AreaShape_Area",
        "Nuclei_AreaShape_Area",
    ]
    assert table["meta_cell_index"].to_list() == [10, 11, 12]
    assert table["Cells_AreaShape_Area"].to_list() == [100.0, 200.0, 300.0]


def test_build_tile_table_raises_on_cellprofiler_row_count_mismatch(tmp_path: Path):
    seg_csv, reads_csv = _tile(tmp_path)
    cp_csv = tmp_path / "t" / "cellprofiler_my_pipeline.csv"
    _write_segmentation_csv(seg_csv, index=[1, 2, 3], bbox_x1=[1, 2, 3])
    _write_reads_csv(
        reads_csv, index=[1, 2, 3], edit_distance=[0, 0, 0], barcode=["a", "b", "c"]
    )
    pd.DataFrame({"Cells_AreaShape_Area": [100.0, 200.0]}, index=[1, 2]).to_csv(cp_csv)

    with pytest.raises(ValueError, match="row-position join"):
        mod.build_tile_table(
            str(seg_csv), str(reads_csv), str(cp_csv), "well1", "tile0x0y", _COLUMNS
        )


# ---------------------------------------------------------------------------
# build_cell_table -- cross-tile diagonal_relaxed concat
# ---------------------------------------------------------------------------


def _tile_info(seg: Path, reads: Path, tile: str, cp: Path | None = None) -> dict:
    return {
        "well": "well1",
        "tile": tile,
        "segmentation_csv": str(seg),
        "reads_csv": str(reads),
        "cellprofiler_csv": str(cp) if cp else "",
    }


def test_build_cell_table_concatenates_across_tiles_with_differing_schemas(
    tmp_path: Path,
):
    """Tiles' CellProfiler columns can differ -- concat must not fail, and
    must fill missing columns with nulls rather than drop rows/columns."""
    tiles = []
    for name, cp_cols in (("t1", {"A_x": [1.0]}), ("t2", {"A_x": [2.0], "B_y": [3.0]})):
        seg, reads = _tile(tmp_path, name)
        cp = tmp_path / name / "cp.csv"
        _write_segmentation_csv(seg, index=[1], bbox_x1=[1])
        _write_reads_csv(reads, index=[1], edit_distance=[0], barcode=["a"])
        pd.DataFrame(cp_cols, index=[1]).to_csv(cp)
        tiles.append(_tile_info(seg, reads, f"tile{name}", cp))

    table = mod.build_cell_table(tiles, _COLUMNS)

    assert table.height == 2
    assert table["B_y"].to_list() == [None, 3.0]


def test_build_cell_table_empty_when_no_tiles():
    table = mod.build_cell_table([], _COLUMNS)
    assert table.height == 0
    assert table.schema == pl.Schema(CELL_META_SCHEMA)


def test_genotype_columns_come_from_the_snakemake_config():
    config = yaml.safe_load(
        "fisseq_barcode_col: bc\nfisseq_aa_changes_col: aa\nfisseq_edit_distance_col: ed\n"
    )
    assert mod.GenotypeColumns.from_snakemake_config(config) == mod.GenotypeColumns(
        "bc", "aa", "ed"
    )


# ---------------------------------------------------------------------------
# build_shards_table -- the shard sidecar
# ---------------------------------------------------------------------------


def test_build_shards_table_keeps_one_row_per_shard_in_order():
    shards = [
        {"well": "well1", "shard_tar": "/p/s/well_1_shard_000000.tar.gz"},
        {"well": "well1", "shard_tar": "/p/s/well_1_shard_000001.tar.gz"},
    ]

    table = mod.build_shards_table(shards)

    assert table.columns == ["well", "shard_tar"]
    assert table.to_dicts() == shards


def test_build_shards_table_empty_when_no_shards():
    table = mod.build_shards_table([])
    assert table.height == 0
    assert table.columns == ["well", "shard_tar"]
