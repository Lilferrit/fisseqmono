"""Tests for snakemake/fisseq_targets.py: every file the nested snakemake's
`fisseq_shards`/`fisseq_tiles_manifest` rules ask for, and the manifest rows
`build_cell_images_table.py` reads.

The module runs inside the ops env's snakemake interpreter, not this
package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "snakemake" / "fisseq_targets.py"
_spec = importlib.util.spec_from_file_location("fisseq_targets", _PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _rows(**overrides):
    kwargs = dict(
        phenotyping_dir="/e/phenotyping/",
        sequencing_dir="/e/sequencing/",
        wells=["well1"],
        grid_size=1,
        segmentation_type="cells",
    )
    kwargs.update(overrides)
    return mod.tile_rows(**kwargs)


def _shard_dirs(**overrides):
    kwargs = dict(
        phenotyping_dir="/e/phenotyping/",
        wells=["well1"],
        grid_size=1,
        segmentation_type="cells",
        image="raw",
        window=224,
        shard_size=None,
    )
    kwargs.update(overrides)
    return mod.well_shard_dirs(**kwargs)


def test_tile_names_cover_the_whole_grid_named_like_starcall():
    assert mod.tile_names(2) == ["tile00x00y", "tile00x01y", "tile01x00y", "tile01x01y"]


def test_tile_rows_name_every_file_of_a_tile():
    (row,) = _rows()
    tile_dir = "/e/phenotyping/well1_grid1/tile00x00y"
    assert row == {
        "well": "well1",
        "tile": "tile00x00y",
        "segmentation_csv": f"{tile_dir}/cells.csv",
        "reads_csv": "/e/sequencing/well1_grid1/tile00x00y/cells_reads.csv",
        "cellprofiler_csv": "",
    }
    assert list(row) == mod.MANIFEST_FIELDNAMES


def test_tile_rows_join_dirs_with_or_without_trailing_slash():
    assert _rows() == _rows(
        phenotyping_dir="/e/phenotyping", sequencing_dir="/e/sequencing"
    )


def test_tile_rows_span_every_well_and_tile():
    rows = _rows(wells=["well1", "well2"], grid_size=3)
    assert len(rows) == 18
    assert [r["well"] for r in rows[:9]] == ["well1"] * 9
    assert rows[9]["segmentation_csv"].startswith(
        "/e/phenotyping/well2_grid3/tile00x00y/"
    )


def test_well_shard_dir_sits_next_to_the_tiles():
    assert _shard_dirs(wells=["well1", "well3"], grid_size=20) == [
        {
            "well": "well1",
            "shard_dir": "/e/phenotyping/well1_grid20/cells_raw_shards_224_all",
        },
        {
            "well": "well3",
            "shard_dir": "/e/phenotyping/well3_grid20/cells_raw_shards_224_all",
        },
    ]


def test_well_shard_dir_is_named_by_image_window_and_shard_size():
    """Changing any of them requests new shards instead of reusing stale
    ones."""
    (row,) = _shard_dirs(image="corrected", window=180, shard_size=1000)
    assert row["shard_dir"].endswith("/well1_grid1/cells_corrected_shards_180_1000")


def test_tile_shard_tars_are_every_tile_of_the_well_in_order():
    tars = mod.tile_shard_tars("/e/phenotyping/well1_grid2/", 2, "cells", "raw", 224)
    assert tars == [
        f"/e/phenotyping/well1_grid2/{tile}/cells_raw_shard_224.tar"
        for tile in mod.tile_names(2)
    ]


def test_shard_rows_list_each_wells_shards_in_order(tmp_path: Path):
    dirs = []
    for well, names in (
        ("well1", ["well_1_shard_000001.tar.gz", "well_1_shard_000000.tar.gz"]),
        ("well2", ["well_2_shard_000000.tar.gz"]),
    ):
        shard_dir = tmp_path / well
        shard_dir.mkdir()
        for name in names:
            (shard_dir / name).touch()
        dirs.append({"well": well, "shard_dir": str(shard_dir)})
    (tmp_path / "well1" / "make_well_shards.log").touch()

    rows = mod.shard_rows(dirs)

    assert rows == [
        {
            "well": "well1",
            "shard_tar": str(tmp_path / "well1" / "well_1_shard_000000.tar.gz"),
        },
        {
            "well": "well1",
            "shard_tar": str(tmp_path / "well1" / "well_1_shard_000001.tar.gz"),
        },
        {
            "well": "well2",
            "shard_tar": str(tmp_path / "well2" / "well_2_shard_000000.tar.gz"),
        },
    ]
    assert list(rows[0]) == mod.SHARDS_MANIFEST_FIELDNAMES


def test_reads_csv_carries_sequencing_reads_params():
    (row,) = _rows(sequencing_reads_params="_ed1")
    assert row["reads_csv"].endswith("/cells_reads_ed1.csv")


def test_cellprofiler_csv_only_with_cp_features():
    (row,) = _rows(
        cp_features=True, cellprofiler_cycle="cycle0", cellprofiler_pipeline="p"
    )
    assert row["cellprofiler_csv"] == (
        "/e/phenotyping/well1_grid1/tile00x00y/cellprofilercycle0_p.csv"
    )


def test_row_targets_are_the_tables_only():
    """Never the whole-tile image/mask: those stay temp() upstream, so
    snakemake can delete them."""
    (row,) = _rows()
    targets = mod.row_targets([row])
    assert targets == [row["segmentation_csv"], row["reads_csv"]]


def test_row_targets_include_the_cellprofiler_csv_when_set():
    rows = _rows(cp_features=True, cellprofiler_pipeline="p")
    assert mod.row_targets(rows)[-1] == rows[0]["cellprofiler_csv"]


def test_shard_targets_are_only_the_shard_dirs():
    """The fisseq_shards pass: no table, so nothing downstream of the
    shards' temp() image and mask is in that DAG."""
    dirs = _shard_dirs(wells=["well1", "well2"], grid_size=2)
    assert mod.shard_targets(dirs) == [row["shard_dir"] for row in dirs]
