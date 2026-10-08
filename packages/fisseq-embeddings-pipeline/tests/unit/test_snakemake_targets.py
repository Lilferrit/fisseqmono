"""Tests for snakemake/fisseq_targets.py: every tile file the nested
snakemake's `fisseq_tiles_manifest` rule asks for, and the manifest rows
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
        image="raw",
        window=224,
    )
    kwargs.update(overrides)
    return mod.tile_rows(**kwargs)


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
        "shard_tar": f"{tile_dir}/cells_raw_shard_224.tar",
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
    assert rows[9]["shard_tar"].startswith("/e/phenotyping/well2_grid3/tile00x00y/")


def test_shard_is_named_by_image_and_window():
    """Changing either requests a new shard instead of reusing a stale one."""
    (row,) = _rows(image="corrected", window=180)
    assert row["shard_tar"].endswith("/cells_corrected_shard_180.tar")


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


def test_row_targets_skip_the_whole_tile_images():
    """The shard, not the image/mask it's cut from: those stay temp()
    upstream, so snakemake can delete them."""
    (row,) = _rows()
    targets = mod.row_targets([row])
    assert targets == [row["shard_tar"], row["segmentation_csv"], row["reads_csv"]]
    assert not any(t.endswith(".tif") for t in targets)


def test_row_targets_include_the_cellprofiler_csv_when_set():
    rows = _rows(cp_features=True, cellprofiler_pipeline="p")
    assert mod.row_targets(rows)[-1] == rows[0]["cellprofiler_csv"]
