"""Tests for well_shards -- the nested snakemake's make_well_shards rule."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import webdataset as wds

from fisseq_embeddings_pipeline.embed import EmbedCellsConfig, load_embedding_dataloader
from fisseq_embeddings_pipeline.well_shards import shard_filename, write_well_shards


def _tile_tar(path: Path, tile: str, n_cells: int) -> str:
    """A tile shard the way tile_shard.py writes one."""
    with wds.TarWriter(str(path)) as sink:
        for i in range(n_cells):
            sink.write(
                {
                    "__key__": f"well1_{tile}_{i}",
                    "crop.npy": np.full((2, 4, 4), i, dtype=np.uint16),
                    "mask.npy": np.ones((4, 4), dtype=np.uint8),
                    "meta.json": {"meta_well": "well1", "meta_tile": tile},
                }
            )
    return str(path)


def _tiles(tmp_path: Path, counts: list[int]) -> list[str]:
    return [
        _tile_tar(tmp_path / f"tile{i:02}.tar", f"tile{i:02}x00y", n)
        for i, n in enumerate(counts)
    ]


def _keys(shard: str) -> list[str]:
    dataset = wds.WebDataset(shard, shardshuffle=False, empty_check=False)
    return [sample["__key__"] for sample in dataset]


def test_shard_filename_uses_the_well_number_zero_padded():
    assert shard_filename("well3", 0) == "well_3_shard_000000.tar.gz"
    assert shard_filename("well12", 41) == "well_12_shard_000041.tar.gz"
    assert shard_filename("A1", 2) == "well_A1_shard_000002.tar.gz"


def test_no_shard_size_packs_the_whole_well_into_one_shard(tmp_path: Path):
    tile_tars = _tiles(tmp_path, [2, 0, 3])

    paths, n_cells = write_well_shards(tile_tars, "well1", None, str(tmp_path / "out"))

    assert [Path(p).name for p in paths] == ["well_1_shard_000000.tar.gz"]
    assert n_cells == 5
    assert _keys(paths[0]) == [
        "well1_tile00x00y_0",
        "well1_tile00x00y_1",
        "well1_tile02x00y_0",
        "well1_tile02x00y_1",
        "well1_tile02x00y_2",
    ]


def test_shard_size_splits_across_tiles_in_order(tmp_path: Path):
    """Shards fill to shard_size cells regardless of tile boundaries; the
    last one holds the remainder."""
    tile_tars = _tiles(tmp_path, [3, 2, 2])

    paths, _ = write_well_shards(tile_tars, "well1", 3, str(tmp_path / "out"))

    assert [Path(p).name for p in paths] == [
        "well_1_shard_000000.tar.gz",
        "well_1_shard_000001.tar.gz",
        "well_1_shard_000002.tar.gz",
    ]
    assert [len(_keys(p)) for p in paths] == [3, 3, 1]
    assert _keys(paths[1]) == [
        "well1_tile01x00y_0",
        "well1_tile01x00y_1",
        "well1_tile02x00y_0",
    ]


def test_shards_are_gzipped(tmp_path: Path):
    (path,), _ = write_well_shards(
        _tiles(tmp_path, [1]), "well1", None, str(tmp_path / "out")
    )
    assert Path(path).read_bytes()[:2] == b"\x1f\x8b"


def test_samples_round_trip_unchanged(tmp_path: Path):
    (path,), _ = write_well_shards(
        _tiles(tmp_path, [2]), "well1", None, str(tmp_path / "out")
    )
    samples = list(wds.WebDataset(path, shardshuffle=False).decode())
    np.testing.assert_array_equal(
        samples[1]["crop.npy"], np.full((2, 4, 4), 1, dtype=np.uint16)
    )
    np.testing.assert_array_equal(samples[1]["mask.npy"], np.ones((4, 4)))
    assert samples[1]["meta.json"] == {"meta_well": "well1", "meta_tile": "tile00x00y"}


def test_empty_well_still_gets_one_empty_shard(tmp_path: Path):
    paths, n_cells = write_well_shards(
        _tiles(tmp_path, [0, 0]), "well1", 10, str(tmp_path / "out")
    )
    assert [Path(p).name for p in paths] == ["well_1_shard_000000.tar.gz"]
    assert n_cells == 0
    assert _keys(paths[0]) == []


@pytest.mark.parametrize("shard_size", [0, -1])
def test_rejects_a_non_positive_shard_size(tmp_path: Path, shard_size: int):
    with pytest.raises(ValueError, match="shard_size"):
        write_well_shards([], "well1", shard_size, str(tmp_path / "out"))


def test_embed_cells_reads_the_gzipped_shards(tmp_path: Path):
    """EMBED_CELLS' dataloader streams the packed shards as they are."""
    paths, _ = write_well_shards(
        _tiles(tmp_path, [2, 1]), "well1", 2, str(tmp_path / "out")
    )
    shards_path = tmp_path / "shards.parquet"
    pl.DataFrame({"well": ["well1"] * len(paths), "shard_tar": paths}).write_parquet(
        shards_path
    )
    cfg = EmbedCellsConfig(
        output_dir=str(tmp_path),
        shards_path=str(shards_path),
        batch_stem="batch1",
        checkpoint_path="unused",
        batch_size=10,
        num_workers=0,
    )

    keys = [k for batch_keys, *_ in load_embedding_dataloader(cfg) for k in batch_keys]

    assert keys == ["well1_tile00x00y_0", "well1_tile00x00y_1", "well1_tile01x00y_0"]


def test_main_runs_via_cli(tmp_path: Path):
    tile_list = tmp_path / "tile_tars.txt"
    tile_list.write_text("\n".join(_tiles(tmp_path, [2, 1])) + "\n")
    shard_dir = tmp_path / "phenotyping" / "well1_grid2" / "cells_raw_shards_224_2"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.well_shards",
            f"output_dir={tmp_path / 'log'}",
            f"hydra.run.dir={tmp_path / 'log'}",
            f"tile_tars_file={tile_list}",
            "well=well1",
            "shard_size=2",
            f"shard_dir={shard_dir}",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in shard_dir.iterdir()) == [
        "well_1_shard_000000.tar.gz",
        "well_1_shard_000001.tar.gz",
    ]
