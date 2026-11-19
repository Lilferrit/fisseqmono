"""Tests for well_shards -- the nested snakemake's make_well_shards rule."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
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


def test_no_shard_max_bytes_packs_the_whole_well_into_one_shard(tmp_path: Path):
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


def _noisy_tiles(tmp_path: Path, counts: list[int]) -> list[str]:
    """Tile shards whose crops don't compress, so each sample adds a
    predictable ~32 KiB to a shard on disk."""
    rng = np.random.default_rng(0)
    tars = []
    for t, n_cells in enumerate(counts):
        tile = f"tile{t:02}x00y"
        path = tmp_path / f"noisy{t:02}.tar"
        with wds.TarWriter(str(path)) as sink:
            for i in range(n_cells):
                sink.write(
                    {
                        "__key__": f"well1_{tile}_{i}",
                        "crop.npy": rng.integers(
                            0, 2**16, (4, 64, 64), dtype=np.uint16
                        ),
                        "mask.npy": np.ones((64, 64), dtype=np.uint8),
                        "meta.json": {"meta_well": "well1", "meta_tile": tile},
                    }
                )
        tars.append(str(path))
    return tars


def test_shard_max_bytes_splits_across_tiles_in_order(tmp_path: Path):
    """A shard closes after the sample that takes it to shard_max_bytes on
    disk, regardless of tile boundaries; samples stay in order."""
    tile_tars = _noisy_tiles(tmp_path, [3, 2, 2])

    paths, n_cells = write_well_shards(
        tile_tars, "well1", 70_000, str(tmp_path / "out")
    )

    assert n_cells == 7
    assert [Path(p).name for p in paths] == [
        "well_1_shard_000000.tar.gz",
        "well_1_shard_000001.tar.gz",
        "well_1_shard_000002.tar.gz",
    ]
    # ~32 KiB per sample: the third sample takes a shard past 70 kB.
    assert [len(_keys(p)) for p in paths] == [3, 3, 1]
    assert [k for p in paths for k in _keys(p)] == [
        f"well1_tile{t:02}x00y_{i}" for t, n in enumerate([3, 2, 2]) for i in range(n)
    ]


def test_shards_stay_near_shard_max_bytes(tmp_path: Path):
    """Every shard but the last reaches the limit, and overshoots it by at
    most one sample plus what gzip had buffered (GzipFile's 32 KiB, zlib's
    own)."""
    tile_tars = _noisy_tiles(tmp_path, [20])
    limit = 200_000

    paths, _ = write_well_shards(tile_tars, "well1", limit, str(tmp_path / "out"))

    sizes = [Path(p).stat().st_size for p in paths]
    assert len(sizes) > 1
    for size in sizes[:-1]:
        assert limit <= size < limit + 80_000
    assert sizes[-1] < limit + 80_000


def test_a_sample_bigger_than_shard_max_bytes_gets_its_own_shard(tmp_path: Path):
    paths, _ = write_well_shards(
        _noisy_tiles(tmp_path, [2]), "well1", 1, str(tmp_path / "out")
    )
    assert [len(_keys(p)) for p in paths] == [1, 1]


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


@pytest.mark.parametrize("shard_max_bytes", [0, -1])
def test_rejects_a_non_positive_shard_max_bytes(tmp_path: Path, shard_max_bytes: int):
    with pytest.raises(ValueError, match="shard_max_bytes"):
        write_well_shards([], "well1", shard_max_bytes, str(tmp_path / "out"))


def test_embed_cells_reads_the_gzipped_shards(tmp_path: Path):
    """EMBED_CELLS' dataloader streams the packed shards as they are."""
    paths, _ = write_well_shards(
        _noisy_tiles(tmp_path, [2, 1]), "well1", 1, str(tmp_path / "out")
    )
    assert len(paths) == 3
    cfg = EmbedCellsConfig(
        output_dir=str(tmp_path),
        shards_dir=str(tmp_path / "out"),
        batch_stem="batch1",
        checkpoint_path="unused",
        batch_size=10,
        num_workers=0,
    )

    keys = [k for batch_keys, *_ in load_embedding_dataloader(cfg) for k in batch_keys]

    assert keys == ["well1_tile00x00y_0", "well1_tile00x00y_1", "well1_tile01x00y_0"]


def test_main_runs_via_cli(tmp_path: Path):
    tile_list = tmp_path / "tile_tars.txt"
    tile_list.write_text("\n".join(_noisy_tiles(tmp_path, [2, 1])) + "\n")
    shard_dir = tmp_path / "phenotyping" / "well1_grid2" / "cells_raw_shards_224_1B"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.well_shards",
            f"output_dir={tmp_path / 'log'}",
            f"hydra.run.dir={tmp_path / 'log'}",
            f"tile_tars_file={tile_list}",
            "well=well1",
            "shard_max_bytes=1",
            f"shard_dir={shard_dir}",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in shard_dir.iterdir()) == [
        "well_1_shard_000000.tar.gz",
        "well_1_shard_000001.tar.gz",
        "well_1_shard_000002.tar.gz",
    ]
