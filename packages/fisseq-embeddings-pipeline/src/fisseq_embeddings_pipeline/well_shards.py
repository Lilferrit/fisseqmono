"""One well's WebDataset shards -- the nested snakemake's ``make_well_shards``.

Hydra entry point (`python -m fisseq_embeddings_pipeline.well_shards`), run
once per well by the ``make_well_shards`` rule this repo adds on top of
starcall-workflow's own Snakefile (``snakemake/Snakefile``). Packs the
well's per-tile shards (``tile_shard.py``, one ``make_cell_shard`` job per
tile, ``temp()`` so snakemake deletes them once this has run) into
gzipped shards of ``shard_size`` cells each, named
``well_{n}_shard_{k:06}.tar.gz`` (:func:`shard_filename`) --
or, with ``shard_size`` unset, one shard holding the whole well. See
``docs/architecture.md`` decision 17.

Samples are copied tar member by tar member, never decoded, in tile order
and then each tile's own order, so a sample's key and contents are exactly
what ``tile_shard.py`` wrote. A sample never straddles two shards.
"""

import dataclasses
import io
import logging
import pathlib
import tarfile
from typing import Iterator, List, Optional, Tuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.utils.log import setup_logging

from .config import AppConfig

# gzip's own default; tarfile's is 9, which is much slower for little gain
# on image data.
_COMPRESSLEVEL = 6

_Sample = List[Tuple[tarfile.TarInfo, bytes]]


@dataclasses.dataclass
class WellShardsConfig(AppConfig):
    """
    Hydra structured configuration for one well's shards.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    ``output_dir`` holds only this run's log -- the shards go to
    ``shard_dir``.

    Attributes
    ----------
    tile_tars_file : str
        A text file listing the well's per-tile shards, one path per line,
        in the order their cells are written.
    well : str
        The well's name (``well3``), which names its shards.
    shard_size : int or None
        Cells per shard; the last shard holds the remainder. ``None`` (the
        default) writes the whole well into one shard.
    shard_dir : str
        Directory to write the shards into.
    """

    tile_tars_file: str = MISSING
    well: str = MISSING
    shard_size: Optional[int] = None
    shard_dir: str = MISSING


def shard_filename(well: str, index: int) -> str:
    """``well_{n}_shard_{index:06}.tar.gz``, ``n`` being the well's number
    (``well3`` -> ``3``); a well not named ``well<n>`` keeps its whole name.
    The zero padding keeps a well's shards in order when sorted by name."""
    number = well[len("well") :] if well.startswith("well") else well
    return f"well_{number}_shard_{index:06}.tar.gz"


def _samples(tar_path: str) -> Iterator[_Sample]:
    """Each sample of one tile's shard: its consecutive tar members sharing
    a key (the name up to its first ``.``, webdataset's own rule)."""
    with tarfile.open(tar_path) as src:
        key, sample = None, []
        for member in src:
            if not member.isfile():
                continue
            member_key = member.name.split(".", 1)[0]
            if sample and member_key != key:
                yield sample
                sample = []
            key = member_key
            sample.append((member, src.extractfile(member).read()))
        if sample:
            yield sample


def write_well_shards(
    tile_tars: List[str], well: str, shard_size: Optional[int], shard_dir: str
) -> Tuple[List[str], int]:
    """Pack ``tile_tars``' samples into ``shard_dir``.

    A well with no cells still gets one valid, empty shard, so every well
    has at least one.

    Returns
    -------
    (list[str], int)
        The shards written, in order, and the number of cells in them.

    Raises
    ------
    ValueError
        If ``shard_size`` is set and not positive.
    """
    if shard_size is not None and shard_size < 1:
        raise ValueError(f"shard_size must be null or positive, got {shard_size}.")
    out_dir = pathlib.Path(shard_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    sink: Optional[tarfile.TarFile] = None
    in_shard = 0
    n_cells = 0

    def next_shard() -> tarfile.TarFile:
        path = out_dir / shard_filename(well, len(paths))
        paths.append(str(path))
        return tarfile.open(path, "w:gz", compresslevel=_COMPRESSLEVEL)

    try:
        for tar_path in tile_tars:
            for sample in _samples(tar_path):
                if sink is None or (shard_size is not None and in_shard == shard_size):
                    if sink is not None:
                        sink.close()
                    sink = next_shard()
                    in_shard = 0
                for member, data in sample:
                    sink.addfile(member, io.BytesIO(data))
                in_shard += 1
                n_cells += 1
        if sink is None:
            logging.info("No cells in %s; writing an empty shard", well)
            sink = next_shard()
    finally:
        if sink is not None:
            sink.close()
    return paths, n_cells


_cs = ConfigStore.instance()
_cs.store(name="well_shards_main", node=WellShardsConfig)


@hydra.main(version_base=None, config_path=None, config_name="well_shards_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: pack one well's per-tile shards.

    Configuration
    -------------
    Normally run by the ``make_well_shards`` rule, e.g.::

        python -m fisseq_embeddings_pipeline.well_shards \\
            output_dir=/tmp/log \\
            tile_tars_file=/tmp/log/tile_tars.txt \\
            well=well1 shard_size=1000 \\
            shard_dir=phenotyping/well1_grid4/cells_raw_shards_224_1000
    """
    shards_cfg: WellShardsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(shards_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_cfg.output_dir = str(output_dir)
    setup_logging(shards_cfg, "well_shards")

    tile_tars = pathlib.Path(shards_cfg.tile_tars_file).read_text().splitlines()
    tile_tars = [p for p in tile_tars if p]
    paths, n_cells = write_well_shards(
        tile_tars, shards_cfg.well, shards_cfg.shard_size, shards_cfg.shard_dir
    )
    logging.info(
        "Wrote %d shard(s) of %s (%d cell(s) from %d tile(s)) to %s",
        len(paths),
        shards_cfg.well,
        n_cells,
        len(tile_tars),
        shards_cfg.shard_dir,
    )


if __name__ == "__main__":
    main()
