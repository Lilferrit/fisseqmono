"""CONCAT_EMBEDDINGS.

Hydra entry point (`python -m fisseq_embeddings_pipeline.concat_embeddings`).
EMBED_CELLS runs once per well, so a GPU task killed partway loses one
well rather than the experiment (and ``-resume`` reruns only that well).
This stage stacks those per-well tables into the experiment's
``embeddings.parquet``, the table everything downstream reads.

The wells are stacked in natural order (``well2`` before ``well10``) and
streamed (``scan_parquet`` -> ``sink_parquet``), so the experiment's table
never has to fit in memory at once. A well with no cells has no ``emb_*``
columns (EMBED_CELLS writes just ``CELL_METADATA_SCHEMA`` for it); the
diagonal concat leaves it no rows to fill.
"""

import dataclasses
import logging
import pathlib
import re
from typing import List

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from polars import selectors as cs

from fisseq_common.utils.log import setup_logging

from .config import AppConfig
from .utils.cell_table import CELL_METADATA_SCHEMA


@dataclasses.dataclass
class ConcatEmbeddingsConfig(AppConfig):
    """
    Hydra structured configuration for CONCAT_EMBEDDINGS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    inputs_dir : str
        Directory holding one experiment's per-well EMBED_CELLS tables
        (``*.parquet``, e.g. ``embeddings_well1.parquet``); every one is
        stacked.
    output : str
        Output parquet filename (relative to ``output_dir``). Defaults to
        ``"embeddings.parquet"``.
    """

    inputs_dir: str = MISSING
    output: str = "embeddings.parquet"


def _natural_key(path: pathlib.Path) -> List[object]:
    """``embeddings_well10`` after ``embeddings_well2``."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def well_tables(inputs_dir: str) -> List[pathlib.Path]:
    """Every ``*.parquet`` in ``inputs_dir``, in natural name order.

    Raises
    ------
    ValueError
        If there are none.
    """
    paths = sorted(pathlib.Path(inputs_dir).glob("*.parquet"), key=_natural_key)
    if not paths:
        raise ValueError(f"inputs_dir={inputs_dir!r} holds no *.parquet tables")
    return paths


def concat_embeddings(paths: List[pathlib.Path]) -> pl.LazyFrame:
    """The wells' tables stacked, ``CELL_METADATA_SCHEMA`` first and then
    the ``emb_*`` columns, in EMBED_CELLS' own order."""
    stacked = pl.concat([pl.scan_parquet(p) for p in paths], how="diagonal_relaxed")
    return stacked.select(*CELL_METADATA_SCHEMA, cs.starts_with("emb_"))


_cs = ConfigStore.instance()
_cs.store(name="concat_embeddings_main", node=ConcatEmbeddingsConfig)


@hydra.main(version_base=None, config_path=None, config_name="concat_embeddings_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: stack one experiment's per-well embeddings.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.concat_embeddings \\
            output_dir=./out \\
            inputs_dir=./wells \\
            random_seed=0
    """
    concat_cfg: ConcatEmbeddingsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(concat_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concat_cfg.output_dir = str(output_dir)
    setup_logging(concat_cfg, "concat_embeddings")

    paths = well_tables(concat_cfg.inputs_dir)
    logging.info(
        "Stacking %d well(s): %s", len(paths), ", ".join(p.name for p in paths)
    )
    output_path = output_dir / concat_cfg.output
    concat_embeddings(paths).sink_parquet(output_path)
    n_cells = pl.scan_parquet(output_path).select(pl.len()).collect().item()
    logging.info("Wrote %s (%d cells)", output_path, n_cells)


if __name__ == "__main__":
    main()
