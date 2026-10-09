"""BUILD_CELL_METADATA.

Projects BUILD_CELL_IMAGES' ``cell_table.parquet`` down to the seven
``meta_*`` columns QC_FILTER reads (``utils.cell_table.CELL_METADATA_SCHEMA``:
``meta_batch``, added here, plus the table's own key and QC fields), as one
per-experiment ``metadata.parquet`` -- no images, no feature columns, no
starcall-workflow tree access.

This stage exists to make QC_FILTER the pipeline's shared fan-out point,
independent of the image-reading cellDINO track: QC depends only on the
cell table, so QC thresholds can be retuned (and the whole QC report
regenerated) without touching the shards at all.

Structurally the analogue of ``fisseq-data-pipeline``'s own ``INPUT``
stage: the cheap read-and-normalize step that turns what the upstream
produced into the one per-batch parquet QC_FILTER consumes. The table
already carries canonical ``meta_*`` names (BUILD_CELL_IMAGES renames the
reads tables' genotype columns), so all this adds is ``meta_batch`` --
which neither the table nor the shards carry, being a run-level name --
and it leaves out ``meta_variant_class``, which QC and every stage after
it derive from ``meta_aa_changes`` themselves.

EMBED_CELLS builds the same seven columns from each shard sample's
``meta.json``, written from the same per-tile CSVs by the same function
(``build_cell_images_table.tile_cell_meta``), so a cell's ``meta_*``
values are identical in QC's input and in ``embeddings.parquet``.

Note this stage sees every row of ``cell_table.parquet``. Every downstream
consumer joins ``filtered_cells.parquet`` back on ``JOIN_KEYS`` with an
inner join, so any extra rows drop out where they don't apply.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.utils.log import setup_logging

from .config import AppConfig
from .utils.cell_table import CELL_METADATA_SCHEMA, cell_metadata_exprs


@dataclasses.dataclass
class CellMetadataConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CELL_METADATA.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    cell_table : str
        Path to BUILD_CELL_IMAGES' ``cell_table.parquet``. Unlike
        ``CpFeaturesConfig.cell_images_dir``, this is the file itself,
        not the directory holding it: the BUILD_CELL_METADATA process
        gets it straight from BUILD_CELL_IMAGES' output.
    batch_stem : str
        This experiment's identifier, written into every row as
        ``meta_batch`` -- one run covers exactly one experiment, matching
        BUILD_CP_FEATURES' convention.
    """

    cell_table: str = MISSING
    batch_stem: str = MISSING


def build_cell_metadata(cfg: CellMetadataConfig) -> pl.DataFrame:
    """
    Project one experiment's cell table down to its ``meta_*`` columns.

    Parameters
    ----------
    cfg : CellMetadataConfig
        Supplies ``cell_table`` and ``batch_stem``.

    Returns
    -------
    pl.DataFrame
        One row per cell, with exactly ``CELL_METADATA_SCHEMA``'s columns:
        ``meta_batch``, ``meta_well``, ``meta_tile``, ``meta_cell_index``,
        ``meta_barcode``, ``meta_aa_changes``, ``meta_edit_distance``.
        An empty frame carrying that schema if the cell table has no rows
        (nothing to infer column types from otherwise).
    """
    cell_table_path = pathlib.Path(cfg.cell_table)
    table = pl.read_parquet(cell_table_path)

    if table.height == 0:
        logging.info("cell table at %s has no rows", cell_table_path)
        return pl.DataFrame(schema=CELL_METADATA_SCHEMA)

    return table.select(*cell_metadata_exprs(cfg.batch_stem))


_cs = ConfigStore.instance()
_cs.store(name="cell_metadata_main", node=CellMetadataConfig)


@hydra.main(version_base=None, config_path=None, config_name="cell_metadata_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: build one experiment's QC metadata table.

    Steps
    -----
    1. Create ``output_dir``.
    2. Read ``cell_table`` and project it via :func:`build_cell_metadata`.
    3. Write ``{prefix}metadata.parquet`` to ``output_dir``.

    Output file
    ------------
    - ``{prefix}metadata.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.cell_metadata \\
            output_dir=./out \\
            cell_table=/pipeline/cell_images/experiment1/cell_table.parquet \\
            batch_stem=experiment1 \\
            random_seed=0
    """
    meta_cfg: CellMetadataConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(meta_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_cfg.output_dir = str(output_dir)
    setup_logging(meta_cfg, "cell_metadata")

    logging.info(
        "Building cell metadata for batch %s from %s",
        meta_cfg.batch_stem,
        meta_cfg.cell_table,
    )
    metadata_df = build_cell_metadata(meta_cfg)

    prefix = f"{meta_cfg.output_root}." if meta_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}metadata.parquet"
    logging.info("Writing %s (%d cells)", out_path, metadata_df.height)
    metadata_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
