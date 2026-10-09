"""BUILD_CP_FEATURES.

Input conversion for the CellProfiler-feature track: reads BUILD_CELL_IMAGES'
``cell_table.parquet`` and keeps its CellProfiler feature columns (every
column without a ``meta_`` prefix, under CellProfiler's own names), into one
per-experiment ``cp_features.parquet`` -- the CellProfiler-feature analog of
EMBED_CELLS' ``embeddings.parquet``, and the input to FILTER_CP_FEATURES.
Its ``meta_*`` columns are the same seven BUILD_CELL_METADATA writes
(``utils.cell_table.cell_metadata_exprs``), ``meta_batch`` included.

This module never touches starcall-workflow's tree: BUILD_CELL_IMAGES is
the only place in the pipeline that reads CellProfiler's raw output,
joining each tile's CellProfiler CSV to its cells by row position
(CellProfiler's own ``ObjectNumber`` numbering follows ascending mask-label
order, i.e. row position, not any shared index value).
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf
from polars import selectors as cs

from fisseq_common.utils.log import setup_logging

from .config import AppConfig
from .utils.cell_table import CELL_METADATA_SCHEMA, cell_metadata_exprs


@dataclasses.dataclass
class CpFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CP_FEATURES.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    BUILD_CP_FEATURES' own logic doesn't consume random_seed itself, but
    every stage config inherits it uniformly.

    Attributes
    ----------
    cell_images_dir : str
        BUILD_CELL_IMAGES' per-experiment output directory -- holds
        ``cell_table.parquet``, which carries this experiment's CellProfiler
        feature columns alongside its ``meta_*`` columns.
    batch_stem : str
        This experiment's identifier, written into every row as
        meta_batch -- one BUILD_CP_FEATURES run covers exactly one
        experiment, matching BUILD_CELL_METADATA's convention.
    """

    cell_images_dir: str = MISSING
    batch_stem: str = MISSING


def build_cp_features(cfg: CpFeaturesConfig) -> pl.DataFrame:
    """
    Select this experiment's CellProfiler feature columns out of
    BUILD_CELL_IMAGES' ``cell_table.parquet``.

    Parameters
    ----------
    cfg : CpFeaturesConfig
        Supplies ``cell_images_dir`` and ``batch_stem``.

    Returns
    -------
    pl.DataFrame
        One row per cell: ``meta_batch``, ``meta_well``, ``meta_tile``,
        ``meta_cell_index``, ``meta_barcode``, ``meta_aa_changes``,
        ``meta_edit_distance``, plus every CellProfiler feature column,
        in the table's order.
    """
    cell_table_path = pathlib.Path(cfg.cell_images_dir) / "cell_table.parquet"
    table = pl.read_parquet(cell_table_path)

    if table.height == 0:
        logging.info("cell_table.parquet at %s has no rows", cell_table_path)
        return pl.DataFrame(schema=CELL_METADATA_SCHEMA)

    if not table.select(~cs.starts_with("meta_")).columns:
        logging.warning(
            "No CellProfiler feature columns found in %s -- "
            "was this experiment's cp_features flag actually enabled when "
            "BUILD_CELL_IMAGES ran?",
            cell_table_path,
        )

    return table.select(*cell_metadata_exprs(cfg.batch_stem), ~cs.starts_with("meta_"))


_cs = ConfigStore.instance()
_cs.store(name="cp_features_main", node=CpFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="cp_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: build one experiment's CellProfiler-feature table.

    Steps
    -----
    1. Create ``output_dir``.
    2. Read ``cell_images_dir/cell_table.parquet`` and select this
       experiment's CellProfiler feature columns via :func:`build_cp_features`.
    3. Write ``{prefix}cp_features.parquet`` to ``output_dir``.

    Output file
    ------------
    - ``{prefix}cp_features.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.cp_features \\
            output_dir=./out \\
            cell_images_dir=/pipeline/cell_images/experiment1 \\
            batch_stem=experiment1 \\
            random_seed=0
    """
    cp_cfg: CpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(cp_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cp_cfg.output_dir = str(output_dir)
    setup_logging(cp_cfg, "cp_features")

    logging.info(
        "Building CellProfiler features for batch %s from %s",
        cp_cfg.batch_stem,
        cp_cfg.cell_images_dir,
    )
    features_df = build_cp_features(cp_cfg)

    prefix = f"{cp_cfg.output_root}." if cp_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}cp_features.parquet"
    logging.info("Writing %s (%d cells)", out_path, features_df.height)
    features_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
