"""FILTER_EMBEDDINGS: QC-passed cell keys plus the synonymous-fitted Normalizer.

Hydra entry point (``python -m fisseq_embeddings_pipeline.filter``). The stage is shared with
fisseq-data-pipeline and documented in :mod:`fisseq_common.stages.filter`: it publishes only
the QC-passed join keys and the fitted normalizer, never a second copy of the embedding
matrix, and every downstream stage rebuilds the normalized embeddings with
:func:`load_filtered_embeddings`.

Here the controls are the untagged synonymous variants, and a cell is identified by
:data:`JOIN_KEYS`.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.stages import filter as _filter
from fisseq_common.stages.filter import (  # noqa: F401 (imported from here by other stages)
    SYNONYMOUS_CONTROL,
    FilterParams,
    variant_classification,
    write_filter_outputs,
)
from fisseq_common.utils.log import setup_logging

# The composite key each tile shard's WebDataset sample keys are built from -- the only
# column set that's both experiment-unique and present, under identical names, in both
# EMBED_CELLS' embeddings.parquet and QC_FILTER's filtered_cells.parquet. meta_cell_index
# alone repeats across tiles (it's a local per-tile cell-table index, not
# experiment-unique), so the full composite key is required.
JOIN_KEYS = ["meta_batch", "meta_well", "meta_tile", "meta_cell_index"]


@dataclasses.dataclass
class FilterEmbeddingsConfig(FilterParams):
    """
    Hydra structured configuration for FILTER_EMBEDDINGS.

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet. Required.
    qc_passed_file : str
        Path to QC_FILTER's filtered_cells.parquet. Required.
    label_column : str
        Name of the variant label column used to classify control (synonymous,
        untagged) rows. Defaults to ``"meta_aa_changes"``.
    """

    embeddings_file: str = MISSING
    qc_passed_file: str = MISSING


def filter_and_fit_normalizer(
    embeddings_lf: pl.LazyFrame, qc_passed_lf: pl.LazyFrame, label_column: str
) -> "tuple[pl.LazyFrame, Normalizer]":
    """:func:`fisseq_common.stages.filter.filter_and_fit_normalizer` with this pipeline's
    join keys and synonymous controls."""
    return _filter.filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, label_column, JOIN_KEYS, SYNONYMOUS_CONTROL
    )


def load_filtered_embeddings(
    embeddings_lf: pl.LazyFrame,
    filtered_keys_lf: pl.LazyFrame,
    normalizer: Normalizer,
) -> pl.LazyFrame:
    """
    Reconstruct the QC-passed, synonymous-corrected embedding (or CellProfiler feature)
    table on demand: :func:`fisseq_common.stages.filter.load_filtered_cells` on
    :data:`JOIN_KEYS`.
    """
    return _filter.load_filtered_cells(
        embeddings_lf, filtered_keys_lf, normalizer, JOIN_KEYS
    )


_cs = ConfigStore.instance()
_cs.store(name="filter_main", node=FilterEmbeddingsConfig)


@hydra.main(version_base=None, config_path=None, config_name="filter_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: determine QC-passed cells and fit the synonymous z-score.

    Writes ``{prefix}filtered_keys.parquet`` (no ``emb_*`` columns) and
    ``{prefix}normalizer.parquet`` to ``output_dir``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.filter \\
            output_dir=./out \\
            embeddings_file=embeddings.parquet \\
            qc_passed_file=filtered_cells.parquet \\
            random_seed=0
    """
    filter_cfg: FilterEmbeddingsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(filter_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filter_cfg.output_dir = str(output_dir)
    setup_logging(filter_cfg, "filter")

    logging.info("Reading embeddings from %s", filter_cfg.embeddings_file)
    embeddings_lf = pl.scan_parquet(filter_cfg.embeddings_file)
    logging.info("Reading QC-passed cells from %s", filter_cfg.qc_passed_file)
    qc_passed_lf = pl.scan_parquet(filter_cfg.qc_passed_file)

    logging.info("Determining QC-passed keys and fitting normalizer")
    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        embeddings_lf, qc_passed_lf, filter_cfg.label_column
    )
    write_filter_outputs(filtered_keys_lf, normalizer, filter_cfg)


if __name__ == "__main__":
    main()
