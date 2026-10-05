"""OVWT_BATCHWISE: one-vs-wildtype scoring of every variant on its Cell-DINO embeddings.

Hydra entry point (``python -m fisseq_embeddings_pipeline.ovwt``). The scoring itself is shared
with fisseq-data-pipeline and documented in :mod:`fisseq_common.stages.ovwt`; this module
reconstructs the QC-passed, synonymous-corrected embeddings from EMBED_CELLS' and
FILTER_EMBEDDINGS' outputs (:func:`~fisseq_embeddings_pipeline.filter.load_filtered_embeddings`)
and scores the ``emb_*`` columns (``EMBEDDING_SELECTOR``).
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import EMBEDDING_SELECTOR
from fisseq_common.stages.ovwt import (  # noqa: F401 (CV_MODES: config/experiments.py)
    CV_MODE_BARCODE_HOLDOUT,
    CV_MODE_KFOLD,
    CV_MODES,
    OvwtParams,
    run_ovwt,
)
from fisseq_common.utils.log import setup_logging

from .filter import load_filtered_embeddings


@dataclasses.dataclass
class OvwtEmbeddingConfig(OvwtParams):
    """
    Hydra structured configuration for OVWT_BATCHWISE.

    The scoring settings (``label_column``, ``wt_label``, ``cv_mode``, ``n_folds``,
    ``calibrate``, ``min_cells``, ``downsample_wt``, ``xgboost``) are
    :class:`~fisseq_common.stages.ovwt.OvwtParams`'.

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet. Required.
    filtered_keys_file : str
        Path to FILTER_EMBEDDINGS' filtered_keys.parquet. Required.
    normalizer_file : str
        Path to FILTER_EMBEDDINGS' normalizer.parquet. Required.
    """

    embeddings_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING


_cs = ConfigStore.instance()
_cs.store(name="ovwt_main", node=OvwtEmbeddingConfig)


@hydra.main(version_base=None, config_path=None, config_name="ovwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: k-fold one-vs-wildtype scoring for every variant in an experiment.

    Writes ``{prefix}results.parquet``, ``{prefix}cell_scores.parquet`` and
    ``{prefix}models.pkl`` to ``output_dir`` (see
    :func:`fisseq_common.stages.ovwt.run_ovwt`).

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.ovwt \\
            output_dir=./out \\
            embeddings_file=embeddings.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            cv_mode=kfold \\
            n_folds=5 \\
            calibrate=true \\
            min_cells=250 \\
            downsample_wt=true
    """
    ovwt_cfg: OvwtEmbeddingConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwt_cfg.output_dir = str(output_dir)
    setup_logging(ovwt_cfg, "ovwt")

    logging.info("Reading embeddings from %s", ovwt_cfg.embeddings_file)
    embeddings_lf = pl.scan_parquet(ovwt_cfg.embeddings_file)
    logging.info("Reading filtered keys from %s", ovwt_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(ovwt_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", ovwt_cfg.normalizer_file)
    normalizer = Normalizer.load(ovwt_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected embeddings")
    filtered_lf = load_filtered_embeddings(embeddings_lf, filtered_keys_lf, normalizer)

    run_ovwt(filtered_lf, ovwt_cfg, EMBEDDING_SELECTOR)


if __name__ == "__main__":
    main()
