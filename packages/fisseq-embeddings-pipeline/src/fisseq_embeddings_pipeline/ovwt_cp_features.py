"""OVWT_BATCHWISE_CP_FEATURES.

Thin Hydra entry point around the shared OvWT scoring
(:mod:`fisseq_common.stages.ovwt`) and
:func:`~fisseq_embeddings_pipeline.filter.load_filtered_embeddings`, scoring the
CellProfiler features (``FEATURE_SELECTOR``: every non-``meta_*`` column) instead of
OVWT_BATCHWISE's ``EMBEDDING_SELECTOR``.

OVWT hyperparameters (``wt_label``, ``cv_mode``, ``n_folds``, ``calibrate``,
``min_cells``, ``downsample_wt``, ``xgboost``) are about scoring
methodology, not feature type -- this stage's config shares
:class:`~fisseq_common.stages.ovwt.OvwtParams` with OVWT_BATCHWISE, and its Nextflow
module passes the same ``params.yaml`` OVWT values.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import FEATURE_SELECTOR
from fisseq_common.stages.ovwt import OvwtParams, run_ovwt
from fisseq_common.utils.log import setup_logging

from .filter import load_filtered_embeddings


@dataclasses.dataclass
class OvwtCpFeaturesConfig(OvwtParams):
    """
    Hydra structured configuration for OVWT_BATCHWISE_CP_FEATURES.

    The scoring settings are :class:`~fisseq_common.stages.ovwt.OvwtParams`'.

    Attributes
    ----------
    cp_features_file : str
        Path to BUILD_CP_FEATURES' cp_features.parquet. Required.
    filtered_keys_file : str
        Path to FILTER_CP_FEATURES' filtered_keys.parquet. Required.
    normalizer_file : str
        Path to FILTER_CP_FEATURES' normalizer.parquet. Required.
    """

    cp_features_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING


_cs = ConfigStore.instance()
_cs.store(name="ovwt_cp_features_main", node=OvwtCpFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="ovwt_cp_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: k-fold one-vs-wildtype scoring for every variant,
    for the CellProfiler-feature track.

    Reads ``cp_features_file``, ``filtered_keys_file``, and
    ``normalizer_file``, reconstructs the QC-passed, synonymous-corrected
    feature table via
    :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`,
    and scores it with :func:`fisseq_common.stages.ovwt.run_ovwt` using
    ``FEATURE_SELECTOR``, which writes ``{prefix}results.parquet``,
    ``{prefix}cell_scores.parquet`` and ``{prefix}models.pkl`` to ``output_dir``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.ovwt_cp_features \\
            output_dir=./out \\
            cp_features_file=cp_features.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            cv_mode=kfold \\
            n_folds=5 \\
            calibrate=true \\
            min_cells=250 \\
            downsample_wt=true
    """
    ovwt_cfg: OvwtCpFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwt_cfg.output_dir = str(output_dir)
    setup_logging(ovwt_cfg, "ovwt_cp_features")

    logging.info("Reading CellProfiler features from %s", ovwt_cfg.cp_features_file)
    cp_features_lf = pl.scan_parquet(ovwt_cfg.cp_features_file)
    logging.info("Reading filtered keys from %s", ovwt_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(ovwt_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", ovwt_cfg.normalizer_file)
    normalizer = Normalizer.load(ovwt_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected features")
    filtered_lf = load_filtered_embeddings(cp_features_lf, filtered_keys_lf, normalizer)

    run_ovwt(filtered_lf, ovwt_cfg, FEATURE_SELECTOR)


if __name__ == "__main__":
    main()
