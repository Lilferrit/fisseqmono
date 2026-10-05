"""CORRELATE_FEATURES: per-feature correlation between two pseudo-replicate halves.

Hydra entry point (``python -m fisseq_embeddings_pipeline.correlatefeatures``). The stage is shared with fisseq-data-pipeline and documented
in :mod:`fisseq_common.stages.correlatefeatures`.
"""

import dataclasses
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.stages.correlatefeatures import (  # noqa: F401 (re-exported)
    CorrelateFeaturesParams,
    compute_feature_correlations,
    run_correlate_features,
)
from fisseq_common.utils.log import setup_logging


@dataclasses.dataclass
class CorrelateFeaturesConfig(CorrelateFeaturesParams):
    """Hydra structured configuration for CORRELATE_FEATURES; every field is
    :class:`~fisseq_common.stages.correlatefeatures.CorrelateFeaturesParams`'."""


_cs = ConfigStore.instance()
_cs.store(name="correlate_features_main", node=CorrelateFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="correlate_features_main")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: see :func:`fisseq_common.stages.correlatefeatures.run_correlate_features`.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.correlatefeatures \\
            output_dir=./out \\
            half1_file=half1.parquet \\
            half2_file=half2.parquet
    """
    stage_cfg: CorrelateFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(stage_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg.output_dir = str(output_dir)
    setup_logging(stage_cfg, "correlate_features")

    run_correlate_features(stage_cfg)


if __name__ == "__main__":
    main()
