"""BLOCKLIST: the median-across-replicates reproducibility verdict for one method.

Hydra entry point (``python -m fisseq_embeddings_pipeline.blocklist``). The stage is shared with fisseq-data-pipeline and documented
in :mod:`fisseq_common.stages.blocklist`.
"""

import dataclasses
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.stages.blocklist import (  # noqa: F401 (re-exported)
    BlocklistParams,
    compute_blocklist,
    run_blocklist,
)
from fisseq_common.utils.log import setup_logging


@dataclasses.dataclass
class BlocklistConfig(BlocklistParams):
    """Hydra structured configuration for BLOCKLIST; every field is
    :class:`~fisseq_common.stages.blocklist.BlocklistParams`'."""


_cs = ConfigStore.instance()
_cs.store(name="blocklist_main", node=BlocklistConfig)


@hydra.main(version_base=None, config_path=None, config_name="blocklist_main")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: see :func:`fisseq_common.stages.blocklist.run_blocklist`.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.blocklist \\
            output_dir=./out \\
            'correlation_files=./correlations/*.parquet' \\
            minimum_correlation=0.5
    """
    stage_cfg: BlocklistConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(stage_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg.output_dir = str(output_dir)
    setup_logging(stage_cfg, "blocklist")

    run_blocklist(stage_cfg)


if __name__ == "__main__":
    main()
