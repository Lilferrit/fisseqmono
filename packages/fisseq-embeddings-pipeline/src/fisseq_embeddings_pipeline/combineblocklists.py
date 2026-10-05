"""COMBINE_BLOCKLISTS: one experiment's per-method blocklists in one table.

Hydra entry point (``python -m fisseq_embeddings_pipeline.combineblocklists``). The stage is shared with fisseq-data-pipeline and documented
in :mod:`fisseq_common.stages.combineblocklists`.
"""

import dataclasses
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.stages.combineblocklists import (  # noqa: F401 (re-exported)
    CombineBlocklistsParams,
    run_combine_blocklists,
)
from fisseq_common.utils.log import setup_logging


@dataclasses.dataclass
class CombineBlocklistsConfig(CombineBlocklistsParams):
    """Hydra structured configuration for COMBINE_BLOCKLISTS; every field is
    :class:`~fisseq_common.stages.combineblocklists.CombineBlocklistsParams`'."""


_cs = ConfigStore.instance()
_cs.store(name="combine_blocklists_main", node=CombineBlocklistsConfig)


@hydra.main(version_base=None, config_path=None, config_name="combine_blocklists_main")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: see :func:`fisseq_common.stages.combineblocklists.run_combine_blocklists`.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.combineblocklists \\
            output_dir=./out \\
            'blocklist_files=./blocklists/*.parquet'
    """
    stage_cfg: CombineBlocklistsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(stage_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg.output_dir = str(output_dir)
    setup_logging(stage_cfg, "combine_blocklists")

    run_combine_blocklists(stage_cfg)


if __name__ == "__main__":
    main()
