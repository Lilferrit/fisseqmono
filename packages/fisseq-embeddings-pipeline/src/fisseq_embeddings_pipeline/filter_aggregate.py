"""FILTER_AGGREGATE: drop the non-reproducible columns, then attach the passthroughs.

Hydra entry point (``python -m fisseq_embeddings_pipeline.filter_aggregate``). Its pycytominer
variance/correlation selection is not run here: those filters are written for named
morphological features, so the reproducibility verdict is the only selection. The stage is shared with fisseq-data-pipeline's FINALIZE_FEATURE_SELECT and documented
in :mod:`fisseq_common.stages.filter_aggregate`.
"""

import dataclasses
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.stages.filter_aggregate import (  # noqa: F401 (re-exported)
    FilterAggregateParams,
    apply_blocklist,
    blocked_features,
    join_passthrough,
    run_filter_aggregate,
)
from fisseq_common.utils.log import setup_logging


@dataclasses.dataclass
class FilterAggregateConfig(FilterAggregateParams):
    """Hydra structured configuration for FILTER_AGGREGATE; every field is
    :class:`~fisseq_common.stages.filter_aggregate.FilterAggregateParams`'."""


_cs = ConfigStore.instance()
_cs.store(name="filter_aggregate_main", node=FilterAggregateConfig)


@hydra.main(version_base=None, config_path=None, config_name="filter_aggregate_main")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: see :func:`fisseq_common.stages.filter_aggregate.run_filter_aggregate`.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.filter_aggregate \\
            output_dir=./out \\
            aggregate_file=aggregate.parquet \\
            blocklist_file=blocklist.parquet \\
            'passthrough_files=./passthrough_aggregates/*.parquet'
    """
    stage_cfg: FilterAggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(stage_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_cfg.output_dir = str(output_dir)
    setup_logging(stage_cfg, "filter_aggregate")

    run_filter_aggregate(stage_cfg)


if __name__ == "__main__":
    main()
