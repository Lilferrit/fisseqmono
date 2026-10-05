"""GENERATE_SPLIT: one bootstrap pseudo-replicate 50/50 cell split.

Stage 2a of the reproducibility-filtering chain (GENERATE_SPLIT -> AGGREGATE_HALF ->
CORRELATE_FEATURES -> BLOCKLIST -> COMBINE_BLOCKLISTS -> FILTER_AGGREGATE). Hydra entry point
(``python -m fisseq_embeddings_pipeline.generatesplit``); the stage is shared with
fisseq-data-pipeline and documented in :mod:`fisseq_common.stages.generatesplit`.

"Replicate" here is a *pseudo*-replicate: there is only one cell population per experiment, so
reproducibility is measured by splitting it in half, aggregating each half independently, and
correlating the two per-variant aggregates dimension by dimension. Reads FILTER_EMBEDDINGS'
``filtered_keys.parquet`` only, and writes each half as :data:`~.filter.JOIN_KEYS` rows.
"""

import dataclasses
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.stages.generatesplit import (  # noqa: F401 (re-exported)
    GenerateSplitParams,
    run_generate_split,
    split_keys,
)
from fisseq_common.utils.log import setup_logging

from .filter import JOIN_KEYS


@dataclasses.dataclass
class GenerateSplitConfig(GenerateSplitParams):
    """Hydra structured configuration for GENERATE_SPLIT; every field is
    :class:`~fisseq_common.stages.generatesplit.GenerateSplitParams`'."""


_cs = ConfigStore.instance()
_cs.store(name="generate_split_main", node=GenerateSplitConfig)


@hydra.main(version_base=None, config_path=None, config_name="generate_split_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: write ``{prefix}half1.parquet`` and ``{prefix}half2.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.generatesplit \\
            output_dir=./out \\
            filtered_keys_file=filtered_keys.parquet \\
            bootstrap_idx=3 \\
            random_seed=0
    """
    split_cfg: GenerateSplitConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(split_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_cfg.output_dir = str(output_dir)
    setup_logging(split_cfg, "generate_split")

    run_generate_split(split_cfg, JOIN_KEYS)


if __name__ == "__main__":
    main()
