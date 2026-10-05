"""GENERATE_SPLIT: one stratified 50/50 pseudo-replicate split per bootstrap replicate.

Hydra entry point (``python -m fisseq_data_pipeline.generatesplit``) / feature-selection stage
2a. The stage is shared with fisseq-embeddings-pipeline and documented in
:mod:`fisseq_common.stages.generatesplit`: it reads NORMALIZE's ``filtered_keys.parquet`` and
writes each half as the cells' keys, ``(meta_cell_index, meta_variant_tag)``, which
AGGREGATE_HALF applies with a semi-join.

The split is seeded with ``random_seed + bootstrap_idx``. ``bootstrap_idx`` defaults to 0 here,
so an invocation that already folds the replicate into ``random_seed`` keeps its seed.

Deprecated: ``input_file`` (a normalized cell table) instead of ``filtered_keys_file`` writes
the old positional split files (a ``tmp_cell_idx`` column of row indices).
"""

import dataclasses
import logging
import pathlib
import warnings
from typing import Optional

import hydra
import polars as pl
import sklearn.model_selection
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.stages.generatesplit import GenerateSplitParams, run_generate_split
from fisseq_common.utils.batches import load_batches
from fisseq_common.utils.log import setup_logging
from fisseq_common.utils.metadata import get_column

from .cells import JOIN_KEYS
from .utils.splits import TMP_IDX_COL, add_row_index


@dataclasses.dataclass
class GenerateSplitConfig(GenerateSplitParams):
    """
    Hydra structured configuration for GENERATE_SPLIT.

    The fields are :class:`~fisseq_common.stages.generatesplit.GenerateSplitParams`', with
    ``bootstrap_idx`` defaulting to ``0``.

    Attributes
    ----------
    filtered_keys_file : str, optional
        NORMALIZE's ``filtered_keys.parquet``.
    input_file : str, optional
        Deprecated: a normalized cell table (or glob); writes positional split files.
    """

    filtered_keys_file: Optional[str] = None
    bootstrap_idx: int = 0
    input_file: Optional[str] = None


def _legacy_positional_split(cfg: GenerateSplitConfig) -> None:
    warnings.warn(
        "generatesplit's input_file is deprecated; pass filtered_keys_file",
        DeprecationWarning,
        stacklevel=2,
    )
    logging.warning("input_file is deprecated; pass filtered_keys_file instead")
    output_dir = pathlib.Path(cfg.output_dir)
    lf, _ = load_batches(cfg.input_file)
    lf = add_row_index(lf)
    idx = get_column(lf, TMP_IDX_COL)
    labels = get_column(lf, cfg.label_column)
    half1_idx, half2_idx, _, _ = sklearn.model_selection.train_test_split(
        idx,
        labels,
        stratify=labels,
        random_state=cfg.random_seed + cfg.bootstrap_idx,
        test_size=0.5,
    )
    pl.DataFrame({TMP_IDX_COL: half1_idx}).write_parquet(output_dir / "half1.parquet")
    pl.DataFrame({TMP_IDX_COL: half2_idx}).write_parquet(output_dir / "half2.parquet")


_cs = ConfigStore.instance()
_cs.store(name="generate_split_main", node=GenerateSplitConfig)


@hydra.main(version_base=None, config_path=None, config_name="generate_split_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: write ``half1.parquet`` and ``half2.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.generatesplit \\
            output_dir=./out \\
            filtered_keys_file=normalization/batch1/filtered_keys.parquet \\
            bootstrap_idx=3 \\
            random_seed=0
    """
    split_cfg: GenerateSplitConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(split_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_cfg.output_dir = str(output_dir)
    setup_logging(split_cfg, "generate_split")

    if split_cfg.filtered_keys_file is not None:
        run_generate_split(split_cfg, JOIN_KEYS)
    elif split_cfg.input_file is not None:
        _legacy_positional_split(split_cfg)
    else:
        raise ValueError("Set filtered_keys_file (or the deprecated input_file)")


if __name__ == "__main__":
    main()
