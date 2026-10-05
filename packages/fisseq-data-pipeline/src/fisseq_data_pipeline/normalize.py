"""NORMALIZE: the QC-passed cells' keys plus a Normalizer fitted on wildtype cells.

Hydra entry point (``python -m fisseq_data_pipeline.normalize``) / Nextflow process
``NORMALIZE`` (second pipeline stage). The stage is shared with fisseq-embeddings-pipeline and
documented in :mod:`fisseq_common.stages.filter`. It writes only

- ``filtered_keys.parquet``: every ``meta_*`` column of the QC-passed cells, plus
  ``meta_is_control`` and ``meta_batch``, and no features;
- ``normalizer.parquet``: per-feature means and standard deviations of the control cells.

No normalized copy of the cell table is written. Every stage that needs it rebuilds it from
QC_FILTER's ``filtered_cells.parquet`` and these two files (:mod:`.cells`).

The control cells are the rows matching ``control_sample_query``, by default wildtype. The
embeddings pipeline uses untagged synonymous variants instead; this pipeline keeps wildtype
because its per-variant aggregates are z-scored against the synonymous variants afterwards
(AGGREGATE_FEATURE_TYPE's ``normalize_to_synonymous``), which needs the synonymous cells to
survive aggregation as ordinary variants.
"""

import dataclasses
import logging
import pathlib
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.schema import META_BATCH_COL
from fisseq_common.stages.filter import (
    FilterParams,
    filter_and_fit_normalizer,
    write_filter_outputs,
)
from fisseq_common.utils.log import setup_logging

from .cells import JOIN_KEYS


@dataclasses.dataclass
class NormalizeConfig(FilterParams):
    """
    Hydra structured configuration for the normalization entry point.

    Attributes
    ----------
    input_file : str
        QC_FILTER's ``filtered_cells.parquet``. Required.
    control_sample_query : str
        SQL-like WHERE clause identifying the control rows the normalizer is fitted on.
        Defaults to ``"meta_aa_changes = 'WT'"``.
    batch_name : str, optional
        The experiment's name, stored as ``meta_batch`` in ``filtered_keys.parquet``.
        Defaults to ``input_file``'s stem.
    save_normalizer : bool
        Deprecated and ignored: ``normalizer.parquet`` is always written, since no
        normalized cell table is.
    """

    input_file: str = MISSING
    control_sample_query: str = "meta_aa_changes = 'WT'"
    batch_name: Optional[str] = None
    save_normalizer: bool = True


_cs = ConfigStore.instance()
_cs.store(name="normalize_main", node=NormalizeConfig)


@hydra.main(version_base=None, config_path=None, config_name="normalize_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: determine the QC-passed cells and fit the control z-score.

    Writes ``{prefix}filtered_keys.parquet`` and ``{prefix}normalizer.parquet`` to
    ``output_dir``, where ``prefix`` is ``{output_root}.`` when ``output_root`` is set.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.normalize \\
            output_dir=./out \\
            input_file=qc_filter/batch1/filtered_cells.parquet \\
            batch_name=batch1
    """
    norm_cfg: NormalizeConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(norm_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_cfg.output_dir = str(output_dir)
    setup_logging(norm_cfg, "normalize")

    if not norm_cfg.save_normalizer:
        logging.warning(
            "save_normalizer is deprecated and ignored: normalizer.parquet is always written"
        )

    input_path = pathlib.Path(norm_cfg.input_file)
    batch_name = norm_cfg.batch_name or input_path.stem
    logging.info("Loading QC-passed cells from %s (batch %s)", input_path, batch_name)
    cells_lf = pl.scan_parquet(input_path).with_columns(
        pl.lit(batch_name).alias(META_BATCH_COL)
    )

    logging.info(
        "Fitting normalizer on rows matching %r", norm_cfg.control_sample_query
    )
    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        cells_lf,
        cells_lf,
        norm_cfg.label_column,
        JOIN_KEYS,
        control=norm_cfg.control_sample_query,
        sort_by=JOIN_KEYS,
    )
    write_filter_outputs(filtered_keys_lf, normalizer, norm_cfg)


if __name__ == "__main__":
    main()
