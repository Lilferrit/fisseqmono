"""QC_FILTER: edit-distance and barcode-count QC filtering of BUILD_CELL_METADATA's cells.

Hydra entry point (``python -m fisseq_embeddings_pipeline.qcfilter``). The filtering is shared
with fisseq-data-pipeline and documented in :mod:`fisseq_common.stages.qcfilter`. Here
``cell_files`` is BUILD_CELL_METADATA's ``metadata.parquet``, whose barcode, amino-acid-change
and edit-distance columns already carry their canonical ``meta_*`` names, so those are the
column-name defaults.
"""

import dataclasses
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.stages.qcfilter import (  # noqa: F401 (re-exported)
    VARIANT_DOWNSAMPLE_CLASSES,
    VARIANT_DOWNSAMPLE_MODES,
    QcFilterParams,
    run_qc_filter,
)
from fisseq_common.utils.log import setup_logging

from .filter import JOIN_KEYS


@dataclasses.dataclass
class QcFilterConfig(QcFilterParams):
    """
    Hydra structured configuration for QC_FILTER.

    Every field is :class:`~fisseq_common.stages.qcfilter.QcFilterParams`'. The column-name
    defaults (``meta_barcode``, ``meta_aa_changes``, ``meta_edit_distance``) are
    ``metadata.parquet``'s own column names. ``downsample_amounts`` (pseudo variants) is left
    off: per-dimension calibration doesn't translate to embedding dimensions.
    """

    barcode_col_name: str = "meta_barcode"
    aa_changes_col_name: str = "meta_aa_changes"
    edit_distance_col_name: str = "meta_edit_distance"


_cs = ConfigStore.instance()
_cs.store(name="qc_filter_main", node=QcFilterConfig)


@hydra.main(version_base=None, config_path=None, config_name="qc_filter_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: QC filtering of cell-level data files.

    Writes ``{prefix}filtered_cells.parquet``, ``{prefix}barcode_counts.parquet`` and
    ``{prefix}variants_per_barcode.parquet`` to ``output_dir`` (see
    :func:`fisseq_common.stages.qcfilter.run_qc_filter`).

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.qcfilter \\
            output_dir=./out \\
            'cell_files=[data/metadata.parquet]' \\
            bc_threshold=10 \\
            random_seed=0
    """
    qc_cfg: QcFilterConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(qc_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_cfg.output_dir = str(output_dir)
    setup_logging(qc_cfg, "qc_filter")

    # A stable row order for every seeded step downstream (OvWT folds, bootstrap splits):
    # sort on the cell keys, as the data pipeline does on its own.
    run_qc_filter(qc_cfg, sort_output_by=JOIN_KEYS)


if __name__ == "__main__":
    main()
