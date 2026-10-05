"""Edit-distance and barcode-count QC filtering for raw FISSEQ cell data.

Hydra entry point (``python -m fisseq_data_pipeline.qcfilter``) / Nextflow process ``QC_FILTER`` (first
pipeline stage). The filtering is shared with fisseq-embeddings-pipeline and lives in
:mod:`fisseq_common.stages.qcfilter`. Reads one or more raw CSV/Parquet cell files, renames columns to
canonical ``meta_*`` names, and applies sequential edit-distance, barcode-count,
and variant-barcode-count filters. Writes ``filtered_cells.parquet``,
``barcode_counts.parquet``, and ``variants_per_barcode.parquet``.

If ``n_variants`` is set, variants whose classified label is in
``variant_downsample_classes`` (default ``["Single Missense"]``) are
restricted to at most ``n_variants`` distinct variants before QC
thresholding runs — either the highest-cell-count variants (``"top"``
mode) or a seeded random sample (``"random"`` mode); see
:func:`select_variants`. Disabled by default.

If ``downsample_amounts`` is set (a single float/int, or a list of them),
``filtered_cells.parquet`` is additionally augmented with reproducibly-
downsampled "pseudo variant" rows for QC/calibration purposes, built from
cells whose classified label is in ``downsample_classes`` (default
``["Synonymous", "Single Missense"]``) that already survived the three QC
filters above (not the raw pre-QC population — this is what makes the
pseudo-variants a valid calibration of the post-QC analysis population).
Each amount produces its own distinctly-tagged group (``:downsample-{amount}``).
``barcode_counts.parquet`` and ``variants_per_barcode.parquet`` are computed
before this step runs, so they never include pseudo-variant rows. Disabled
by default; see :func:`add_downsampled_pseudo_variants`.
"""

import dataclasses
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.schema import META_CELL_INDEX_COL, META_VARIANT_TAG_COL
from fisseq_common.stages.qcfilter import (  # noqa: F401 (re-exported)
    DOWNSAMPLE_CLASSES,
    DOWNSAMPLE_TAG,
    VARIANT_DOWNSAMPLE_CLASSES,
    VARIANT_DOWNSAMPLE_MODES,
    QcFilterParams,
    run_qc_filter,
)
from fisseq_common.utils.log import setup_logging


@dataclasses.dataclass
class QcFilterConfig(QcFilterParams):
    """
    Hydra structured configuration for QC_FILTER.

    Every field is :class:`~fisseq_common.stages.qcfilter.QcFilterParams`'. The column-name
    defaults are the raw starcall cell table's: ``upBarcode``, ``aaChanges``,
    ``editDistance``.
    """

    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"


_cs = ConfigStore.instance()
_cs.store(name="qc_filter_main", node=QcFilterConfig)


@hydra.main(version_base=None, config_path=None, config_name="qc_filter_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: QC filtering of raw cell-level data files.

    Writes ``{prefix}filtered_cells.parquet``, ``{prefix}barcode_counts.parquet`` and
    ``{prefix}variants_per_barcode.parquet`` to ``output_dir`` (see
    :func:`fisseq_common.stages.qcfilter.run_qc_filter`). Every cell gets a
    ``meta_cell_index`` from the raw input row order, and ``filtered_cells`` is sorted on
    ``(meta_cell_index, meta_variant_tag)`` so its row order is reproducible.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.qcfilter \\
            output_dir=./out \\
            'cell_files=[data/batch1.csv]' \\
            bc_threshold=10 \\
            random_seed=0
    """
    qc_cfg: QcFilterConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(qc_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_cfg.output_dir = str(output_dir)
    setup_logging(qc_cfg, "qc_filter")

    run_qc_filter(qc_cfg, sort_output_by=[META_CELL_INDEX_COL, META_VARIANT_TAG_COL])


if __name__ == "__main__":
    main()
