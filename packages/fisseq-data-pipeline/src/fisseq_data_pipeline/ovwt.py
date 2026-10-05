"""OVWT_BATCHWISE: cross-validated one-vs-wildtype variant scoring.

Hydra entry point (``python -m fisseq_data_pipeline.ovwt``). The scoring is shared with
fisseq-embeddings-pipeline and lives in :mod:`fisseq_common.stages.ovwt`; this module reads
the normalized cells (:mod:`.cells`) and scores its CellProfiler features (``FEATURE_SELECTOR``). K-fold
cross-validation stratified jointly on ``(meta_barcode, is_wt)`` gives every cell an
out-of-fold score and every variant several distinguishability numbers:

- ``auroc_pooled`` -- over all of the variant's cells at once.
- ``auroc_median_barcode`` -- each of the variant's barcodes scored separately
  against the full wildtype set, then medianed. This surfaces whether a
  variant's apparent distinguishability is broad-based across its barcodes or
  driven by one or two outliers, which a single pooled number hides.
- ``auroc_folds`` / ``auroc_median_fold`` -- each fold's test slice scored
  separately by that fold's own model (a list, one entry per fold, ``null``
  where a fold's test slice holds a single class), and the median of the
  defined entries. This is the older way of reporting distinguishability. The
  pooled and per-barcode numbers mix scores from different fold models into
  one ROC curve, and those scores do not share a scale, which can inflate
  them. A per-fold AUROC only ever ranks scores from one model against each
  other.

**One deliberate divergence from the reference implementation.** There, the
features fed to the classifier are z-scored against the experiment's own
*synonymous* variants. Here they are ``NORMALIZE``'s output, z-scored against
*wildtype* cells -- this pipeline's cell-level normalization is unchanged. The
synonymous re-centering happens outside this pipeline, on the AUROCs rather than
on the features, in the downstream ``fisseqborn`` package. Do not "fix" this
by adding a second normalizer fit here.

**Two cross-validation schemes**, selected by ``OvwtConfig.cv_mode``:

- ``"kfold"`` (the default) -- ``cfg.n_folds`` folds, stratified jointly on
  ``(meta_barcode, is_wt)``. Every fold's model has seen every barcode, so
  ``auroc_median_barcode`` measures separability *within* the barcodes the
  classifier was trained on.
- ``"barcode_holdout"`` -- each fold holds a whole barcode (or, when
  ``cfg.n_folds`` groups them, a whole set of barcodes) out of training, so no
  model is ever trained on a barcode it later scores. ``cfg.n_folds = None``
  gives one fold per barcode; an integer packs the barcodes into that many
  cell-count-balanced groups; a value above the barcode count degrades back to
  one fold per barcode. Wildtype cells are still split across the folds, so
  every cell still gets exactly one out-of-fold score and both AUROC columns
  keep the same meaning they have under k-fold. What changes is what they
  measure: whether a variant's signal *generalizes to a barcode the model has
  never seen*, rather than whether it is separable at all. A barcode-specific
  technical artifact inflates the k-fold number invisibly and is penalized
  here.

The feature-filtered and barcode-filtered variants of this stage are gone along
with ANOVA_BLOCKLIST and BARCODE_BLOCKLIST, as is the separate
``ovwtcellscores`` pass -- ``cell_scores.parquet`` below is emitted directly.
"""

import dataclasses
import logging
import pathlib

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.schema import FEATURE_SELECTOR
from fisseq_common.stages.ovwt import (  # noqa: F401 (CV_MODES: test_nextflow_params.py)
    CV_MODE_BARCODE_HOLDOUT,
    CV_MODE_KFOLD,
    CV_MODES,
    OvwtParams,
    run_ovwt,
)
from fisseq_common.utils.log import setup_logging

from .cells import CellsInput, load_cells


@dataclasses.dataclass
class OvwtConfig(CellsInput, OvwtParams):
    """
    Hydra structured configuration for OVWT_BATCHWISE.

    The scoring settings (``label_column``, ``wt_label``, ``cv_mode``, ``n_folds``,
    ``calibrate``, ``min_cells``, ``downsample_wt``, ``xgboost``) are
    :class:`~fisseq_common.stages.ovwt.OvwtParams`'.

    The normalized cells come from ``cells_file`` + ``filtered_keys_file`` +
    ``normalizer_file`` (QC_FILTER's and NORMALIZE's outputs), or from the deprecated
    ``input_file``; see :class:`~fisseq_data_pipeline.cells.CellsInput`.
    """


_cs = ConfigStore.instance()
_cs.store(name="ovwt_main", node=OvwtConfig)


@hydra.main(version_base=None, config_path=None, config_name="ovwt_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: cross-validated one-vs-wildtype scoring for every
    variant in an experiment.

    Writes ``{prefix}results.parquet``, ``{prefix}cell_scores.parquet`` and
    ``{prefix}models.pkl`` to ``output_dir`` (see
    :func:`fisseq_common.stages.ovwt.run_ovwt`).

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_data_pipeline.ovwt \\
            output_dir=./out \\
            input_file=out/normalized.parquet \\
            cv_mode=kfold \\
            n_folds=5 \\
            calibrate=true \\
            min_cells=250 \\
            downsample_wt=true \\
            random_seed=0
    """
    ovwt_cfg: OvwtConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(ovwt_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ovwt_cfg.output_dir = str(output_dir)
    setup_logging(ovwt_cfg, "ovwt")

    logging.info("Loading normalized cells from %s", ovwt_cfg.input_file)
    cells_lf = load_cells(ovwt_cfg)[0]
    run_ovwt(cells_lf, ovwt_cfg, FEATURE_SELECTOR)


if __name__ == "__main__":
    main()
