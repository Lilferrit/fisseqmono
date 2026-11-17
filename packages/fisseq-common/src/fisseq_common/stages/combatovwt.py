"""COMBAT_OVWT: every experiment's OvWT ``results.parquet``, stacked and ComBat-corrected
(:mod:`fisseq_common.combat`), one row per (variant, experiment).

The score columns are corrected jointly, in place. The per-fold ``auroc_folds`` list is
dropped (it isn't corrected); the other columns (``meta_n_cells``, ``meta_n_barcodes``) are
kept, and the output adds ``batch_column``.

Entry point: ``python -m fisseq_common.stages.combatovwt``.
"""

import dataclasses
import logging
import pathlib
from typing import List

from fisseq_common.combat import combat_frame, stack_batches
from fisseq_common.global_aggregation import OVWT_SCORES

from .config import CombatInput, read_batches, stage_main


@dataclasses.dataclass
class CombatOvwtParams(CombatInput):
    """
    Hydra structured configuration for COMBAT_OVWT (see :class:`CombatInput`).

    Attributes
    ----------
    scores : list of str
        The score columns to correct; those missing from the results are skipped. Defaults
        to ``auroc_pooled``, ``auroc_median_barcode`` and ``auroc_median_fold``.
    """

    scores: List[str] = dataclasses.field(default_factory=lambda: list(OVWT_SCORES))


def run_combat_ovwt(cfg: CombatOvwtParams) -> None:
    """Run the stage with ``cfg`` (``output_dir`` must exist)."""
    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""
    batches = read_batches(cfg)
    several = [name for name, frames in batches.items() if len(frames) > 1]
    if several:
        raise ValueError(f"One results file per experiment; several for {several}")
    stacked = stack_batches(
        {name: frames[0] for name, frames in batches.items()},
        batch_column=cfg.batch_column,
    )
    stacked = stacked.drop("auroc_folds", strict=False)
    scores = [s for s in cfg.scores if s in stacked.columns]
    absent = [s for s in cfg.scores if s not in stacked.columns]
    if absent:
        logging.warning("Score column(s) not in the results: %s", ", ".join(absent))
    if not scores:
        raise ValueError(f"None of the score columns {list(cfg.scores)} is present")
    corrected = combat_frame(
        stacked,
        scores,
        label_column=cfg.label_column,
        batch_column=cfg.batch_column,
        mean_only=cfg.mean_only,
    )
    out_path = pathlib.Path(cfg.output_dir) / f"{prefix}ovwt_scores.parquet"
    logging.info("Writing %s (%d rows)", out_path, corrected.height)
    corrected.write_parquet(out_path)


main = stage_main("combat_ovwt_main", CombatOvwtParams, run_combat_ovwt)

if __name__ == "__main__":
    main()
