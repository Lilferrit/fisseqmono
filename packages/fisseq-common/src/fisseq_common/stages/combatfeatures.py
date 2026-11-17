"""COMBAT_FEATURES: every experiment's per-variant aggregates, stacked and ComBat-corrected
(:mod:`fisseq_common.combat`), one row per (variant, experiment).

The inputs are the aggregate stage's outputs (``aggregates/<method>.parquet``: ``[label] +
<feature>_<method>``, every feature, before any blocklist), one file per method per
experiment. Each experiment's methods are joined on the label (inner) into one wide table;
every non-``meta_`` column is corrected in place; the output adds ``batch_column``.

Entry point: ``python -m fisseq_common.stages.combatfeatures``.
"""

import dataclasses
import logging
import pathlib

import polars as pl

from fisseq_common.combat import combat_frame, stack_batches
from fisseq_common.schema import FEATURE_SELECTOR

from .config import CombatInput, read_batches, stage_main


def _join_methods(frames: list[pl.DataFrame], label_column: str) -> pl.DataFrame:
    """One experiment's per-method aggregates side by side, on the variants they share."""
    joined = frames[0]
    for frame in frames[1:]:
        clash = (set(joined.columns) & set(frame.columns)) - {label_column}
        if clash:
            raise ValueError(
                f"Two aggregate files share column(s) {sorted(clash)[:5]}; pass each "
                "method once"
            )
        joined = joined.join(frame, on=label_column, how="inner")
    return joined


@dataclasses.dataclass
class CombatFeaturesParams(CombatInput):
    """Hydra structured configuration for COMBAT_FEATURES (see :class:`CombatInput`)."""


def run_combat_features(cfg: CombatFeaturesParams) -> None:
    """Run the stage with ``cfg`` (``output_dir`` must exist)."""
    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""
    batches = {
        name: _join_methods(frames, cfg.label_column)
        for name, frames in read_batches(cfg).items()
    }
    stacked = stack_batches(batches, batch_column=cfg.batch_column)
    features = stacked.select(FEATURE_SELECTOR).columns
    logging.info(
        "Correcting %d feature(s) over %d row(s) from %d experiment(s)",
        len(features),
        stacked.height,
        len(batches),
    )
    corrected = combat_frame(
        stacked,
        features,
        label_column=cfg.label_column,
        batch_column=cfg.batch_column,
        mean_only=cfg.mean_only,
    )
    out_path = pathlib.Path(cfg.output_dir) / f"{prefix}features.parquet"
    logging.info("Writing %s (%d rows)", out_path, corrected.height)
    corrected.write_parquet(out_path)


main = stage_main("combat_features_main", CombatFeaturesParams, run_combat_features)

if __name__ == "__main__":
    main()
