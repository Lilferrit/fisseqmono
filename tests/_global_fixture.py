"""Synthetic per-experiment outputs for the cross-experiment (global) aggregation.

The pipeline fixtures are too small for the global steps to compute anything meaningful: every
OvWT AUROC is 0.5, so the per-experiment z-scores are undefined, and only two variants reach the
PCA. This module writes richer per-experiment outputs in the embeddings pipeline's layout, for
both tracks:

- ``feature_select_batchwise/<b>/{aggregate,blocklist}.parquet``
- ``ovwt_batchwise/<b>/results.parquet``
- ``feature_select_batchwise_cp_features/<b>/aggregate.parquet``
- ``ovwt_batchwise_cp_features/<b>/results.parquet``

Three experiments, 24 variants including five synonymous ones (one of them tagged). Not every
variant is in every experiment, one embedding column is missing from one experiment, one is
missing from the blocklists, and the blocklist votes disagree between experiments. Unlike the
pipeline's own aggregates, these keep the synonymous rows, so the impact score has controls.

Before they were deleted, the embeddings pipeline's GLOBAL_* stages (the stages fisseqborn
replaces) ran on these inputs the way ``workflows/embeddings.nf`` invoked them, in batch order,
with ``label_column=meta_aa_changes``, ``random_seed=0`` and ``cumulative_variance_explained=0.9``,
once per :data:`MIN_BATCHES_OK` value (GLOBAL_BLOCKLIST + GLOBAL_VARIANT_EMBEDDINGS) and once for
the other three stages. Their outputs are saved under
``tests/reference/embeddings_global/global/`` (commit efb32e9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

BATCHES = ("batch1", "batch2", "batch3")
DIMS = 6
METHODS = ("median", "KS", "AUROC")
CP_FEATURES = (
    "Cells_AreaShape_Area",
    "Cells_Intensity_MeanIntensity_CH1",
    "Nuclei_AreaShape_Area",
    "Nuclei_Texture_Contrast_CH2",
)
SYNONYMOUS = ("A1A", "C3C", "G7G", "L9L", "S12S")
MISSENSE = tuple(
    f"{aa}{pos}{alt}"
    for pos, (aa, alt) in enumerate(
        [
            ("M", "K"),
            ("D", "E"),
            ("R", "W"),
            ("P", "L"),
            ("V", "A"),
            ("T", "I"),
            ("E", "K"),
            ("K", "N"),
            ("F", "S"),
            ("Y", "C"),
            ("H", "R"),
            ("Q", "P"),
            ("N", "D"),
            ("I", "T"),
            ("W", "R"),
        ],
        start=20,
    )
)
OTHER = ("WT", "A1A:downsampled-half", "M40fs", "R41*")
VARIANTS = SYNONYMOUS + MISSENSE + OTHER

# min_batches_ok values the old GLOBAL_BLOCKLIST + GLOBAL_VARIANT_EMBEDDINGS ran with, and the
# reference subdirectory of each.
MIN_BATCHES_OK = {"embeddings": None, "embeddings_min2": 2}
LABEL = "meta_aa_changes"


def _emb_columns(batch_index: int) -> list[str]:
    cols = [f"emb_{d:04d}_{m}" for d in range(DIMS) for m in METHODS]
    if batch_index == 2:
        cols.remove("emb_0005_AUROC")  # missing from one experiment
    return cols


def _variants(rng: np.random.Generator, batch_index: int) -> list[str]:
    # Each experiment misses a few non-control variants.
    dropped = set(rng.choice(MISSENSE, size=2 + batch_index, replace=False))
    return [v for v in VARIANTS if v not in dropped]


def write_inputs(pipeline_dir: Path) -> None:
    for i, batch in enumerate(BATCHES):
        rng = np.random.default_rng(100 + i)
        variants = _variants(rng, i)
        n = len(variants)
        effect = np.array(
            [0.0 if v in SYNONYMOUS else rng.normal(0, 1.5) for v in variants]
        )

        fs = pipeline_dir / "feature_select_batchwise" / batch
        fs.mkdir(parents=True, exist_ok=True)
        cols = _emb_columns(i)
        pl.DataFrame(
            {
                LABEL: variants,
                "meta_n_cells": rng.integers(20, 200, n),
                **{
                    c: effect * rng.normal(1, 0.2) + rng.normal(0, 1, n) + 0.3 * i
                    for c in cols
                },
            }
        ).write_parquet(fs / "aggregate.parquet")
        voted = [c for c in cols if c != "emb_0004_KS"]  # one feature never voted on
        median_r = rng.uniform(-0.2, 1.0, len(voted))
        pl.DataFrame(
            {
                "feature": voted,
                "median_r": median_r,
                "feature_ok": median_r >= 0.3,
            }
        ).write_parquet(fs / "blocklist.parquet")

        fs_cp = pipeline_dir / "feature_select_batchwise_cp_features" / batch
        fs_cp.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                LABEL: variants,
                "meta_n_cells": rng.integers(20, 200, n),
                **{
                    c: effect * rng.normal(1, 0.2) + rng.normal(0, 1, n)
                    for c in CP_FEATURES
                },
            }
        ).write_parquet(fs_cp / "aggregate.parquet")

        for track in ("ovwt_batchwise", "ovwt_batchwise_cp_features"):
            scored = [v for v in variants if v != "WT"]
            m = len(scored)
            signal = np.array(
                [0.0 if v in SYNONYMOUS else abs(rng.normal(0, 0.15)) for v in scored]
            )
            pooled = np.clip(0.5 + signal + rng.normal(0, 0.03, m), 0, 1)
            folds = [list(np.clip(p + rng.normal(0, 0.05, 3), 0, 1)) for p in pooled]
            out = pipeline_dir / track / batch
            out.mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {
                    LABEL: scored,
                    "auroc_pooled": pooled,
                    "auroc_median_barcode": np.clip(
                        pooled + rng.normal(0, 0.02, m), 0, 1
                    ),
                    "auroc_folds": folds,
                    "auroc_median_fold": [float(np.median(f)) for f in folds],
                    "meta_n_barcodes": rng.integers(2, 10, m),
                    "meta_n_cells": rng.integers(20, 200, m),
                }
            ).write_parquet(out / "results.parquet")
