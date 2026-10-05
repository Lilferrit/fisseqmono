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

:func:`run_old_global_stages` ran the embeddings pipeline's GLOBAL_* stages on them (the stages
fisseqborn replaces), the way ``workflows/embeddings.nf`` invoked them. Their outputs are saved
under ``tests/reference/embeddings_global/global/``.
"""

from __future__ import annotations

import subprocess
import sys
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

# min_batches_ok values the old GLOBAL_BLOCKLIST + GLOBAL_VARIANT_EMBEDDINGS ran with.
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


def _hydra_list(key: str, values: list) -> str:
    return f"{key}=[{','.join(str(v) for v in values)}]"


def _run(module: str, output_dir: Path, *args: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            f"fisseq_embeddings_pipeline.{module}",
            f"output_dir={output_dir}",
            *args,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{module} failed\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        )


def run_old_global_stages(pipeline_dir: Path) -> None:
    """Run the embeddings pipeline's GLOBAL_* stages on :func:`write_inputs`' outputs."""
    out = pipeline_dir / "global"
    batches = list(BATCHES)  # workflows/embeddings.nf's sortedPairs order
    common = [f"label_column={LABEL}", "random_seed=0"]

    def files(track: str, name: str) -> list[str]:
        return [str(pipeline_dir / track / b / name) for b in batches]

    for name, min_ok in MIN_BATCHES_OK.items():
        _run(
            "global_blocklist",
            out / name,
            _hydra_list(
                "input_files", files("feature_select_batchwise", "blocklist.parquet")
            ),
            f"min_batches_ok={'null' if min_ok is None else min_ok}",
            "random_seed=0",
        )
        _run(
            "global_embeddings",
            out / name,
            _hydra_list(
                "input_files", files("feature_select_batchwise", "aggregate.parquet")
            ),
            _hydra_list("batch_stems", batches),
            f"blocklist_file={out / name / 'blocklist.parquet'}",
            "cumulative_variance_explained=0.9",
            *common,
        )
    _run(
        "global_distinguishability",
        out / "distinguishability",
        _hydra_list("input_files", files("ovwt_batchwise", "results.parquet")),
        _hydra_list("batch_stems", batches),
        *common,
    )
    _run(
        "global_variant_cp_features",
        out / "cp_features",
        _hydra_list(
            "input_files",
            files("feature_select_batchwise_cp_features", "aggregate.parquet"),
        ),
        _hydra_list("batch_stems", batches),
        "cumulative_variance_explained=0.9",
        *common,
    )
    _run(
        "global_variant_distinguishability_cp_features",
        out / "distinguishability_cp_features",
        _hydra_list(
            "input_files", files("ovwt_batchwise_cp_features", "results.parquet")
        ),
        _hydra_list("batch_stems", batches),
        *common,
    )
