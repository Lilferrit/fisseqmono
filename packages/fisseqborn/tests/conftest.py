import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from fisseqborn import fisseq


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def profiles() -> pl.DataFrame:
    """Synthetic per-variant profiles shaped like the fisseq notebooks' data."""
    rng = np.random.default_rng(42)
    n = 300
    variant_type = rng.choice(["Synonymous", "Single Missense", "Frameshift"], size=n, p=[0.3, 0.6, 0.1])
    shift = np.select(
        [variant_type == "Synonymous", variant_type == "Frameshift"], [0.0, 0.3], default=0.1
    )
    score = np.clip(0.55 + shift + rng.normal(0, 0.08, n), 0, 1)
    clinvar = np.where(
        (variant_type == "Single Missense") & (rng.random(n) < 0.15), fisseq.PATHOGENIC, variant_type
    )
    cluster = rng.integers(0, 12, n).astype(str)
    return pl.DataFrame(
        {
            "meta_aa_changes": [f"A{i}V" for i in range(n)],
            "meta_variant_type": variant_type,
            "meta_clinvar_annotation": clinvar,
            "meta_cluster_idx": cluster,
            "meta_experiment": rng.choice(["T1_R1", "T1_R2", "T2_R1", "T10_R1"], size=n),
            "meta_notebook_umap_1": rng.normal(0, 3, n) + shift * 10,
            "meta_notebook_umap_2": rng.normal(0, 3, n),
            "meta_distinguishability_score": score,
            "meta_distinguishability_score_rep2": np.clip(score + rng.normal(0, 0.05, n), 0, 1),
            "meta_impact_score": score * 0.8 + rng.normal(0, 0.05, n),
            "meta_num_cells": rng.integers(20, 500, n),
            **{f"feature_{i}": rng.normal(shift * i, 1, n) for i in range(8)},
        }
    )


PIPELINE_VARIANTS = ["A1A", "C2C", "G6G", "D3V", "E4K", "F5fs", "H7*"]
PIPELINE_FEATURES = ["AreaShape_Area", "Mean_Nuclei_Intensity_MeanIntensity_CH1", "Constant"]
# Written out of order to check that batches come back in natural order.
PIPELINE_BATCHES = ["T2_R1", "T10_R1", "T1_R1"]


@pytest.fixture
def pipeline_dir(tmp_path):
    """A tiny pipeline output directory with the layout of the real runs."""
    rng = np.random.default_rng(0)
    root = tmp_path / "run"
    n = len(PIPELINE_VARIANTS)
    for b, batch in enumerate(PIPELINE_BATCHES):
        fs = root / "feature_select_batchwise" / batch
        tables = {
            ("aggregates", "median"): lambda f, b=b: (
                np.full(n, 3.0) if f == "Constant" else rng.normal(b, 1 + b, n)
            ),
            ("aggregates", "KS"): lambda f: rng.uniform(0, 1, n),
            ("passthrough_aggregates", "KSnegLogP"): lambda f: rng.uniform(0, 10, n),
        }
        for (folder, stat), values in tables.items():
            (fs / folder).mkdir(parents=True, exist_ok=True)
            pl.DataFrame(
                {"meta_aa_changes": PIPELINE_VARIANTS}
                | {f"{f}_{stat}": values(f) for f in PIPELINE_FEATURES}
            ).write_parquet(fs / folder / f"{stat}.parquet")
        (fs / "blocklists").mkdir()
        for stat in ("median", "KS"):
            names = [f"{f}_{stat}" for f in PIPELINE_FEATURES]
            median_r = [0.9, 0.8 if b else 0.6, 0.1]
            pl.DataFrame(
                {"feature": names, "median_r": median_r, "feature_ok": [r > 0.5 for r in median_r]}
            ).write_parquet(fs / "blocklists" / f"{stat}.parquet")

        ovwt = root / "ovwt_batchwise" / batch
        ovwt.mkdir(parents=True)
        auroc = np.clip(0.5 + 0.1 * b + rng.normal(0, 0.1, n), 0, 1)
        pl.DataFrame(
            {
                "meta_aa_changes": PIPELINE_VARIANTS,
                "auroc_pooled": auroc,
                "auroc_median_barcode": auroc - 0.01,
                "auroc_folds": [[a, a] for a in auroc],
                "auroc_median_fold": auroc,
                "meta_n_barcodes": rng.integers(1, 5, n),
                "meta_n_cells": rng.integers(10, 100, n),
            }
        ).write_parquet(ovwt / "results.parquet")
    return root
