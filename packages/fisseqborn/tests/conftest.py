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
