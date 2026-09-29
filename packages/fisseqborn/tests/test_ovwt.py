import numpy as np
import polars as pl
import pytest
from conftest import PIPELINE_BATCHES, PIPELINE_VARIANTS

import fisseqborn as fb


@pytest.fixture
def batch_scores() -> fb.OvwtScores:
    # b1 is shifted up by 0.2 and b2 is twice as spread out as b0 (synonymous rows first)
    return fb.OvwtScores(
        pl.DataFrame(
            {
                "meta_aa_changes": ["A1A", "C2C", "D3V"] * 3,
                "meta_experiment": ["b0"] * 3 + ["b1"] * 3 + ["b2"] * 3,
                "auroc_pooled": [0.5, 0.6, 0.9, 0.7, 0.8, 1.0, 0.4, 0.6, 0.8],
                "meta_n_cells": [10] * 9,
            }
        )
    )


def test_from_pipeline(pipeline_dir):
    scores = fb.OvwtScores.from_pipeline(pipeline_dir)
    df = scores.df
    assert df.height == len(PIPELINE_VARIANTS) * len(PIPELINE_BATCHES)
    assert df["meta_experiment"].unique(maintain_order=True).to_list() == [
        "T1_R1",
        "T2_R1",
        "T10_R1",
    ]
    assert {"auroc_pooled", "auroc_median_barcode", "auroc_folds"} <= set(df.columns)


def test_correct_matches_hand_computation(batch_scores):
    df = batch_scores.correct().df
    # synonymous (A1A, C2C) mean/std per batch: b0 0.55/0.0707, b1 0.75/0.0707, b2 0.5/0.1414
    means = {"b0": 0.55, "b1": 0.75, "b2": 0.5}
    stds = {"b0": np.sqrt(0.005), "b1": np.sqrt(0.005), "b2": np.sqrt(0.02)}
    target_mean, target_std = 0.55, np.sqrt(0.005)
    expected = [
        (v - means[b]) / stds[b] * target_std + target_mean
        for v, b in zip(df["auroc_pooled"], df["meta_experiment"])
    ]
    np.testing.assert_allclose(df["auroc_pooled_corrected"].to_numpy(), expected)


def test_correct_reference_all(batch_scores):
    df = batch_scores.correct(reference="all", output_col="c").df
    stats = df.group_by("meta_experiment").agg(
        pl.col("auroc_pooled").mean().alias("raw_mean"),
        pl.col("auroc_pooled").std().alias("raw_std"),
        pl.col("c").mean().alias("mean"),
        pl.col("c").std().alias("std"),
    )
    # every batch ends up with the median batch mean and std
    np.testing.assert_allclose(stats["mean"].to_numpy(), stats["raw_mean"].median())
    np.testing.assert_allclose(stats["std"].to_numpy(), stats["raw_std"].median())
    with pytest.raises(ValueError, match="reference"):
        batch_scores.correct(reference="other")


def test_per_variant(batch_scores):
    df = batch_scores.per_variant().df
    assert df["meta_aa_changes"].to_list() == ["A1A", "C2C", "D3V"]
    assert df["auroc_pooled"].to_list() == pytest.approx([0.5, 0.6, 0.9])
    assert df["meta_n_experiments"].to_list() == [3, 3, 3]
    assert df["meta_n_cells"].to_list() == [30, 30, 30]


def test_distinguishability_join(batch_scores):
    profiles = fb.Profiles(pl.DataFrame({"meta_aa_changes": ["D3V", "A1A", "Z9Z"]}))
    corrected = batch_scores.correct().per_variant("auroc_pooled_corrected").df
    joined = profiles.distinguishability(batch_scores).df
    expected = dict(zip(corrected["meta_aa_changes"], corrected["auroc_pooled_corrected"]))
    assert joined["meta_distinguishability_score"].to_list()[:2] == pytest.approx(
        [expected["D3V"], expected["A1A"]]
    )
    assert joined["meta_distinguishability_score"][2] is None
    raw = profiles.distinguishability(batch_scores, reference=None).df
    assert raw["meta_distinguishability_score"].to_list()[:2] == pytest.approx([0.9, 0.5])
    # an already per-variant table is joined as-is
    as_is = profiles.distinguishability(batch_scores.per_variant(), output_col="s").df
    assert as_is["s"].to_list()[:2] == pytest.approx([0.9, 0.5])


def test_from_global(tmp_path):
    path = tmp_path / "global" / "main" / "ovwt_distinguishability"
    path.mkdir(parents=True)
    pl.DataFrame(
        {"meta_aa_changes": ["A1V"], "meta_median_auroc_pooled": [1.5], "meta_num_experiments": [3]}
    ).write_parquet(path / "global_scores.parquet")
    scores = fb.OvwtScores.from_global(tmp_path, "main")
    profiles = fb.Profiles(pl.DataFrame({"meta_aa_changes": ["A1V"]}))
    joined = profiles.distinguishability(scores, score="meta_median_auroc_pooled").df
    assert joined["meta_distinguishability_score"].to_list() == [1.5]


def test_blocklists(pipeline_dir):
    blocklists = fb.Blocklists.from_pipeline(pipeline_dir)
    df = blocklists.df
    assert set(df["meta_feature_type"]) == {"median", "KS"}
    assert df.height == 3 * 2 * len(PIPELINE_BATCHES)
    intensity = "Mean_Nuclei_Intensity_MeanIntensity_CH1"
    assert set(blocklists.consensus()) == {
        "AreaShape_Area_median",
        "AreaShape_Area_KS",
        f"{intensity}_median",
        f"{intensity}_KS",
    }
    # the intensity feature's median_r is 0.6 in one batch and 0.8 in the others
    strict = blocklists.rethreshold(0.7)
    assert set(strict.consensus()) == {"AreaShape_Area_median", "AreaShape_Area_KS"}
    assert len(strict.consensus(min_batches=2)) == 4
    only_median = fb.Blocklists.from_pipeline(pipeline_dir, types=["median"])
    assert set(only_median.df["meta_feature_type"]) == {"median"}
