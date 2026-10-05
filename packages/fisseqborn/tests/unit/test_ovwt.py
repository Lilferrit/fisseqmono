import numpy as np
import polars as pl
import pytest
from fisseqborn_testdata import PIPELINE_BATCHES, PIPELINE_VARIANTS

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


def test_from_global_is_deprecated(tmp_path):
    path = tmp_path / "global" / "main" / "ovwt_distinguishability"
    path.mkdir(parents=True)
    pl.DataFrame(
        {"meta_aa_changes": ["A1V"], "meta_median_auroc_pooled": [1.5], "meta_num_experiments": [3]}
    ).write_parquet(path / "global_scores.parquet")
    with pytest.warns(DeprecationWarning, match="global_scores.parquet"):
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
    assert len(blocklists.rethreshold(0.6).consensus()) == 4  # median_r >= min_r
    strict = blocklists.rethreshold(0.7)
    assert set(strict.consensus()) == {"AreaShape_Area_median", "AreaShape_Area_KS"}
    assert len(strict.consensus(min_batches=2)) == 4
    only_median = fb.Blocklists.from_pipeline(pipeline_dir, types=["median"])
    assert set(only_median.df["meta_feature_type"]) == {"median"}


def test_from_pipeline_reads_legacy_schema(tmp_path):
    root = tmp_path / "old_run"
    for b, batch in enumerate(["T1_R1", "T2_R1"]):
        d = root / "ovwt_batchwise" / batch
        d.mkdir(parents=True)
        pl.DataFrame(
            {
                "variant": ["A1A", "C2C", "D3V"],
                "train_auroc": [0.9, 0.9, 0.9],
                "test_auroc": [0.5 + b / 10, 0.6, 0.9],
                "meta_num_cells": pl.Series([10, 20, 30], dtype=pl.UInt32),
                "meta_barcode_num_unique": pl.Series([1, 2, 3], dtype=pl.UInt32),
            }
        ).write_parquet(d / "results.parquet")
    scores = fb.OvwtScores.from_pipeline(root)
    assert {"meta_aa_changes", "test_auroc", "meta_n_cells", "meta_n_barcodes"} <= set(
        scores.columns
    )
    assert "variant" not in scores.columns

    per_variant = scores.per_variant("test_auroc").df.sort("meta_aa_changes")
    assert per_variant["meta_n_cells"].to_list() == [20, 40, 60]
    assert per_variant["test_auroc"].to_list() == pytest.approx([0.55, 0.6, 0.9])

    profiles = fb.Profiles(pl.DataFrame({"meta_aa_changes": ["A1A", "D3V"], "f_median": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="test_auroc"):
        profiles.distinguishability(scores)
    joined = profiles.distinguishability(scores, score="test_auroc", reference=None).df
    assert joined["meta_distinguishability_score"].to_list() == pytest.approx([0.55, 0.9])


def test_blocklist_table_and_missing(pipeline_dir):
    blocklists = fb.Blocklists.from_pipeline(pipeline_dir, types=["median"])
    table = blocklists.rethreshold(0.7).table()
    assert table.columns == ["feature", "n_batches", "n_ok", "feature_ok"]
    rows = {r["feature"]: r for r in table.iter_rows(named=True)}
    intensity = rows["Mean_Nuclei_Intensity_MeanIntensity_CH1_median"]
    assert (intensity["n_batches"], intensity["n_ok"], intensity["feature_ok"]) == (3, 2, False)
    assert rows["AreaShape_Area_median"]["feature_ok"]
    assert blocklists.rethreshold(0.7).table(min_batches=2).filter("feature_ok").height == 2

    # T1_R1's blocklist no longer reports the area feature
    path = pipeline_dir / "feature_select_batchwise" / "T1_R1" / "blocklists" / "median.parquet"
    pl.read_parquet(path).filter(pl.col("feature") != "AreaShape_Area_median").write_parquet(path)
    partial = fb.Blocklists.from_pipeline(pipeline_dir, types=["median"])
    assert "AreaShape_Area_median" not in partial.consensus()
    assert "AreaShape_Area_median" in partial.consensus(missing="ignore")
    area = partial.table(missing="ignore").filter(pl.col("feature") == "AreaShape_Area_median")
    assert area.row(0) == ("AreaShape_Area_median", 2, 2, True)
    assert partial.table().filter(pl.col("feature") == "AreaShape_Area_median")["feature_ok"][0] is False
    with pytest.raises(ValueError, match="missing"):
        partial.consensus(missing="drop")


def test_correct_defaults_to_every_score(pipeline_dir):
    scores = fb.OvwtScores.from_pipeline(pipeline_dir)
    corrected = scores.correct()
    assert {f"{s}_corrected" for s in fb.ovwt.SCORES} <= set(corrected.columns)
    assert "auroc_folds_corrected" not in corrected.columns
    # each score is corrected as it would be on its own
    alone = scores.correct("auroc_median_barcode").df["auroc_median_barcode_corrected"]
    assert corrected.df["auroc_median_barcode_corrected"].to_list() == pytest.approx(alone.to_list())
    with pytest.raises(ValueError, match="output_col"):
        scores.correct(["auroc_pooled", "auroc_median_fold"], output_col="x")


def test_correct_without_rescale_is_a_zscore(batch_scores):
    df = batch_scores.correct(rescale=False).df
    means = {"b0": 0.55, "b1": 0.75, "b2": 0.5}
    stds = {"b0": np.sqrt(0.005), "b1": np.sqrt(0.005), "b2": np.sqrt(0.02)}  # ddof=1
    expected = [
        (v - means[b]) / stds[b] for v, b in zip(df["auroc_pooled"], df["meta_experiment"])
    ]
    np.testing.assert_allclose(df["auroc_pooled_corrected"].to_numpy(), expected)


def test_correct_constant_and_missing_controls_give_null(caplog):
    scores = fb.OvwtScores(
        pl.DataFrame(
            {
                "meta_aa_changes": ["A1A", "C2C", "D3V", "A1A", "D3V", "A1A", "C2C", "D3V"],
                "meta_experiment": ["flat"] * 3 + ["one"] * 2 + ["ok"] * 3,
                "auroc_pooled": [0.6, 0.6, 0.9, 0.5, 0.9, 0.5, 0.7, 0.9],
            }
        )
    )
    with caplog.at_level("WARNING"):
        df = scores.correct(rescale=False).df
    by_batch = dict(
        df.group_by("meta_experiment").agg(pl.col("auroc_pooled_corrected").is_null().all()).iter_rows()
    )
    assert by_batch == {"flat": True, "one": True, "ok": False}
    # only the experiment with a single synonymous control is warned about
    assert len(caplog.records) == 1
    assert "one" in caplog.records[0].getMessage()
    assert "fewer than 2" in caplog.records[0].getMessage()
    # the rescaled correction also gives null rather than inf for a zero std
    assert df.filter(pl.col("meta_experiment") == "flat")["auroc_pooled_corrected"].null_count() == 3
    rescaled = scores.correct().df.filter(pl.col("meta_experiment") == "flat")
    assert rescaled["auroc_pooled_corrected"].null_count() == 3


def test_auroc_folds_is_not_a_score(pipeline_dir):
    scores = fb.OvwtScores.from_pipeline(pipeline_dir)
    with pytest.raises(ValueError, match="auroc_folds"):
        scores.correct("auroc_folds")
    with pytest.raises(ValueError, match="auroc_folds"):
        scores.per_variant(["auroc_pooled", "auroc_folds"])
    legacy = fb.OvwtScores(pl.DataFrame({"meta_aa_changes": ["A1A"], "test_auroc": [0.5]}))
    with pytest.raises(ValueError, match="test_auroc"):
        legacy.correct()


def test_per_variant_several_scores(batch_scores):
    two = batch_scores.with_columns((pl.col("auroc_pooled") * 2).alias("auroc_median_fold"))
    two = two.with_columns(
        pl.when(pl.col("meta_experiment") == "b2").then(None).otherwise(pl.col("auroc_pooled")).alias(
            "auroc_pooled"
        )
    )
    df = two.per_variant(["auroc_pooled", "auroc_median_fold"]).df
    assert df["auroc_pooled"].to_list() == pytest.approx([0.6, 0.7, 0.95])
    assert df["auroc_median_fold"].to_list() == pytest.approx([1.0, 1.2, 1.8])
    # b2 still has auroc_median_fold, so it still counts
    assert df["meta_n_experiments"].to_list() == [3, 3, 3]


def test_distinguishability_without_rescale(batch_scores):
    profiles = fb.Profiles(pl.DataFrame({"meta_aa_changes": ["D3V"]}))
    joined = profiles.distinguishability(batch_scores, rescale=False).df
    expected = batch_scores.correct(rescale=False).per_variant("auroc_pooled_corrected").df
    assert joined["meta_distinguishability_score"][0] == pytest.approx(
        expected.filter(pl.col("meta_aa_changes") == "D3V")["auroc_pooled_corrected"][0]
    )
