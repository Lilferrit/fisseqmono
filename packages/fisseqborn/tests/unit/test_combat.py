"""Profiles.combat / OvwtScores.combat (fisseq_common.combat) and the readers of the
pipeline's combat/ outputs."""

import logging

import numpy as np
import polars as pl
import pytest
from fisseqborn_testdata import PIPELINE_BATCHES, PIPELINE_VARIANTS

import fisseqborn as fb


def test_profiles_combat_corrects_values_and_leaves_passthrough(pipeline_dir) -> None:
    profiles = fb.Profiles.from_pipeline(
        pipeline_dir, types=["median"], passthrough=["KSnegLogP"]
    )
    out = profiles.combat()
    assert out.columns == profiles.columns
    assert out.df.height == profiles.df.height
    joined = out.df.join(
        profiles.df, on=["meta_aa_changes", "meta_experiment"], suffix="_raw"
    )
    p = "AreaShape_Area_KSnegLogP"
    assert (joined[p] == joined[f"{p}_raw"]).all()
    v = "AreaShape_Area_median"
    assert not np.allclose(joined[v], joined[f"{v}_raw"])
    # the fixture's batches are shifted by 0, 1 and 2: corrected, their means line up
    means = out.df.group_by("meta_experiment").agg(pl.col(v).mean())[v]
    raw = profiles.df.group_by("meta_experiment").agg(pl.col(v).mean())[v]
    assert np.ptp(means.to_numpy()) < np.ptp(raw.to_numpy())


def test_profiles_combat_drops_single_batch_variants(pipeline_dir, caplog) -> None:
    profiles = fb.Profiles.from_pipeline(pipeline_dir, types=["median"])
    profiles = profiles.filter(
        ~((pl.col("meta_aa_changes") == "H7*") & (pl.col("meta_experiment") != "T1_R1"))
    )
    with caplog.at_level(logging.WARNING, logger="fisseq_common.combat"):
        out = profiles.combat()
    assert "H7*" not in out.df["meta_aa_changes"].to_list()
    assert "only one experiment (H7*)" in caplog.text


def test_ovwt_combat_writes_corrected_columns(pipeline_dir) -> None:
    scores = fb.OvwtScores.from_pipeline(pipeline_dir)
    out = scores.combat()
    for s in ("auroc_pooled", "auroc_median_barcode", "auroc_median_fold"):
        assert f"{s}_corrected" in out.columns
        assert s in out.columns
    assert out.df.height == len(PIPELINE_VARIANTS) * len(PIPELINE_BATCHES)
    # chains like correct() does
    per_variant = out.per_variant("auroc_pooled_corrected")
    assert per_variant.df.height == len(PIPELINE_VARIANTS)


def test_ovwt_combat_one_score(pipeline_dir) -> None:
    out = fb.OvwtScores.from_pipeline(pipeline_dir).combat("auroc_pooled")
    assert "auroc_pooled_corrected" in out.columns
    assert "auroc_median_barcode_corrected" not in out.columns
    assert out.df["auroc_pooled_corrected"].is_finite().all()


def _write_combat(root, name: str, df: pl.DataFrame) -> None:
    path = root / "combat" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def test_readers(pipeline_dir) -> None:
    profiles = fb.Profiles.from_pipeline(pipeline_dir, types=["median"]).combat()
    scores = fb.OvwtScores.from_pipeline(pipeline_dir)
    _write_combat(pipeline_dir, "features.parquet", profiles.df)
    _write_combat(pipeline_dir, "ovwt_scores.parquet", scores.df.drop("auroc_folds"))
    read = fb.Profiles.from_combat(pipeline_dir, layout="data")
    assert read.df.equals(profiles.df)
    ovwt = fb.OvwtScores.from_combat(pipeline_dir, layout="data")
    assert ovwt.batch_col == "meta_experiment"
    assert "auroc_folds" not in ovwt.columns


def test_reader_names_the_missing_file(pipeline_dir) -> None:
    with pytest.raises(FileNotFoundError, match="combat/features.parquet"):
        fb.Profiles.from_combat(pipeline_dir, layout="data").df
