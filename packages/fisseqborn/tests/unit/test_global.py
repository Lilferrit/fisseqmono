import numpy as np
import polars as pl
import pytest
from fisseqborn_testdata import PIPELINE_BATCHES, PIPELINE_VARIANTS

import fisseqborn as fb
from fisseqborn import global_aggregate


def _files(out):
    return {p.relative_to(out).as_posix() for p in out.rglob("*.parquet")}


def test_cli_writes_the_old_global_artifacts(pipeline_dir, tmp_path, capsys):
    pytest.importorskip("pycytominer")
    out = tmp_path / "global"
    global_aggregate.main(
        [
            str(pipeline_dir),
            "--out",
            str(out),
            "--types",
            "median",
            "KS",
            "--passthrough",
            "KSnegLogP",
            "--paired",
            "median:KSnegLogP",
            "--metadata",
            "--pca",
            "2",
        ]
    )
    assert _files(out) == {
        "feature_select/aggregate.parquet",
        "feature_select/blocklist.parquet",
        "feature_select/pca_components.parquet",
        "ovwt_distinguishability/global_scores.parquet",
    }
    assert len(capsys.readouterr().out.splitlines()) == 4

    aggregate = pl.read_parquet(out / "feature_select" / "aggregate.parquet")
    assert aggregate["meta_aa_changes"].to_list() == PIPELINE_VARIANTS
    assert {"meta_impact_score", "meta_variant_type", "meta_pc_1", "meta_pc_2"} <= set(
        aggregate.columns
    )
    assert aggregate["meta_n_experiments"].to_list() == [3] * len(PIPELINE_VARIANTS)
    assert aggregate["meta_num_cells"].to_list() == [60] * len(PIPELINE_VARIANTS)
    # Constant fails the blocklist in every batch
    assert not any(c.startswith("Constant") for c in aggregate.columns)
    features = [c for c in aggregate.columns if not c.startswith("meta_")]
    assert features and set(features) <= {
        f"{f}_{s}"
        for f in ("AreaShape_Area", "Mean_Nuclei_Intensity_MeanIntensity_CH1")
        for s in ("median", "KS", "KSnegLogP")
    }

    blocklist = pl.read_parquet(out / "feature_select" / "blocklist.parquet")
    assert blocklist.columns == ["feature", "n_batches", "n_ok", "feature_ok"]
    assert blocklist.height == 6

    loadings = pl.read_parquet(out / "feature_select" / "pca_components.parquet")
    assert loadings["component"].to_list() == ["meta_pc_1", "meta_pc_2"]

    global_scores = pl.read_parquet(out / "ovwt_distinguishability" / "global_scores.parquet")
    assert global_scores.columns == [
        "meta_aa_changes",
        "meta_median_auroc_pooled",
        "meta_median_auroc_median_barcode",
        "meta_median_auroc_median_fold",
        "meta_num_experiments",
    ]
    assert global_scores["meta_num_experiments"].to_list() == [len(PIPELINE_BATCHES)] * len(
        PIPELINE_VARIANTS
    )
    # z-score each experiment against its synonymous controls, then take the median
    raw = fb.OvwtScores.from_pipeline(pipeline_dir).variant_type().df
    controls = (
        raw.filter("meta_is_control")
        .group_by("meta_experiment")
        .agg(pl.col("auroc_pooled").mean().alias("mean"), pl.col("auroc_pooled").std().alias("std"))
    )
    expected = (
        raw.join(controls, on="meta_experiment")
        .group_by("meta_aa_changes", maintain_order=True)
        .agg(((pl.col("auroc_pooled") - pl.col("mean")) / pl.col("std")).median())
    )
    np.testing.assert_allclose(
        global_scores["meta_median_auroc_pooled"].to_numpy(), expected["auroc_pooled"].to_numpy()
    )


def test_write_global_options(pipeline_dir, tmp_path):
    out = tmp_path / "global"
    written = fb.write_global(
        pipeline_dir, out, exclude="T10_*", operations=(), min_correlation=0.7, ovwt=False
    )
    assert set(written) == {"aggregate", "blocklist"}
    blocklist = pl.read_parquet(written["blocklist"])
    assert blocklist["n_batches"].to_list() == [2, 2, 2]
    # intensity has median_r 0.6 in T2_R1, so it fails at 0.7
    assert blocklist["feature_ok"].to_list() == [True, False, False]
    aggregate = pl.read_parquet(written["aggregate"])
    assert [c for c in aggregate.columns if not c.startswith("meta_")] == ["AreaShape_Area_median"]
    assert aggregate["meta_n_experiments"].to_list() == [2] * len(PIPELINE_VARIANTS)


def test_cli_exclude_regex_and_bad_paired(pipeline_dir, tmp_path):
    out = tmp_path / "global"
    global_aggregate.main(
        [str(pipeline_dir), "--out", str(out), "--exclude-regex", "^T1_", "--operations"]
    )
    scores = pl.read_parquet(out / "ovwt_distinguishability" / "global_scores.parquet")
    assert scores["meta_num_experiments"].to_list() == [2] * len(PIPELINE_VARIANTS)
    with pytest.raises(SystemExit):
        global_aggregate.main([str(pipeline_dir), "--out", str(out), "--paired", "median"])


def test_documented_chain_matches_write_global(pipeline_dir, tmp_path):
    pytest.importorskip("pycytominer")
    run, out = pipeline_dir, tmp_path / "global"
    written = fb.write_global(
        run,
        out,
        types=["median", "KS"],
        passthrough=["KSnegLogP"],
        paired={"_median": "_KSnegLogP"},
        min_correlation=0.7,
        metadata=True,
        pca=2,
    )

    # the chain from docs/data.md ("Reproduce the old global feature select")
    blocklists = fb.Blocklists.from_pipeline(run, types=["median", "KS"]).rethreshold(0.7)
    table = blocklists.table()
    ok = table.filter("feature_ok")["feature"].to_list()
    assert ok == blocklists.consensus()
    aggregate = (
        fb.Profiles.from_pipeline(
            run, types=["median", "KS"], passthrough=["KSnegLogP"], metadata=True
        )
        .keep_features(ok)
        .median_across_batches(
            paired={"_median": "_KSnegLogP"},
            sum_cols=["meta_num_cells", "meta_barcode_num_unique"],
        )
        .variant_type()
        .feature_select()
        .impact_score()
    )
    pcs = aggregate.drop_nonfinite().pca(2)
    aggregate = aggregate.with_columns(pcs.df.select("^meta_pc_.*$").get_columns())
    scores = fb.ovwt.SCORES
    global_scores = (
        fb.OvwtScores.from_pipeline(run)
        .correct(scores, rescale=False)
        .per_variant([f"{s}_corrected" for s in scores], n_col="meta_num_experiments")
        .select(
            "meta_aa_changes",
            *[pl.col(f"{s}_corrected").alias(f"meta_median_{s}") for s in scores],
            "meta_num_experiments",
        )
    )

    assert pl.read_parquet(written["blocklist"]).equals(table)
    assert pl.read_parquet(written["aggregate"]).equals(aggregate.df)
    assert pl.read_parquet(written["pca_components"]).equals(pcs.pca_loadings)
    assert pl.read_parquet(written["global_scores"]).equals(global_scores.df)
