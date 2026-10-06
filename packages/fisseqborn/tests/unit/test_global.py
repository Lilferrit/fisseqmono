import numpy as np
import polars as pl
import pytest
from fisseqborn_testdata import PIPELINE_BATCHES, PIPELINE_VARIANTS

import fisseqborn as fb
from fisseq_common.global_aggregation import blocklist_vote, median_across_batches
from fisseqborn import global_aggregate

TRACK_FILES = {
    "blocklist",
    "median_aggregate",
    "pca_scores",
    "pca_components",
    "pca_variance_explained",
    "pca_reduced",
}


def _files(out):
    return {p.relative_to(out).as_posix() for p in out.rglob("*.parquet")}


def test_cli_writes_the_global_outputs(pipeline_dir, tmp_path, capsys):
    out = tmp_path / "global"
    global_aggregate.main([str(pipeline_dir), "--out", str(out)])
    assert _files(out) == {f"feature_select/{f}.parquet" for f in TRACK_FILES} | {
        "ovwt_distinguishability/global_scores.parquet"
    }
    assert len(capsys.readouterr().out.splitlines()) == len(TRACK_FILES) + 1

    fs = out / "feature_select"
    # Every method the run aggregated (median and KS); Constant fails the blocklist in
    # every batch.
    median = pl.read_parquet(fs / "median_aggregate.parquet")
    assert median["meta_aa_changes"].to_list() == sorted(PIPELINE_VARIANTS)
    features = [c for c in median.columns if c != "meta_aa_changes"]
    assert features == sorted(
        f"{f}_{s}"
        for f in ("AreaShape_Area", "Mean_Nuclei_Intensity_MeanIntensity_CH1")
        for s in ("median", "KS")
    )

    blocklist = pl.read_parquet(fs / "blocklist.parquet")
    assert blocklist.columns == ["feature", "n_batches", "n_ok", "feature_ok"]
    assert blocklist["feature"].to_list() == sorted(blocklist["feature"].to_list())
    assert blocklist.height == 6

    # Full rank: as many components as the data supports.
    variance = pl.read_parquet(fs / "pca_variance_explained.parquet")
    n = min(len(PIPELINE_VARIANTS), len(features))
    assert variance["meta_component_idx"].to_list() == list(range(1, n + 1))
    scores = pl.read_parquet(fs / "pca_scores.parquet")
    assert scores.columns == [
        "meta_aa_changes",
        *[f"meta_pc_{i}" for i in range(1, n + 1)],
    ]
    components = pl.read_parquet(fs / "pca_components.parquet")
    assert components.columns == ["meta_component_idx", *features]
    reduced = pl.read_parquet(fs / "pca_reduced.parquet")
    k = next(
        i
        for i, c in enumerate(variance["meta_cumulative_variance_explained"], start=1)
        if c >= 0.9
    )
    assert reduced.columns == [
        "meta_aa_changes",
        *[f"meta_pc_{i}" for i in range(1, k + 1)],
        "meta_is_control",
        "meta_impact_score",
    ]

    global_scores = pl.read_parquet(
        out / "ovwt_distinguishability" / "global_scores.parquet"
    )
    assert global_scores.columns == [
        "meta_aa_changes",
        "meta_median_auroc_pooled",
        "meta_median_auroc_median_barcode",
        "meta_median_auroc_median_fold",
        "meta_num_experiments",
    ]
    assert global_scores["meta_num_experiments"].to_list() == [
        len(PIPELINE_BATCHES)
    ] * len(PIPELINE_VARIANTS)
    # z-score each experiment against its synonymous controls, then take the median
    raw = fb.OvwtScores.from_pipeline(pipeline_dir).variant_type().df
    controls = (
        raw.filter("meta_is_control")
        .group_by("meta_experiment")
        .agg(
            pl.col("auroc_pooled").mean().alias("mean"),
            pl.col("auroc_pooled").std().alias("std"),
        )
    )
    expected = (
        raw.join(controls, on="meta_experiment")
        .group_by("meta_aa_changes")
        .agg(((pl.col("auroc_pooled") - pl.col("mean")) / pl.col("std")).median())
        .sort("meta_aa_changes")
    )
    np.testing.assert_allclose(
        global_scores["meta_median_auroc_pooled"].to_numpy(),
        expected["auroc_pooled"].to_numpy(),
    )


def test_pooling_is_vote_then_drop_then_median(pipeline_dir, tmp_path):
    """median_aggregate: the per-experiment aggregates, minus the features the vote marks
    not OK, medianed across experiments."""
    written = fb.write_global(
        pipeline_dir, tmp_path / "global", types=["median"], ovwt=False
    )
    blocklists = [
        pl.read_parquet(
            pipeline_dir / "feature_select_batchwise" / b / "blocklists" / f
        )
        for b in PIPELINE_BATCHES
        for f in ("median.parquet", "KS.parquet")
    ]
    vote = blocklist_vote(blocklists)
    assert pl.read_parquet(written["feature_select/blocklist"]).equals(vote)
    blocked = vote.filter(~pl.col("feature_ok"))["feature"].to_list()
    frames = [
        pl.scan_parquet(
            pipeline_dir
            / "feature_select_batchwise"
            / b
            / "aggregates"
            / "median.parquet"
        ).drop(blocked, strict=False)
        for b in PIPELINE_BATCHES
    ]
    expected = median_across_batches(frames, "meta_aa_changes").sort("meta_aa_changes")
    assert pl.read_parquet(written["feature_select/median_aggregate"]).equals(expected)


def test_write_global_options(pipeline_dir, tmp_path):
    written = fb.write_global(
        pipeline_dir,
        tmp_path / "global",
        types=["median"],
        exclude="T10_*",
        min_correlation=0.7,
        ovwt=False,
    )
    assert set(written) == {f"feature_select/{f}" for f in TRACK_FILES}
    blocklist = pl.read_parquet(written["feature_select/blocklist"])
    # Both methods' blocklists vote; T10_R1 is excluded.
    assert blocklist["n_batches"].to_list() == [2] * 6
    # intensity has median_r 0.6 in T2_R1, so it fails at 0.7
    ok = dict(blocklist.select("feature", "feature_ok").iter_rows())
    assert ok["AreaShape_Area_median"] and not ok["Constant_median"]
    assert not ok["Mean_Nuclei_Intensity_MeanIntensity_CH1_median"]
    median = pl.read_parquet(written["feature_select/median_aggregate"])
    assert median.columns == ["meta_aa_changes", "AreaShape_Area_median"]


def test_min_batches_and_missing(pipeline_dir, tmp_path):
    # T1_R1's blocklist no longer reports the area feature.
    path = (
        pipeline_dir
        / "feature_select_batchwise"
        / "T1_R1"
        / "blocklists"
        / "median.parquet"
    )
    pl.read_parquet(path).filter(
        pl.col("feature") != "AreaShape_Area_median"
    ).write_parquet(path)

    def area(**kw):
        written = fb.write_global(
            pipeline_dir, tmp_path / str(kw), types=["median"], ovwt=False, **kw
        )
        vote = pl.read_parquet(written["feature_select/blocklist"])
        return vote.filter(pl.col("feature") == "AreaShape_Area_median").row(0)[1:]

    assert area() == (2, 2, True)  # judged on the two batches reporting it
    assert area(missing="fail") == (2, 2, False)
    assert area(min_batches=3) == (2, 2, False)


def test_opt_in_extras(pipeline_dir, tmp_path):
    pytest.importorskip("pycytominer")
    written = fb.write_global(
        pipeline_dir,
        tmp_path / "global",
        types=["median", "KS"],
        passthrough=["KSnegLogP"],
        paired={"_median": "_KSnegLogP"},
        operations=global_aggregate.OPERATIONS,
        impact_score=True,
        metadata=True,
        ovwt=False,
    )
    median = pl.read_parquet(written["feature_select/median_aggregate"])
    assert {"meta_impact_score", "meta_num_cells", "meta_barcode_num_unique"} <= set(
        median.columns
    )
    assert median["meta_num_cells"].to_list() == [60] * len(PIPELINE_VARIANTS)


def test_cli_exclude_regex_and_bad_paired(pipeline_dir, tmp_path):
    out = tmp_path / "global"
    global_aggregate.main(
        [str(pipeline_dir), "--out", str(out), "--exclude-regex", "^T1_"]
    )
    scores = pl.read_parquet(out / "ovwt_distinguishability" / "global_scores.parquet")
    assert scores["meta_num_experiments"].to_list() == [2] * len(PIPELINE_VARIANTS)
    with pytest.raises(SystemExit):
        global_aggregate.main(
            [str(pipeline_dir), "--out", str(out), "--paired", "median"]
        )


def test_pca_argument_is_deprecated(pipeline_dir, tmp_path):
    with pytest.warns(DeprecationWarning, match="pca"):
        fb.write_global(pipeline_dir, tmp_path / "global", pca=2, ovwt=False)
