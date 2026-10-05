import numpy as np
import polars as pl
import pytest

import fisseqborn as fb
from fisseqborn import fisseq


@pytest.fixture
def variants(profiles) -> fb.Dataset:
    rng = np.random.default_rng(5)
    domains = rng.choice([*fisseq.LMNA_DOMAIN_REGIONS, None], size=profiles.height)
    return fb.Dataset(
        profiles.with_columns(
            pl.Series("meta_domain", domains, dtype=pl.String),
            (pl.col("meta_variant_type") == "Synonymous").alias("meta_is_control"),
        )
    )


CLASSES = {
    fisseq.PATHOGENIC: pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC,
    "Synonymous": pl.col("meta_variant_type") == "Synonymous",
}


def test_cluster_summary_matches_group_by(variants):
    summary = variants.cluster_summary(
        medians={
            "feature_1": "feature_1",
            "meta_distinguishability_score": "Median score",
        },
        zscore=["feature_1"],
        shares={"class": CLASSES, "domain": "meta_domain"},
    )
    df = summary.df
    assert df["meta_cluster_idx"].to_list() == [str(i) for i in range(12)]

    src = variants.df
    syn = src.filter("meta_is_control")["feature_1"]
    expected = (
        src.with_columns((pl.col("feature_1") - syn.mean()) / syn.std())
        .group_by("meta_cluster_idx")
        .agg(
            pl.len().alias("n"),
            pl.col("feature_1").median(),
            pl.col("meta_distinguishability_score").median().alias("Median score"),
            *[e.sum().alias(k) for k, e in CLASSES.items()],
        )
        .sort(pl.col("meta_cluster_idx").cast(pl.Int64))
    )
    for col in ["n", "feature_1", "Median score"]:
        np.testing.assert_allclose(df[col], expected[col])
    for level in CLASSES:
        np.testing.assert_allclose(df[level], expected[level] / expected[level].sum())
    assert df["label"][0] == f"0 (n={expected['n'][0]})"

    present = set(src["meta_domain"].drop_nulls())
    assert summary.shares("domain") == [
        d for d in fisseq.LMNA_DOMAIN_REGIONS if d in present
    ]
    for col in summary.shares("class") + summary.shares("domain"):
        assert df[col].sum() == pytest.approx(1.0)
    assert summary.totals["class"]["Synonymous"] == int(
        src["meta_variant_type"].eq("Synonymous").sum()
    )
    assert summary.totals["domain"]["Head"] == int(src["meta_domain"].eq("Head").sum())


def test_cluster_summary_group_feeds_clustermap(variants):
    summary = variants.cluster_summary(
        medians=["feature_0", "feature_1"],
        shares={"class": CLASSES, "domain": "meta_domain"},
    )
    group = summary.group("class", "Share of class", annot=True)
    assert group.name == "Share of class" and group.vmin == 0 and group.annot
    n_syn = summary.totals["class"]["Synonymous"]
    assert group.labels["Synonymous"] == f"Synonymous (n={n_syn})"
    _, axes = fb.ClusterMap(
        summary,
        row_labels="label",
        standardize=True,
        groups=[
            fb.FeatureGroup("Medians", ["feature_0", "feature_1"]),
            group,
            summary.group("domain"),
        ],
    ).plot()
    assert set(axes.heatmap_axes) == {"Medians", "Share of class", "domain"}


def test_cluster_summary_explicit_levels_and_errors(variants):
    domains = list(fisseq.LMNA_DOMAIN_REGIONS)
    summary = variants.filter(pl.col("meta_domain") != "Head").cluster_summary(
        shares={"domain": "meta_domain"}, levels={"domain": domains}
    )
    assert summary.shares("domain") == domains
    assert summary.totals["domain"]["Head"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        variants.cluster_summary(shares={"a": CLASSES, "b": CLASSES})
    with pytest.raises(ValueError, match="variant_type"):
        variants.drop("meta_is_control").cluster_summary(
            medians=["feature_0"], zscore=True
        )
    with pytest.raises(KeyError):
        summary.group("class")


def test_cluster_summary_is_lazy(variants):
    summary = variants.cluster_summary(shares={"class": CLASSES})
    assert summary._df is None
