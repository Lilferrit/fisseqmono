import numpy as np
import polars as pl
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

import fisseqborn as fb
from fisseqborn import FeatureGroup


@pytest.fixture
def cluster_summary() -> pl.DataFrame:
    """One row per cluster: two structured feature blocks plus a display-only score."""
    rng = np.random.default_rng(0)
    n = 12
    blocks = np.repeat([[-2.0], [0.0], [2.0]], 4, axis=0)  # three row blocks
    return pl.DataFrame(
        {
            "meta_cluster_idx": [str(i) for i in range(n)],
            **{f"landmark_{i}": (blocks[:, 0] + rng.normal(0, 0.1, n)) for i in range(4)},
            "prop_synonymous": rng.random(n),
            "prop_frameshift": rng.random(n),
            # large-scale noise that would dominate unscaled distances if it were clustered
            "median_score": rng.normal(0, 100, n),
        }
    )


GROUPS = [
    FeatureGroup("Landmarks", [f"landmark_{i}" for i in range(4)], cmap="RdBu_r", center=0),
    FeatureGroup("Class proportion", ["prop_synonymous", "prop_frameshift"], cmap="Greens", vmin=0),
    FeatureGroup("Score", "median_score", cluster=False, cmap="Reds", annot=True),
]


def _expected_order(df: pl.DataFrame, cols: list[str], standardize: bool = False) -> list[int]:
    x = df.select(cols).to_numpy()
    if standardize:
        x = (x - x.mean(0)) / x.std(0)
    return leaves_list(linkage(pdist(x), method="average")).tolist()


def test_row_order_uses_only_clustering_groups(cluster_summary):
    plot = fb.ClusterMap(cluster_summary, groups=GROUPS, row_labels="meta_cluster_idx")
    clustered = [f"landmark_{i}" for i in range(4)] + ["prop_synonymous", "prop_frameshift"]
    assert plot.row_order() == _expected_order(cluster_summary, clustered)
    with_score = _expected_order(cluster_summary, clustered + ["median_score"])
    assert plot.row_order() != with_score


def test_standardize(cluster_summary):
    plot = fb.ClusterMap(cluster_summary, groups=GROUPS, standardize=True)
    clustered = [f"landmark_{i}" for i in range(4)] + ["prop_synonymous", "prop_frameshift"]
    assert plot.row_order() == _expected_order(cluster_summary, clustered, standardize=True)


def test_no_clustering_group_keeps_input_order(cluster_summary):
    groups = [FeatureGroup(g.name, g.features, cluster=False) for g in GROUPS]
    plot = fb.ClusterMap(cluster_summary, groups=groups)
    fig, axes = plot.plot()
    assert axes.row_order == list(range(cluster_summary.height))
    assert axes.row_dendrogram_ax is None and axes.row_linkage is None


def test_draws_one_block_per_group(cluster_summary, tmp_path):
    fig, axes = (
        fb.ClusterMap(cluster_summary, groups=GROUPS, row_labels="meta_cluster_idx",
                      title="Cluster summary")
        .save(tmp_path / "cm.png")
        .plot()
    )
    assert list(axes.heatmap_axes) == ["Landmarks", "Class proportion", "Score"]
    assert [ax.get_title() for ax in axes.colorbar_axes.values()] == list(axes.heatmap_axes)
    row_labels = [t.get_text() for t in axes.heatmap_axes["Score"].get_yticklabels()]
    assert row_labels == [str(i) for i in axes.row_order]
    assert len(axes.heatmap_axes["Score"].texts) == cluster_summary.height  # annotated
    assert axes.feature_order["Class proportion"] == ["prop_synonymous", "prop_frameshift"]
    assert (tmp_path / "cm.png").exists()


def test_cluster_features_and_display_names(cluster_summary):
    groups = [
        FeatureGroup("Landmarks", [f"landmark_{i}" for i in range(4)], cluster_features=True,
                     labels={"landmark_0": "Nuclear size"}),
        GROUPS[2],
    ]
    _, axes = fb.ClusterMap(cluster_summary, groups=groups).plot()
    assert sorted(axes.feature_order["Landmarks"]) == [f"landmark_{i}" for i in range(4)]
    labels = [t.get_text() for t in axes.heatmap_axes["Landmarks"].get_xticklabels()]
    assert "Nuclear size" in labels


def test_display_only_group_keeps_missing_values(cluster_summary, caplog):
    df = cluster_summary.with_columns(
        pl.when(pl.col("meta_cluster_idx") == "0").then(None).otherwise(pl.col("median_score"))
        .alias("median_score"),
        pl.lit(None, dtype=pl.Float64).alias("landmark_bad"),
    )
    groups = [
        FeatureGroup("Landmarks", [f"landmark_{i}" for i in range(4)] + ["landmark_bad"]),
        GROUPS[2],
    ]
    plot = fb.ClusterMap(df, groups=groups)
    assert "landmark_bad" not in plot.features and "median_score" in plot.features
    assert "Dropping 1" in caplog.text
    plot.plot()


def test_group_validation(cluster_summary):
    with pytest.raises(ValueError, match="both groups"):
        fb.ClusterMap(cluster_summary, groups=[
            FeatureGroup("a", ["landmark_0"]), FeatureGroup("b", ["landmark_0", "landmark_1"])])
    with pytest.raises(ValueError, match="unique"):
        fb.ClusterMap(cluster_summary, groups=[
            FeatureGroup("a", ["landmark_0"]), FeatureGroup("a", ["landmark_1"])])
    with pytest.raises(ValueError, match="landmark_9"):
        fb.ClusterMap(cluster_summary, groups=[FeatureGroup("a", ["landmark_9"])])
    with pytest.raises(ValueError, match="not both"):
        fb.ClusterMap(cluster_summary, groups=GROUPS, features=["landmark_0"])


def test_default_single_group(profiles):
    df = profiles.with_columns(pl.lit(None, dtype=pl.Float64).alias("feature_bad"))
    plot = fb.ClusterMap(df, row_colors="meta_variant_type")
    assert plot.features == [f"feature_{i}" for i in range(8)]
    fig, axes = plot.refline(x=2).plot()
    assert fig.legends and axes.row_colors_ax is not None
    assert sorted(axes.feature_order[""]) == plot.features  # features clustered by default
    with pytest.raises(TypeError, match="figure-level"):
        plot.plot(ax=axes.heatmap_axes[""])
