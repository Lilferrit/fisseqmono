import matplotlib as mpl
import numpy as np
import polars as pl
import pytest
from matplotlib.colors import to_rgb
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

import fisseqborn as fb
from fisseqborn import FeatureGroup, fisseq


@pytest.fixture
def cluster_summary() -> pl.DataFrame:
    """One row per cluster: two structured feature blocks plus a display-only score."""
    rng = np.random.default_rng(0)
    n = 12
    blocks = np.repeat([[-2.0], [0.0], [2.0]], 4, axis=0)  # three row blocks
    return pl.DataFrame(
        {
            "meta_cluster_idx": [str(i) for i in range(n)],
            **{
                f"landmark_{i}": (blocks[:, 0] + rng.normal(0, 0.1, n))
                for i in range(4)
            },
            "prop_synonymous": rng.random(n),
            "prop_frameshift": rng.random(n),
            # large-scale noise that would dominate unscaled distances if it were clustered
            "median_score": rng.normal(0, 100, n),
        }
    )


GROUPS = [
    FeatureGroup(
        "Landmarks", [f"landmark_{i}" for i in range(4)], cmap="RdBu_r", center=0
    ),
    FeatureGroup(
        "Class proportion",
        ["prop_synonymous", "prop_frameshift"],
        cmap="Greens",
        vmin=0,
    ),
    FeatureGroup("Score", "median_score", cluster=False, cmap="Reds", annot=True),
]


def _expected_order(
    df: pl.DataFrame, cols: list[str], standardize: bool = False
) -> list[int]:
    x = df.select(cols).to_numpy()
    if standardize:
        x = (x - x.mean(0)) / x.std(0)
    return leaves_list(linkage(pdist(x), method="average")).tolist()


def test_row_order_uses_only_clustering_groups(cluster_summary):
    plot = fb.ClusterMap(cluster_summary, groups=GROUPS, row_labels="meta_cluster_idx")
    clustered = [f"landmark_{i}" for i in range(4)] + [
        "prop_synonymous",
        "prop_frameshift",
    ]
    assert plot.row_order() == _expected_order(cluster_summary, clustered)
    with_score = _expected_order(cluster_summary, clustered + ["median_score"])
    assert plot.row_order() != with_score


def test_standardize(cluster_summary):
    plot = fb.ClusterMap(cluster_summary, groups=GROUPS, standardize=True)
    clustered = [f"landmark_{i}" for i in range(4)] + [
        "prop_synonymous",
        "prop_frameshift",
    ]
    assert plot.row_order() == _expected_order(
        cluster_summary, clustered, standardize=True
    )


def test_no_clustering_group_keeps_input_order(cluster_summary):
    groups = [FeatureGroup(g.name, g.features, cluster=False) for g in GROUPS]
    plot = fb.ClusterMap(cluster_summary, groups=groups)
    fig, axes = plot.plot()
    assert axes.row_order == list(range(cluster_summary.height))
    assert axes.row_dendrogram_ax is None and axes.row_linkage is None


def test_draws_one_block_per_group(cluster_summary, tmp_path):
    fig, axes = (
        fb.ClusterMap(
            cluster_summary,
            groups=GROUPS,
            row_labels="meta_cluster_idx",
            title="Cluster summary",
        )
        .save(tmp_path / "cm.png")
        .plot()
    )
    assert list(axes.heatmap_axes) == ["Landmarks", "Class proportion", "Score"]
    assert list(axes.colorbar_axes) == list(axes.heatmap_axes)
    titles = [
        ax.texts[0].get_text().replace("\n", " ") for ax in axes.title_axes.values()
    ]
    assert titles == list(axes.heatmap_axes)
    row_labels = [t.get_text() for t in axes.heatmap_axes["Score"].get_yticklabels()]
    assert row_labels == [str(i) for i in axes.row_order]
    assert len(axes.heatmap_axes["Score"].texts) == cluster_summary.height  # annotated
    assert axes.feature_order["Class proportion"] == [
        "prop_synonymous",
        "prop_frameshift",
    ]
    assert (tmp_path / "cm.png").exists()


def test_cluster_features_and_display_names(cluster_summary):
    groups = [
        FeatureGroup(
            "Landmarks",
            [f"landmark_{i}" for i in range(4)],
            cluster_features=True,
            labels={"landmark_0": "Nuclear size"},
        ),
        GROUPS[2],
    ]
    _, axes = fb.ClusterMap(cluster_summary, groups=groups).plot()
    assert sorted(axes.feature_order["Landmarks"]) == [
        f"landmark_{i}" for i in range(4)
    ]
    labels = [t.get_text() for t in axes.heatmap_axes["Landmarks"].get_xticklabels()]
    assert "Nuclear size" in labels


def test_display_only_group_keeps_missing_values(cluster_summary, caplog):
    df = cluster_summary.with_columns(
        pl.when(pl.col("meta_cluster_idx") == "0")
        .then(None)
        .otherwise(pl.col("median_score"))
        .alias("median_score"),
        pl.lit(None, dtype=pl.Float64).alias("landmark_bad"),
    )
    groups = [
        FeatureGroup(
            "Landmarks", [f"landmark_{i}" for i in range(4)] + ["landmark_bad"]
        ),
        GROUPS[2],
    ]
    plot = fb.ClusterMap(df, groups=groups)
    assert "landmark_bad" not in plot.features and "median_score" in plot.features
    assert "Dropping 1" in caplog.text
    plot.plot()


def test_group_validation(cluster_summary):
    with pytest.raises(ValueError, match="both groups"):
        fb.ClusterMap(
            cluster_summary,
            groups=[
                FeatureGroup("a", ["landmark_0"]),
                FeatureGroup("b", ["landmark_0", "landmark_1"]),
            ],
        )
    with pytest.raises(ValueError, match="unique"):
        fb.ClusterMap(
            cluster_summary,
            groups=[
                FeatureGroup("a", ["landmark_0"]),
                FeatureGroup("a", ["landmark_1"]),
            ],
        )
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
    assert (
        sorted(axes.feature_order[""]) == plot.features
    )  # features clustered by default
    with pytest.raises(TypeError, match="figure-level"):
        plot.plot(ax=axes.heatmap_axes[""])


NOTEBOOK_GROUPS = [
    FeatureGroup(
        "Landmark z-score (vs. synonymous)",
        [f"landmark_{i}" for i in range(4)],
        cmap="RdBu_r",
        center=0,
        clip=3,
    ),
    FeatureGroup("Share of domain", ["prop_synonymous"], cmap="Blues", vmin=0),
    FeatureGroup(
        "Share of class", ["prop_frameshift"], cmap="Greens", vmin=0, annot=True
    ),
    FeatureGroup(
        "Median distinguishability",
        "median_score",
        cluster=False,
        cmap="Reds",
        annot=True,
    ),
]


def _assert_titles_do_not_overlap(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.texts[0].get_window_extent(renderer) for ax in axes.title_axes.values()]
    assert len(boxes) == len(axes.heatmap_axes)
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not a.overlaps(b), (a, b)
    # and each title stays within its own block's extent (plus the gap next to it)
    for (name, tax), box in zip(axes.title_axes.items(), boxes):
        block = axes.heatmap_axes[name].get_window_extent(renderer)
        slack = 0.25 * fig.dpi
        assert box.x0 >= block.x0 - slack and box.x1 <= block.x1 + slack


def test_narrow_group_titles_do_not_overlap(cluster_summary):
    fig, axes = fb.ClusterMap(
        cluster_summary,
        groups=NOTEBOOK_GROUPS,
        row_labels="meta_cluster_idx",
        standardize=True,
    ).plot()
    _assert_titles_do_not_overlap(fig, axes)
    wrapped = axes.title_axes["Median distinguishability"].texts[0].get_text()
    assert wrapped.split() == [
        "Median",
        "distinguishability",
    ]  # wrapped between words only
    assert "\n" in wrapped


def test_vertical_orientation_transposes(cluster_summary):
    kw = {
        "groups": GROUPS,
        "row_labels": "meta_cluster_idx",
        "row_colors": "meta_cluster_idx",
        "standardize": True,
    }
    horizontal = fb.ClusterMap(cluster_summary, **kw)
    vertical = fb.ClusterMap(cluster_summary, orientation="vertical", **kw)
    assert vertical.row_order() == horizontal.row_order()

    _, axes = vertical.plot()
    assert axes.orientation == "vertical"
    order = axes.row_order
    landmarks = axes.heatmap_axes["Landmarks"]
    image = landmarks.images[0].get_array()
    assert image.shape[:2] == (4, cluster_summary.height)  # features x DataFrame rows
    _, h_axes = horizontal.plot()
    np.testing.assert_allclose(
        image, h_axes.heatmap_axes["Landmarks"].images[0].get_array().transpose(1, 0, 2)
    )

    # row labels under the bottom group only, in plotting order; features labelled on the right
    bottom = axes.heatmap_axes["Score"]
    assert [t.get_text() for t in bottom.get_xticklabels()] == [str(i) for i in order]
    assert not landmarks.get_xticklabels()
    assert [t.get_text() for t in landmarks.get_yticklabels()] == axes.feature_order[
        "Landmarks"
    ]
    assert len(bottom.texts) == cluster_summary.height  # annotated

    # dendrogram above, row colors between it and the heatmaps, all sharing the x extent
    dendro = axes.row_dendrogram_ax.get_position()
    colors = axes.row_colors_ax.get_position()
    heat = landmarks.get_position()
    assert dendro.y0 >= colors.y1 - 1e-9 and colors.y0 >= heat.y1 - 1e-9
    assert dendro.x0 == pytest.approx(heat.x0) and dendro.x1 == pytest.approx(heat.x1)
    assert axes.row_colors_ax.images[0].get_array().shape[:2] == (
        1,
        cluster_summary.height,
    )
    # colorbars vertical, left of their own group
    cbar = axes.colorbar_axes["Landmarks"].get_position()
    assert (
        cbar.x1 <= heat.x0 and cbar.y0 >= heat.y0 - 1e-9 and cbar.y1 <= heat.y1 + 1e-9
    )


def test_vertical_titles_do_not_overlap(cluster_summary):
    fig, axes = fb.ClusterMap(
        cluster_summary,
        groups=NOTEBOOK_GROUPS,
        row_labels="meta_cluster_idx",
        orientation="vertical",
    ).plot()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.texts[0].get_window_extent(renderer) for ax in axes.title_axes.values()]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            assert not a.overlaps(b), (a, b)
    for name, box in zip(axes.title_axes, boxes):
        cbar = axes.colorbar_axes[name].get_window_extent(renderer)
        assert box.x1 <= cbar.x0  # titles stay left of the colorbars


def test_orientation_validation(cluster_summary):
    with pytest.raises(ValueError, match="orientation"):
        fb.ClusterMap(cluster_summary, groups=GROUPS, orientation="diagonal")


def _cell_rgb(axes, group, orientation, row, feature):
    """Displayed color of DataFrame row position ``row`` (in plotting order) x feature."""
    image = axes.heatmap_axes[group].images[0].get_array()
    j = axes.feature_order[group].index(feature)
    return image[j, row, :3] if orientation == "vertical" else image[row, j, :3]


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_per_feature_palette(cluster_summary, orientation):
    palette = {"prop_synonymous": "darkgreen", "prop_frameshift": "purple"}
    group = FeatureGroup(
        "Class",
        ["prop_synonymous", "prop_frameshift"],
        palette=palette,
        vmin=0,
        annot=True,
    )
    assert group.per_feature and not GROUPS[0].per_feature
    _, axes = fb.ClusterMap(
        cluster_summary, groups=[GROUPS[0], group], orientation=orientation
    ).plot()
    assert "Class" not in axes.colorbar_axes
    assert list(axes.feature_colorbar_axes["Class"]) == [
        "prop_synonymous",
        "prop_frameshift",
    ]

    values = cluster_summary.select(group.features).to_numpy()[axes.row_order]
    for j, f in enumerate(group.features):
        top = int(values[:, j].argmax())
        # the feature's max is its full palette color; the scalar vmin=0 applies to each feature
        np.testing.assert_allclose(
            _cell_rgb(axes, "Class", orientation, top, f), to_rgb(palette[f]), atol=1e-6
        )
        cbar = axes.feature_colorbar_axes["Class"][f]
        lo, hi = cbar.get_ylim() if orientation == "vertical" else cbar.get_xlim()
        assert lo == pytest.approx(0) and hi == pytest.approx(values[:, j].max())

    # annotations stay readable: white text on the darkest cells
    texts = axes.heatmap_axes["Class"].texts
    assert {t.get_color() for t in texts} >= {"white", "black"}


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_feature_colorbars_align_with_their_cells(cluster_summary, orientation):
    group = FeatureGroup("Class", ["prop_synonymous", "prop_frameshift"], palette=True)
    fig, axes = fb.ClusterMap(
        cluster_summary,
        groups=[GROUPS[0], group],
        orientation=orientation,
        row_labels="meta_cluster_idx",
    ).plot()
    fig.canvas.draw()
    heat = axes.heatmap_axes["Class"].get_window_extent()
    n = len(group.features)
    for j, f in enumerate(axes.feature_order["Class"]):
        box = axes.feature_colorbar_axes["Class"][f].get_window_extent()
        if orientation == "vertical":
            cell_center = heat.y1 - (j + 0.5) * heat.height / n
            assert (box.y0 + box.y1) / 2 == pytest.approx(cell_center, abs=1)
            assert box.x1 <= heat.x0
        else:
            cell_center = heat.x0 + (j + 0.5) * heat.width / n
            assert (box.x0 + box.x1) / 2 == pytest.approx(cell_center, abs=1)
            assert box.y0 >= heat.y1


def test_per_feature_cmap_and_limits(cluster_summary):
    feats = [f"landmark_{i}" for i in range(4)]
    group = FeatureGroup(
        "Landmarks",
        feats,
        cmap={"landmark_0": "Greys"},
        center=0,
        vmax={"landmark_1": 10.0},
    )
    assert group.per_feature
    _, axes = fb.ClusterMap(cluster_summary, groups=[group]).plot()
    bars = axes.feature_colorbar_axes["Landmarks"]
    assert bars["landmark_1"].get_xlim()[1] == pytest.approx(10.0)
    values = cluster_summary["landmark_2"].to_numpy()
    half = np.abs(values).max()  # centered, no clip: symmetric around 0
    assert bars["landmark_2"].get_xlim() == pytest.approx((-half, half))
    # landmark_0 uses its own Greys map, so its largest value is drawn black
    top = axes.row_order.index(int(cluster_summary["landmark_0"].to_numpy().argmax()))
    np.testing.assert_allclose(
        _cell_rgb(axes, "Landmarks", "horizontal", top, "landmark_0"),
        mpl.colormaps["Greys"](1.0)[:3],
        atol=1e-6,
    )


def test_palette_true_uses_fisseq_colors():
    df = pl.DataFrame(
        {
            "meta_cluster_idx": ["0", "1", "2"],
            "Synonymous": [0.1, 0.5, 0.4],
            fisseq.PATHOGENIC: [0.6, 0.2, 0.2],
        }
    )
    group = FeatureGroup(
        "Share of class", ["Synonymous", fisseq.PATHOGENIC], palette=True
    )
    _, axes = fb.ClusterMap(df, groups=[group]).plot()
    order = axes.row_order
    for f in group.features:
        top = order.index(int(df[f].to_numpy().argmax()))
        np.testing.assert_allclose(
            _cell_rgb(axes, "Share of class", "horizontal", top, f),
            to_rgb(fisseq.PALETTE[f]),
            atol=1e-6,
        )
