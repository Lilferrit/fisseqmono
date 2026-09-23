import polars as pl
import pytest
from matplotlib.collections import PolyCollection

import fisseqborn as fb
from fisseqborn import fisseq

UMAP = {"x": "meta_notebook_umap_1", "y": "meta_notebook_umap_2"}


# ----- BoxPlot ------------------------------------------------------------------------


def test_boxplot_counts_and_fisseq_palette(profiles):
    plot = fb.BoxPlot(profiles, x="meta_variant_type", y="meta_impact_score")
    _, ax = plot.plot()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    n_syn = profiles.filter(pl.col("meta_variant_type") == "Synonymous").height
    assert labels[0] == f"Synonymous\n(n = {n_syn})"
    assert [lbl.split("\n")[0] for lbl in labels] == ["Synonymous", "Single Missense", "Frameshift"]
    assert plot.palette["Synonymous"] == "darkgreen"


def test_annotate_pairs_uses_plot_y(profiles, monkeypatch):
    import statannotations.Annotator as mod

    seen = {}
    original = mod.Annotator.__init__

    def spy(self, ax, pairs, **kw):
        seen.update(kw, pairs=pairs)
        original(self, ax, pairs, **kw)

    monkeypatch.setattr(mod.Annotator, "__init__", spy)
    _, ax = (
        fb.BoxPlot(profiles, x="meta_variant_type", y="meta_distinguishability_score")
        .annotate_pairs()
        .plot()
    )
    assert seen["y"] == "meta_distinguishability_score"
    assert len(seen["pairs"]) == 3
    assert any(t.get_text() for t in ax.texts)  # star annotations were drawn


def test_annotate_pairs_with_hue_skips_empty_groups(profiles):
    df = profiles.filter(
        ~((pl.col("meta_experiment") == "T2_R1") & (pl.col("meta_variant_type") == "Frameshift"))
    )
    plot = fb.BoxPlot(df, x="meta_experiment", y="meta_impact_score", hue="meta_variant_type")
    pairs = plot._resolve_pairs("all")
    assert len(pairs) == 4 * 3 - 2
    assert all(p[0][0] == p[1][0] for p in pairs)
    plot.annotate_pairs().plot()


@pytest.mark.parametrize("points", ["strip", "density"])
def test_boxplot_points(profiles, points):
    _, ax = fb.BoxPlot(
        profiles, x="meta_variant_type", y="meta_impact_score", hue="meta_experiment", points=points
    ).plot()
    assert ax.get_legend() is not None
    assert sum(len(c.get_offsets()) for c in ax.collections) == profiles.height


# ----- EmbeddingPlot -------------------------------------------------------------------


def test_embedding_categorical_scatter_with_highlights(profiles):
    _, ax = (
        fb.EmbeddingPlot(profiles, **UMAP, hue="meta_variant_type")
        .highlight(pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC, label=fisseq.PATHOGENIC, marker="^")
        .plot()
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Synonymous", "Single Missense", "Frameshift", fisseq.PATHOGENIC]
    assert ax.get_xlabel() == UMAP["x"]


def test_embedding_numeric_scatter_has_colorbar(profiles):
    fig, _ = fb.EmbeddingPlot(profiles, **UMAP, hue="meta_distinguishability_score", cmap="Reds",
                              vmin=0.5, vmax=1).plot()
    assert len(fig.axes) == 2  # main + colorbar


def test_embedding_hexbin_categorical_mode(profiles):
    fig, ax = fb.EmbeddingPlot(profiles, **UMAP, hue="meta_cluster_idx", kind="hexbin").plot()
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    cbar_labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert cbar_labels == [str(i) for i in range(12)]


def test_embedding_hexbin_centered(profiles):
    plot = fb.EmbeddingPlot(profiles, **UMAP, hue="feature_3", kind="hexbin", cmap="RdBu_r", center=0, clip=1)
    norm, extend = plot._norm(profiles.get_column("feature_3").to_numpy())
    assert (norm.vmin, norm.vmax) == (-1, 1)
    assert extend == "both"
    plot.plot()


# ----- CorrelationPlot -----------------------------------------------------------------


def test_correlation_identity_and_stat(profiles):
    plot = fb.CorrelationPlot(
        profiles, x="meta_distinguishability_score", y="meta_distinguishability_score_rep2",
        hue="meta_variant_type", identity=True, lims=(0, 1), count_sides=True,
    )
    _, ax = plot.plot()
    texts = [t.get_text() for t in ax.texts]
    r, _, n = plot.correlation()
    assert r > 0.8 and n == profiles.height
    assert any(t.startswith("Pearson r = ") for t in texts)
    assert sum(t.startswith("n = ") for t in texts) == 2


def test_correlation_kde_lowess_spearman(profiles):
    _, ax = fb.CorrelationPlot(
        profiles, x="meta_num_cells", y="meta_distinguishability_score",
        kind="kde", fit="lowess", stat="spearman",
    ).plot()
    assert any("Spearman" in t.get_text() for t in ax.texts)


# ----- RocPlot -------------------------------------------------------------------------


def test_roc_multiple_scores(profiles):
    plot = fb.RocPlot(
        profiles, label="meta_variant_type", positive="Single Missense",
        score=["meta_distinguishability_score", "meta_impact_score"],
        names={"meta_impact_score": "impact"},
    )
    _, ax = plot.plot()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 2 and all("AUC = " in lbl for lbl in labels)
    assert labels[1].startswith("impact")
    aucs = plot.aucs()
    assert aucs.height == 2 and (aucs["auc"] > 0.5).all()


def test_roc_group_combined(profiles):
    plot = fb.RocPlot(
        profiles, label="meta_variant_type", positive=["Single Missense", "Frameshift"],
        score="meta_distinguishability_score", group="meta_experiment", combined=True,
    )
    assert plot.aucs()["curve"].to_list() == ["T1_R1", "T1_R2", "T2_R1", "T10_R1", "Combined"]
    plot.plot()


def test_roc_rejects_scores_and_group(profiles):
    with pytest.raises(ValueError, match="not both"):
        fb.RocPlot(profiles, label="meta_variant_type", positive="Frameshift",
                   score=["meta_impact_score", "meta_distinguishability_score"], group="meta_experiment")


# ----- Heatmap / ClusterMap -------------------------------------------------------------


def test_heatmap_long_form_symmetric():
    pairs = pl.DataFrame({
        "barcode_a": ["b1", "b1", "b2"],
        "barcode_b": ["b2", "b3", "b3"],
        "test_auroc": [0.6, 0.7, 0.8],
    })
    plot = fb.Heatmap(pairs, index="barcode_a", columns="barcode_b", values="test_auroc",
                      symmetric=True, fill_value=0.5, cmap="Reds", vmin=0.5, vmax=1)
    mat = plot.matrix()
    assert list(mat.index) == ["b1", "b2", "b3"] == list(mat.columns)
    assert mat.loc["b3", "b1"] == 0.7 and mat.loc["b1", "b1"] == 0.5
    plot.plot()


def test_heatmap_correlation(profiles):
    plot = fb.Heatmap.correlation(profiles, ["feature_0", "feature_1", "feature_2"], method="spearman")
    mat = plot.matrix()
    assert mat.shape == (3, 3)
    assert mat.loc["feature_1", "feature_1"] == pytest.approx(1.0)
    plot.plot()



def test_embedding_draws_largest_group_first(profiles):
    _, ax = fb.EmbeddingPlot(profiles, **UMAP, hue="meta_variant_type").plot()
    colors = ax.collections[0].get_facecolors()
    # Single Missense (grey, the largest class) is drawn first, Frameshift (purple) last
    assert tuple(colors[0][:3]) == pytest.approx((0.5, 0.5, 0.5), abs=0.01)
    assert tuple(colors[-1][:3]) == pytest.approx((0.5, 0.0, 0.5), abs=0.01)
