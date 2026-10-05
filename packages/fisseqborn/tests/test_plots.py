import numpy as np
import polars as pl
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba

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


@pytest.mark.parametrize("method", ["pearson", "spearman", "cosine"])
def test_heatmap_correlation_dense_matches_complete_rows(profiles, method):
    cols = ["feature_0", "feature_1", "feature_2", "feature_3"]
    clean = profiles.select(cols).cast(pl.Float64)
    if method == "spearman":
        clean = clean.select(pl.all().rank())
    mat = clean.to_numpy()
    if method == "cosine":
        unit = mat / np.linalg.norm(mat, axis=0, keepdims=True)
        expected = unit.T @ unit
    else:
        expected = np.corrcoef(mat, rowvar=False)
    plot = fb.Heatmap.correlation(profiles, cols, method=method)
    assert np.allclose(plot.matrix().to_numpy(), expected)
    assert (plot.n_shared.to_numpy() == profiles.height).all()


@pytest.fixture
def block_sparse() -> pl.DataFrame:
    """Columns a1/a2 are scored on rows 0-19, b1/b2 on rows 20-39, c1 on rows 0-1 only."""
    rng = np.random.default_rng(3)
    base = rng.normal(size=40)

    def col(rows):
        out = np.full(40, None, dtype=object)
        out[rows] = base[rows] + rng.normal(0, 0.3, 40)[rows]
        return pl.Series(out.tolist(), dtype=pl.Float64)

    a, b = slice(0, 20), slice(20, 40)
    return pl.DataFrame({"a1": col(a), "a2": col(a), "b1": col(b), "b2": col(b),
                         "c1": col(slice(0, 2))})


def test_heatmap_correlation_block_sparse(block_sparse):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plot = fb.Heatmap.correlation(block_sparse, method="spearman", nan_color="black")
        mat = plot.matrix()
    for i, j in [("a1", "a2"), ("b1", "b2"), ("a1", "a1")]:
        assert np.isfinite(mat.loc[i, j]) and mat.loc[i, j] == mat.loc[j, i]
    assert np.isnan(mat.loc["a1", "b2"]) and np.isnan(mat.loc["b1", "a2"])
    # c1 shares only 2 rows with a1 (and with itself): below min_shared=10.
    assert plot.n_shared.loc["a1", "c1"] == 2 and plot.n_shared.loc["a1", "b1"] == 0
    assert mat["c1"].isna().all()
    assert np.isfinite(
        fb.Heatmap.correlation(block_sparse, min_shared=2).matrix().loc["a1", "c1"]
    )
    plot.plot()


def test_heatmap_correlation_spearman_matches_scipy_on_shared_rows(block_sparse):
    from scipy.stats import spearmanr

    df = block_sparse.with_columns(pl.col("a2").fill_null(pl.col("b1")))  # a2 spans both blocks
    shared = df.select("a2", "b2").drop_nulls()
    expected = spearmanr(shared["a2"], shared["b2"]).statistic
    mat = fb.Heatmap.correlation(df, ["a2", "b2"], method="spearman").matrix()
    assert mat.loc["a2", "b2"] == pytest.approx(expected)


@pytest.fixture
def replicate_scores() -> pl.DataFrame:
    """T1_R1 and T1_R2 share variants v0-v9; T2_R1 has disjoint variants."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1, 10)
    rows = (
        [("T1_R1", f"v{i}", s) for i, s in enumerate(base)]
        + [("T1_R2", f"v{i}", s + rng.normal(0, 0.1)) for i, s in enumerate(base)]
        + [("T2_R1", f"w{i}", rng.uniform()) for i in range(10)]
    )
    return pl.DataFrame(rows, schema=["experiment", "variant", "test_auroc"], orient="row")


def test_batch_correlation_matrix(replicate_scores):
    from scipy.stats import spearmanr

    plot = fb.BatchCorrelationHeatmap(replicate_scores, batch="experiment", label="variant",
                                      score="test_auroc")
    pairs = plot.pairs().filter(pl.col("batch_a") != pl.col("batch_b"))
    t1 = pairs.filter((pl.col("batch_a") == "T1_R1") & (pl.col("batch_b") == "T1_R2"))
    r1 = replicate_scores.filter(pl.col("experiment") == "T1_R1").sort("variant")
    r2 = replicate_scores.filter(pl.col("experiment") == "T1_R2").sort("variant")
    assert t1["r"][0] == pytest.approx(spearmanr(r1["test_auroc"], r2["test_auroc"]).statistic)
    assert t1["n_shared"][0] == 10

    mat = plot.matrix()
    assert list(mat.index) == ["T1_R1", "T1_R2", "T2_R1"] == list(mat.columns)
    assert np.allclose(np.diag(mat), 1.0)
    assert np.isnan(mat.loc["T1_R1", "T2_R1"]) and np.isnan(mat.loc["T2_R1", "T1_R2"])
    assert mat.loc["T1_R2", "T1_R1"] == mat.loc["T1_R1", "T1_R2"]

    _, ax = plot.plot()
    assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)
    assert len(ax.texts) == int(np.isfinite(mat.to_numpy()).sum())


def test_batch_correlation_duplicates_and_min_shared(replicate_scores):
    dup = pl.concat([replicate_scores, replicate_scores.head(1)])
    with pytest.raises(ValueError, match="Duplicate"):
        fb.BatchCorrelationHeatmap(dup, batch="experiment", label="variant", score="test_auroc")
    fb.BatchCorrelationHeatmap(dup, batch="experiment", label="variant", score="test_auroc",
                               aggregate="mean")

    thin = replicate_scores.filter(
        (pl.col("experiment") != "T1_R2") | pl.col("variant").is_in(["v0", "v1"])
    )
    mat = fb.BatchCorrelationHeatmap(thin, batch="experiment", label="variant",
                                     score="test_auroc").matrix()
    assert np.isnan(mat.loc["T1_R1", "T1_R2"]) and np.isnan(mat.loc["T1_R2", "T1_R2"])
    assert mat.loc["T1_R1", "T1_R1"] == 1.0


def test_embedding_draws_largest_group_first(profiles):
    _, ax = fb.EmbeddingPlot(profiles, **UMAP, hue="meta_variant_type").plot()
    colors = ax.collections[0].get_facecolors()
    # Single Missense (grey, the largest class) is drawn first, Frameshift (purple) last
    assert tuple(colors[0][:3]) == pytest.approx((0.5, 0.5, 0.5), abs=0.01)
    assert tuple(colors[-1][:3]) == pytest.approx((0.5, 0.0, 0.5), abs=0.01)


# ----- VolcanoPlot ---------------------------------------------------------------------


@pytest.fixture
def volcano_rows(profiles) -> pl.DataFrame:
    """Long-form (variant, feature) rows: one median value and -log10 p per pair."""
    return (
        profiles.unpivot(
            on=[f"feature_{i}" for i in range(8)],
            index=["meta_aa_changes", "meta_variant_type"],
            variable_name="feature",
            value_name="median",
        )
        .with_columns((pl.col("median").abs() * 3).alias("negLogP"))
    )


def test_volcano_layers_draw_in_call_order(volcano_rows):
    types = ["Single Missense", "Frameshift", "Synonymous"]
    plot = fb.VolcanoPlot(volcano_rows, x="median", y="negLogP")
    for t in types:
        plot = plot.layer(pl.col("meta_variant_type") == t, label=t)
    _, ax = plot.plot()
    counts = volcano_rows.get_column("meta_variant_type").value_counts(name="n")
    n = dict(counts.iter_rows())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == [f"{t} (n={n[t]:,})" for t in types] + ["Bonferroni p = 0.05"]
    # layers stack bottom to top in call order, colored from the fisseq palette
    assert [c.get_zorder() for c in ax.collections] == [1, 2, 3]
    assert tuple(ax.collections[2].get_facecolors()[0][:3]) == pytest.approx(
        (0.0, 100 / 255, 0.0), abs=0.01
    )


def test_volcano_bonferroni_threshold_counts_drawn_points(volcano_rows):
    df = volcano_rows.with_columns(
        pl.when(pl.col("feature") == "feature_0").then(None).otherwise(pl.col("median"))
        .alias("median")
    )
    plot = fb.VolcanoPlot(df, x="median", y="negLogP").layer(
        pl.col("meta_variant_type") != "Frameshift"
    )
    n = df.filter(
        pl.col("median").is_not_null(), pl.col("meta_variant_type") != "Frameshift"
    ).height
    assert plot.significance_threshold() == pytest.approx(-np.log10(0.05 / n))
    _, ax = plot.plot()
    assert ax.lines[0].get_ydata()[0] == pytest.approx(plot.significance_threshold())
    # the unlabelled layer is left out of the legend
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Bonferroni p = 0.05"]


def test_volcano_defaults_and_options(volcano_rows):
    _, ax = fb.VolcanoPlot(volcano_rows, x="median", y="negLogP", alpha=None,
                           x_quantiles=None).plot()
    assert len(ax.collections) == 1 and not ax.lines and ax.get_legend() is None
    assert len(ax.collections[0].get_offsets()) == volcano_rows.height
    assert fb.VolcanoPlot(volcano_rows, x="median", y="negLogP", bonferroni=False) \
        .significance_threshold() == pytest.approx(-np.log10(0.05))
    lo, hi = fb.VolcanoPlot(volcano_rows, x="median", y="negLogP").plot()[1].get_xlim()
    assert lo > volcano_rows["median"].min() and hi < volcano_rows["median"].max()


def test_volcano_from_wide_matches_long_form(profiles):
    wide = profiles.select(
        "meta_aa_changes",
        "meta_variant_type",
        *[pl.col(f"feature_{i}").alias(f"feature_{i}_median") for i in range(3)],
        *[(pl.col(f"feature_{i}").abs() * 3).alias(f"feature_{i}_KSnegLogP") for i in range(3)],
        pl.col("feature_3").alias("feature_3_median"),  # no p-value column: skipped
    )
    plot = fb.VolcanoPlot.from_wide(wide, alpha=None)
    assert (plot.x, plot.y) == ("median", "KSnegLogP")
    assert set(plot.data.columns) == {
        "meta_aa_changes", "meta_variant_type", "feature", "median", "KSnegLogP"
    }
    assert plot.data.height == 3 * profiles.height
    row = plot.data.filter(
        pl.col("meta_aa_changes") == "A5V", pl.col("feature") == "feature_2"
    )
    expected = profiles.filter(pl.col("meta_aa_changes") == "A5V")["feature_2"][0]
    assert row["median"][0] == pytest.approx(expected)
    assert row["KSnegLogP"][0] == pytest.approx(abs(expected) * 3)
    with pytest.raises(ValueError, match="No feature"):
        fb.VolcanoPlot.from_wide(wide, y_suffix="_AUROCnegLogP")


def test_heatmap_correlation_squared(profiles):
    cols = ["feature_0", "feature_1", "feature_2"]
    r = fb.Heatmap.correlation(profiles, cols).matrix()
    plot = fb.Heatmap.correlation(profiles, cols, squared=True)
    assert np.allclose(plot.matrix().to_numpy(), r.to_numpy() ** 2)
    assert plot.heatmap_kw["vmin"] == 0 and plot.heatmap_kw["cmap"] == "Blues"


@pytest.mark.parametrize("method", ["pearson", "spearman", "cosine"])
def test_batch_correlation_methods_and_squared(replicate_scores, method):
    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc", "method": method}
    plain = fb.BatchCorrelationHeatmap(replicate_scores, **kw)
    squared = fb.BatchCorrelationHeatmap(replicate_scores, squared=True, **kw)
    expected = fb.Heatmap.correlation(
        replicate_scores.pivot(on="experiment", index="variant", values="test_auroc"),
        ["T1_R1", "T1_R2", "T2_R1"], method=method,
    ).matrix()
    assert np.allclose(plain.matrix().to_numpy(), expected.to_numpy(), equal_nan=True)
    assert np.allclose(squared.matrix().to_numpy(), expected.to_numpy() ** 2, equal_nan=True)
    pairs = squared.pairs().drop_nans("r")
    assert np.allclose(pairs["r_squared"], pairs["r"] ** 2)
    squared.plot()


def test_batch_correlation_rejects_unknown_method(replicate_scores):
    with pytest.raises(ValueError, match="cosine"):
        fb.BatchCorrelationHeatmap(replicate_scores, batch="experiment", label="variant",
                                   score="test_auroc", method="kendall")


def _cbar_label(ax):
    return ax.collections[0].colorbar.ax.get_ylabel()


@pytest.mark.parametrize(("method", "squared", "label"), [
    ("pearson", False, "Pearson r"), ("spearman", True, "Spearman ρ²"),
    ("cosine", False, "Cosine similarity"),
])
def test_correlation_heatmaps_label_colorbar(profiles, replicate_scores, method, squared, label):
    _, ax = fb.Heatmap.correlation(profiles, ["feature_0", "feature_1"], method=method,
                                   squared=squared).plot()
    assert _cbar_label(ax) == label
    _, ax = fb.BatchCorrelationHeatmap(replicate_scores, batch="experiment", label="variant",
                                       score="test_auroc", method=method,
                                       squared=squared).plot()
    assert _cbar_label(ax) == label


def test_correlation_colorbar_label_can_be_overridden(profiles):
    _, ax = fb.Heatmap.correlation(profiles, ["feature_0", "feature_1"],
                                   cbar_kws={"label": "custom", "shrink": 0.5}).plot()
    assert _cbar_label(ax) == "custom"


def _last_scatter(ax):
    return ax.collections[-1]


def test_highlight_hue_uses_base_palette(profiles):
    # old cluster ids stored as integers still pick up the base plot's string-keyed colors
    df = profiles.with_columns(
        pl.when(pl.col("meta_variant_type") == "Synonymous")
        .then(pl.col("meta_cluster_idx").cast(pl.Int32))
        .alias("meta_old_cluster_idx")
    )
    base = fb.EmbeddingPlot(
        df, x="meta_notebook_umap_1", y="meta_notebook_umap_2", hue="meta_cluster_idx",
        kind="hexbin", palette="tab20",
    )
    plot = base.highlight(
        pl.col("meta_old_cluster_idx").is_not_null(), hue="meta_old_cluster_idx", label="Synonymous"
    )
    _, ax = plot.plot()
    subset = df.filter(pl.col("meta_old_cluster_idx").is_not_null())
    expected = [to_rgba(base.palette[str(c)]) for c in subset["meta_old_cluster_idx"]]
    np.testing.assert_allclose(_last_scatter(ax).get_facecolors(), expected)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Synonymous"]


def test_highlight_hue_explicit_palette_and_missing(profiles):
    plot = fb.EmbeddingPlot(profiles, x="meta_notebook_umap_1", y="meta_notebook_umap_2").highlight(
        pl.col("meta_variant_type") != "Single Missense",
        hue="meta_variant_type",
        palette={"Synonymous": "green"},
    )
    _, ax = plot.plot()
    subset = profiles.filter(pl.col("meta_variant_type") != "Single Missense")
    expected = [
        to_rgba("green" if v == "Synonymous" else "white") for v in subset["meta_variant_type"]
    ]
    np.testing.assert_allclose(_last_scatter(ax).get_facecolors(), expected)


def test_highlight_hue_defaults_to_fisseq_palette(profiles):
    _, ax = (
        fb.EmbeddingPlot(profiles, x="meta_notebook_umap_1", y="meta_notebook_umap_2")
        .highlight(pl.col("meta_variant_type") == "Frameshift", hue="meta_variant_type")
        .plot()
    )
    colors = _last_scatter(ax).get_facecolors()
    np.testing.assert_allclose(colors, [to_rgba(fisseq.PALETTE["Frameshift"])] * len(colors))


def test_highlight_hue_rejects_float_column(profiles):
    plot = fb.EmbeddingPlot(profiles, x="meta_notebook_umap_1", y="meta_notebook_umap_2")
    with pytest.raises(TypeError, match="categorical"):
        plot.highlight(pl.lit(True), hue="meta_impact_score").plot()


# ----- PairPlot -----------------------------------------------------------------------


def test_pairplot_uses_fisseq_palette(profiles):
    plot = fb.PairPlot(profiles, vars=["feature_0", "feature_1", "feature_2"],
                       hue="meta_variant_type", title="PCs")
    fig, grid = plot.plot()
    assert grid.axes.shape == (3, 3)
    assert plot.hue_order == ["Synonymous", "Single Missense", "Frameshift"]
    assert plot.palette == {k: fisseq.VARIANT_TYPE_PALETTE[k] for k in plot.hue_order}
    legend = [t.get_text() for t in grid.legend.get_texts()]
    assert legend == plot.hue_order
    assert fig._suptitle.get_text() == "PCs"


def test_pairplot_natural_cluster_order_and_selector(profiles):
    import polars.selectors as cs

    plot = fb.PairPlot(profiles, vars=cs.starts_with("feature_") & cs.matches("[01]$"),
                       hue="meta_cluster_idx", corner=True)
    assert plot.vars == ["feature_0", "feature_1"]
    assert plot.hue_order == [str(i) for i in range(12)]
    _, grid = plot.plot()
    assert grid.axes[0, 1] is None  # corner=True


def test_pairplot_is_figure_level(profiles):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    with pytest.raises(TypeError, match="figure-level"):
        fb.PairPlot(profiles, vars=["feature_0"]).plot(ax=ax)
