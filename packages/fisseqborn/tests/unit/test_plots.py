import numpy as np
import polars as pl
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from scipy import stats

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
    assert [lbl.split("\n")[0] for lbl in labels] == [
        "Synonymous",
        "Single Missense",
        "Frameshift",
    ]
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
        ~(
            (pl.col("meta_experiment") == "T2_R1")
            & (pl.col("meta_variant_type") == "Frameshift")
        )
    )
    plot = fb.BoxPlot(
        df, x="meta_experiment", y="meta_impact_score", hue="meta_variant_type"
    )
    pairs = plot._resolve_pairs("all")
    assert len(pairs) == 4 * 3 - 2
    assert all(p[0][0] == p[1][0] for p in pairs)
    plot.annotate_pairs().plot()


@pytest.mark.parametrize("points", ["strip", "density"])
def test_boxplot_points(profiles, points):
    _, ax = fb.BoxPlot(
        profiles,
        x="meta_variant_type",
        y="meta_impact_score",
        hue="meta_experiment",
        points=points,
    ).plot()
    assert ax.get_legend() is not None
    assert sum(len(c.get_offsets()) for c in ax.collections) == profiles.height


# ----- EmbeddingPlot -------------------------------------------------------------------


def test_embedding_categorical_scatter_with_highlights(profiles):
    _, ax = (
        fb.EmbeddingPlot(profiles, **UMAP, hue="meta_variant_type")
        .highlight(
            pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC,
            label=fisseq.PATHOGENIC,
            marker="^",
        )
        .plot()
    )
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Synonymous", "Single Missense", "Frameshift", fisseq.PATHOGENIC]
    assert ax.get_xlabel() == UMAP["x"]


def test_embedding_numeric_scatter_has_colorbar(profiles):
    fig, _ = fb.EmbeddingPlot(
        profiles,
        **UMAP,
        hue="meta_distinguishability_score",
        cmap="Reds",
        vmin=0.5,
        vmax=1,
    ).plot()
    assert len(fig.axes) == 2  # main + colorbar


def test_embedding_hexbin_categorical_mode(profiles):
    fig, ax = fb.EmbeddingPlot(
        profiles, **UMAP, hue="meta_cluster_idx", kind="hexbin"
    ).plot()
    assert any(isinstance(c, PolyCollection) for c in ax.collections)
    cbar_labels = [t.get_text() for t in fig.axes[1].get_yticklabels()]
    assert cbar_labels == [str(i) for i in range(12)]


def test_embedding_hexbin_centered(profiles):
    plot = fb.EmbeddingPlot(
        profiles,
        **UMAP,
        hue="feature_3",
        kind="hexbin",
        cmap="RdBu_r",
        center=0,
        clip=1,
    )
    norm, extend = plot._norm(profiles.get_column("feature_3").to_numpy())
    assert (norm.vmin, norm.vmax) == (-1, 1)
    assert extend == "both"
    plot.plot()


# ----- CorrelationPlot -----------------------------------------------------------------


def test_correlation_identity_and_stat(profiles):
    plot = fb.CorrelationPlot(
        profiles,
        x="meta_distinguishability_score",
        y="meta_distinguishability_score_rep2",
        hue="meta_variant_type",
        identity=True,
        lims=(0, 1),
        count_sides=True,
    )
    _, ax = plot.plot()
    texts = [t.get_text() for t in ax.texts]
    r, _, n = plot.correlation()
    assert r > 0.8 and n == profiles.height
    assert any(t.startswith("Pearson r = ") for t in texts)
    assert sum(t.startswith("n = ") for t in texts) == 2


def test_correlation_kde_lowess_spearman(profiles):
    _, ax = fb.CorrelationPlot(
        profiles,
        x="meta_num_cells",
        y="meta_distinguishability_score",
        kind="kde",
        fit="lowess",
        stat="spearman",
    ).plot()
    assert any("Spearman" in t.get_text() for t in ax.texts)


# ----- RocPlot -------------------------------------------------------------------------


def test_roc_multiple_scores(profiles):
    plot = fb.RocPlot(
        profiles,
        label="meta_variant_type",
        positive="Single Missense",
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
        profiles,
        label="meta_variant_type",
        positive=["Single Missense", "Frameshift"],
        score="meta_distinguishability_score",
        group="meta_experiment",
        combined=True,
    )
    assert plot.aucs()["curve"].to_list() == [
        "T1_R1",
        "T1_R2",
        "T2_R1",
        "T10_R1",
        "Combined",
    ]
    plot.plot()


def test_roc_rejects_scores_and_group(profiles):
    with pytest.raises(ValueError, match="not both"):
        fb.RocPlot(
            profiles,
            label="meta_variant_type",
            positive="Frameshift",
            score=["meta_impact_score", "meta_distinguishability_score"],
            group="meta_experiment",
        )


# ----- Heatmap / ClusterMap -------------------------------------------------------------


def test_heatmap_long_form_symmetric():
    pairs = pl.DataFrame(
        {
            "barcode_a": ["b1", "b1", "b2"],
            "barcode_b": ["b2", "b3", "b3"],
            "test_auroc": [0.6, 0.7, 0.8],
        }
    )
    plot = fb.Heatmap(
        pairs,
        index="barcode_a",
        columns="barcode_b",
        values="test_auroc",
        symmetric=True,
        fill_value=0.5,
        cmap="Reds",
        vmin=0.5,
        vmax=1,
    )
    mat = plot.matrix()
    assert list(mat.index) == ["b1", "b2", "b3"] == list(mat.columns)
    assert mat.loc["b3", "b1"] == 0.7 and mat.loc["b1", "b1"] == 0.5
    plot.plot()


def test_heatmap_correlation(profiles):
    plot = fb.Heatmap.correlation(
        profiles, ["feature_0", "feature_1", "feature_2"], method="spearman"
    )
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
    return pl.DataFrame(
        {"a1": col(a), "a2": col(a), "b1": col(b), "b2": col(b), "c1": col(slice(0, 2))}
    )


def test_heatmap_correlation_block_sparse(block_sparse):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plot = fb.Heatmap.correlation(
            block_sparse, method="spearman", nan_color="black"
        )
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

    df = block_sparse.with_columns(
        pl.col("a2").fill_null(pl.col("b1"))
    )  # a2 spans both blocks
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
    return pl.DataFrame(
        rows, schema=["experiment", "variant", "test_auroc"], orient="row"
    )


def test_batch_correlation_matrix(replicate_scores):
    from scipy.stats import spearmanr

    plot = fb.BatchCorrelationHeatmap(
        replicate_scores, batch="experiment", label="variant", score="test_auroc"
    )
    pairs = plot.pairs().filter(pl.col("batch_a") != pl.col("batch_b"))
    t1 = pairs.filter((pl.col("batch_a") == "T1_R1") & (pl.col("batch_b") == "T1_R2"))
    r1 = replicate_scores.filter(pl.col("experiment") == "T1_R1").sort("variant")
    r2 = replicate_scores.filter(pl.col("experiment") == "T1_R2").sort("variant")
    assert t1["r"][0] == pytest.approx(
        spearmanr(r1["test_auroc"], r2["test_auroc"]).statistic
    )
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
        fb.BatchCorrelationHeatmap(
            dup, batch="experiment", label="variant", score="test_auroc"
        )
    fb.BatchCorrelationHeatmap(
        dup, batch="experiment", label="variant", score="test_auroc", aggregate="mean"
    )

    thin = replicate_scores.filter(
        (pl.col("experiment") != "T1_R2") | pl.col("variant").is_in(["v0", "v1"])
    )
    mat = fb.BatchCorrelationHeatmap(
        thin, batch="experiment", label="variant", score="test_auroc"
    ).matrix()
    assert np.isnan(mat.loc["T1_R1", "T1_R2"]) and np.isnan(mat.loc["T1_R2", "T1_R2"])
    assert mat.loc["T1_R1", "T1_R1"] == 1.0


def test_batch_correlation_grid(replicate_scores):
    import matplotlib as mpl

    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc"}
    plot = fb.BatchCorrelationGrid(replicate_scores, **kw)
    assert plot.pairs().equals(
        fb.BatchCorrelationHeatmap(replicate_scores, **kw).pairs()
    )
    fig, axes = plot.plot()
    assert axes.shape == (3, 3)
    assert [ax.get_title() for ax in axes[0]] == ["T1_R1", "T1_R2", "T2_R1"]

    # diagonal: histogram on a twin axes, no scatter
    assert not axes[0, 0].collections
    assert len(fig.axes) == 9 + 3 + 1  # panels, histogram twins, colorbar

    # T1_R2 (row) vs T1_R1 (column): x is T1_R1's scores, y is T1_R2's
    by = replicate_scores.pivot(on="experiment", index="variant", values="test_auroc")
    offsets = axes[1, 0].collections[0].get_offsets()
    assert np.allclose(offsets, by.select("T1_R1", "T1_R2").drop_nulls().to_numpy())
    r = plot.pairs().filter(batch_a="T1_R1", batch_b="T1_R2")["r"][0]
    assert axes[1, 0].get_facecolor() == pytest.approx(
        mpl.colormaps["Blues"](r, alpha=0.6)
    )
    assert axes[1, 0].texts[0].get_text() == f"ρ = {r:.2f}\nn = 10"

    # disjoint tiles: no points, black
    assert not axes[0, 2].collections and not axes[0, 2].texts
    assert axes[0, 2].get_facecolor()[:3] == (0.0, 0.0, 0.0)

    _, axes = fb.BatchCorrelationGrid(
        replicate_scores, tint=False, annot=False, **kw
    ).plot()
    assert axes[1, 0].get_facecolor()[:3] == (1.0, 1.0, 1.0)
    assert not axes[1, 0].texts


def test_batch_correlation_grid_y_score_and_order(replicate_scores):
    import matplotlib.pyplot as plt

    df = replicate_scores.with_columns(missense=pl.col("test_auroc") * 2)
    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc"}
    plot = fb.BatchCorrelationGrid(
        df, y_score="missense", order=["T2_R1", "T1_R2", "T1_R1"], **kw
    )
    fig, axes = plot.plot()
    assert len(fig.axes) == 9 + 1  # no histograms
    assert axes[0, 0].get_title() == "T2_R1"
    # row T1_R2 (missense) vs column T1_R1 (test_auroc), diagonal is a scatter
    by_x = df.pivot(on="experiment", index="variant", values="test_auroc")
    by_y = df.pivot(on="experiment", index="variant", values="missense")
    offsets = axes[1, 2].collections[0].get_offsets()
    assert np.allclose(offsets[:, 0], by_x["T1_R1"].drop_nulls())
    assert np.allclose(offsets[:, 1], by_y["T1_R2"].drop_nulls())
    assert axes[2, 2].collections

    with pytest.raises(ValueError, match="unknown batches"):
        fb.BatchCorrelationGrid(df, order=["T9_R9"], **kw)
    _, ax = plt.subplots()
    with pytest.raises(TypeError, match="figure-level"):
        fb.BatchCorrelationGrid(df, **kw).plot(ax=ax)


def test_batch_correlation_grid_hue(replicate_scores):
    import matplotlib.colors as mcolors

    types = ["Synonymous", "Single Missense", "Frameshift"]
    df = replicate_scores.with_columns(
        meta_variant_type=pl.col("variant")
        .str.extract(r"(\d+)$")
        .cast(pl.Int64)
        .map_elements(lambda i: types[i % 3], return_dtype=pl.String)
    )
    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc"}
    plot = fb.BatchCorrelationGrid(df, hue="meta_variant_type", **kw)
    assert plot.hue_order == types
    fig, axes = plot.plot()

    # one scatter per level, colored by the fisseq palette; the first level drawn last
    panel = axes[1, 0].collections
    assert len(panel) == 3
    for coll, level in zip(panel, types[::-1], strict=True):
        assert mcolors.same_color(
            coll.get_facecolor()[0][:3], fisseq.VARIANT_TYPE_PALETTE[level]
        )
    t1 = df.filter(experiment="T1_R1")
    for coll, level in zip(panel, types[::-1], strict=True):
        assert len(coll.get_offsets()) == t1.filter(meta_variant_type=level).height

    # diagonal: one histogram per level, on shared bins
    hist_ax = next(
        a
        for a in fig.axes
        if a is not axes[0, 0] and a.bbox.bounds == axes[0, 0].bbox.bounds
    )
    assert len(hist_ax.patches) == 3
    assert [t.get_text() for t in fig.legends[0].get_texts()] == types

    # correlations ignore hue
    assert plot.pairs().equals(fb.BatchCorrelationGrid(df, **kw).pairs())

    clash = df.with_columns(
        meta_variant_type=pl.when(pl.col("experiment") == "T1_R2")
        .then(pl.lit("Frameshift"))
        .otherwise(pl.col("meta_variant_type"))
    )
    with pytest.raises(ValueError, match="one value per"):
        fb.BatchCorrelationGrid(clash, hue="meta_variant_type", **kw)


def test_batch_correlation_grid_tick_labels(replicate_scores):
    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc"}
    fig, axes = fb.BatchCorrelationGrid(replicate_scores, **kw).plot()
    fig.canvas.draw()
    for (_, j), ax in np.ndenumerate(axes):
        assert all(t.get_visible() for t in ax.get_xticklabels())
        assert any(t.get_visible() for t in ax.get_yticklabels()) == (j == 0)


def test_batch_correlation_grid_marginals(replicate_scores):
    types = ["Synonymous", "Single Missense", "Frameshift"]
    df = replicate_scores.with_columns(
        meta_variant_type=pl.col("variant")
        .str.extract(r"(\d+)$")
        .cast(pl.Int64)
        .map_elements(lambda i: types[i % 3], return_dtype=pl.String)
    )
    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc"}

    _, axes = fb.BatchCorrelationGrid(df, marginals="hist", **kw).plot()
    top, right = axes[1, 0].child_axes
    assert sum(p.get_height() for p in top.patches) == 10  # the shared labels
    assert sum(p.get_width() for p in right.patches) == 10
    assert top.get_xlim() == axes[1, 0].get_xlim()
    assert right.get_ylim() == axes[1, 0].get_ylim()
    assert not axes[0, 2].child_axes and not axes[0, 0].child_axes  # disjoint, diagonal

    _, axes = fb.BatchCorrelationGrid(
        df, marginals="hist", hue="meta_variant_type", **kw
    ).plot()
    top, _ = axes[1, 0].child_axes
    assert len({p.get_facecolor() for p in top.patches}) == 3

    _, plain = fb.BatchCorrelationGrid(df, **kw).plot()
    _, axes = fb.BatchCorrelationGrid(df, marginals="kde", **kw).plot()
    top, right = axes[1, 0].child_axes
    assert top.lines and right.lines
    assert axes[1, 0].get_xlim() == plain[1, 0].get_xlim()  # strips don't rescale

    with pytest.raises(ValueError, match="marginals"):
        fb.BatchCorrelationGrid(df, marginals="violin", **kw)


def test_batch_correlation_grid_lims_and_label_rotation(replicate_scores):
    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc"}
    _, axes = fb.BatchCorrelationGrid(
        replicate_scores, lims=(0, 1), label_rotation=45, **kw
    ).plot()
    for ax in axes.flat:
        assert ax.get_xlim() == (0, 1) and ax.get_ylim() == (0, 1)
    assert axes[0, 1].title.get_rotation() == 45
    assert axes[1, 0].yaxis.label.get_rotation() == 45

    _, axes = fb.BatchCorrelationGrid(replicate_scores, **kw).plot()
    assert axes[0, 1].title.get_rotation() == 0
    assert axes[1, 0].yaxis.label.get_rotation() == 90


def test_cross_correlation_matches_pairwise(block_sparse, profiles):
    from fisseqborn.heatmap import cross_correlation, pairwise_correlation

    for df in (block_sparse, profiles.select(f"feature_{i}" for i in range(4))):
        m = df.cast(pl.Float64).fill_null(np.nan).to_numpy()
        for method in ("pearson", "spearman", "cosine"):
            corr, n = pairwise_correlation(m, method, min_shared=2)
            cross, cross_n = cross_correlation(m, m, method, min_shared=2)
            assert np.allclose(cross, corr, equal_nan=True)
            assert (cross_n == n).all()


def test_batch_correlation_cross_scores(replicate_scores):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(2)
    df = replicate_scores.with_columns(
        missense=pl.col("test_auroc") + rng.normal(0, 0.2, replicate_scores.height)
    )
    kw = {"batch": "experiment", "label": "variant", "score": "test_auroc"}
    plot = fb.BatchCorrelationHeatmap(df, y_score="missense", **kw)
    mat = plot.matrix()

    r1 = df.filter(pl.col("experiment") == "T1_R1").sort("variant")
    r2 = df.filter(pl.col("experiment") == "T1_R2").sort("variant")
    assert mat.loc["T1_R1", "T1_R2"] == pytest.approx(
        spearmanr(r1["missense"], r2["test_auroc"]).statistic
    )
    assert mat.loc["T1_R2", "T1_R1"] == pytest.approx(
        spearmanr(r2["missense"], r1["test_auroc"]).statistic
    )
    assert mat.loc["T1_R1", "T1_R2"] != mat.loc["T1_R2", "T1_R1"]
    assert mat.loc["T1_R1", "T1_R1"] == pytest.approx(
        spearmanr(r1["missense"], r1["test_auroc"]).statistic
    )
    assert not np.allclose(np.diag(mat), 1.0)
    assert np.isnan(mat.loc["T1_R1", "T2_R1"]) and np.isnan(mat.loc["T2_R1", "T1_R1"])

    pairs = plot.pairs()
    assert pairs.height == 9
    row = pairs.filter(pl.col("batch_a") == "T1_R2", pl.col("batch_b") == "T1_R1")
    assert row["r"][0] == pytest.approx(mat.loc["T1_R2", "T1_R1"])

    _, ax = plot.plot()
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("test_auroc", "missense")

    same = fb.BatchCorrelationHeatmap(df, y_score="test_auroc", **kw)
    plain = fb.BatchCorrelationHeatmap(df, **kw)
    assert np.allclose(same.matrix(), plain.matrix(), equal_nan=True)
    assert same.pairs().equals(plain.pairs())


def test_heatmap_correlation_y_columns(profiles):
    cols, other = ["feature_0", "feature_1", "feature_2"], ["feature_3", "feature_4"]
    plot = fb.Heatmap.correlation(profiles, cols, y_columns=other)
    mat = plot.matrix()
    assert list(mat.index) == other and list(mat.columns) == cols
    full = np.corrcoef(profiles.select(*other, *cols).to_numpy(), rowvar=False)
    assert np.allclose(mat.to_numpy(), full[:2, 2:])
    assert list(plot.n_shared.index) == other and list(plot.n_shared.columns) == cols
    plot.plot()


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
    return profiles.unpivot(
        on=[f"feature_{i}" for i in range(8)],
        index=["meta_aa_changes", "meta_variant_type"],
        variable_name="feature",
        value_name="median",
    ).with_columns((pl.col("median").abs() * 3).alias("negLogP"))


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
        pl.when(pl.col("feature") == "feature_0")
        .then(None)
        .otherwise(pl.col("median"))
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
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "Bonferroni p = 0.05"
    ]


def test_volcano_defaults_and_options(volcano_rows):
    _, ax = fb.VolcanoPlot(
        volcano_rows, x="median", y="negLogP", alpha=None, x_quantiles=None
    ).plot()
    assert len(ax.collections) == 1 and not ax.lines and ax.get_legend() is None
    assert len(ax.collections[0].get_offsets()) == volcano_rows.height
    assert fb.VolcanoPlot(
        volcano_rows, x="median", y="negLogP", bonferroni=False
    ).significance_threshold() == pytest.approx(-np.log10(0.05))
    lo, hi = fb.VolcanoPlot(volcano_rows, x="median", y="negLogP").plot()[1].get_xlim()
    q_lo, q_hi = (
        volcano_rows["median"].quantile(0.001),
        volcano_rows["median"].quantile(0.999),
    )
    pad = 0.05 * (q_hi - q_lo)
    assert (lo, hi) == pytest.approx((q_lo - pad, q_hi + pad))


def test_volcano_from_wide_matches_long_form(profiles):
    wide = profiles.select(
        "meta_aa_changes",
        "meta_variant_type",
        *[pl.col(f"feature_{i}").alias(f"feature_{i}_median") for i in range(3)],
        *[
            (pl.col(f"feature_{i}").abs() * 3).alias(f"feature_{i}_KSnegLogP")
            for i in range(3)
        ],
        pl.col("feature_3").alias("feature_3_median"),  # no p-value column: skipped
    )
    plot = fb.VolcanoPlot.from_wide(wide, alpha=None)
    assert (plot.x, plot.y) == ("median", "KSnegLogP")
    assert set(plot.data.columns) == {
        "meta_aa_changes",
        "meta_variant_type",
        "feature",
        "median",
        "KSnegLogP",
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


def test_volcano_point_style_scales_with_point_count(volcano_rows):
    few = fb.VolcanoPlot._point_style(100)
    many = fb.VolcanoPlot._point_style(10_000_000)
    assert few == {"s": 20, "alpha": 0.8} and many == {"s": 1, "alpha": 0.1}
    _, ax = fb.VolcanoPlot(volcano_rows, x="median", y="negLogP").plot()
    style = fb.VolcanoPlot._point_style(volcano_rows.height)
    assert ax.collections[0].get_sizes()[0] == pytest.approx(style["s"])
    # a scatter kwarg still overrides the scaled default
    _, ax = fb.VolcanoPlot(volcano_rows, x="median", y="negLogP", s=3).plot()
    assert ax.collections[0].get_sizes()[0] == 3


def test_volcano_effect_cutoff_and_top_labels(volcano_rows):
    plot = fb.VolcanoPlot(
        volcano_rows,
        x="median",
        y="negLogP",
        effect_cutoff=-0.2,
        label_top=3,
        label_rank="x",
        label_by="meta_aa_changes",
    )
    _, ax = plot.plot()
    assert sorted(line.get_xdata()[0] for line in ax.lines[1:]) == [-0.2, 0.2]
    assert ax.get_legend().get_texts()[-1].get_text() == "|median| = 0.2"
    threshold = plot.significance_threshold()
    expected = (
        volcano_rows.filter(
            pl.col("negLogP") >= threshold, pl.col("median").abs() >= 0.2
        )
        .sort(pl.col("median").abs(), descending=True)
        .head(3)
    )
    assert expected.height == 3
    assert [t.get_text() for t in ax.texts] == expected["meta_aa_changes"].to_list()
    lo, hi = ax.get_xlim()
    assert all(lo <= x <= hi for x in expected["median"])


def test_volcano_top_labels_rank_by_y_inside_axes():
    # the most significant points are not the largest effects
    rows = pl.DataFrame(
        {
            "feature": [f"f{i}" for i in range(8)],
            "x": [3.0, -3.0, 2.5, 0.5, -0.4, 0.3, 0.1, -0.2],
            "y": [5.0, 5.1, 4.9, 20.0, 19.0, 18.5, 18.0, 0.1],
        }
    )
    plot = fb.VolcanoPlot(rows, x="x", y="y", alpha=None, x_quantiles=None, label_top=4)
    fig, ax = plot.plot()
    assert sorted(t.get_text() for t in ax.texts) == ["f3", "f4", "f5", "f6"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    boxes = [t.get_window_extent(renderer) for t in ax.texts]
    assert all(
        axes_box.contains(b.x0, b.y0) and axes_box.contains(b.x1, b.y1) for b in boxes
    )
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    with pytest.raises(ValueError, match="label_rank"):
        fb.VolcanoPlot(rows, x="x", y="y", label_rank="z")


@pytest.fixture
def labelled_rows() -> pl.DataFrame:
    """Rows with a bool and a 0/1 label; ``shifted`` moves with ``meta_is_case``."""
    rng = np.random.default_rng(0)
    n = 200
    case = rng.random(n) < 0.4
    noisy = rng.normal(0, 1, n)
    noisy[::7] = np.nan
    return pl.DataFrame(
        {
            "meta_id": np.arange(n),
            "meta_is_case": case,
            "meta_other": pl.Series(rng.integers(0, 2, n)).scatter([0, 1], None),
            "shifted": rng.normal(0, 1, n) + 1.5 * case,
            "flat": rng.normal(0, 1, n),
            "noisy": noisy,
        }
    )


@pytest.mark.parametrize(
    "test,column,pvalue",
    [
        ("ks", "KSnegLogP", lambda a, b: stats.ks_2samp(a, b).pvalue),
        (
            "welch",
            "WelchNegLogP",
            lambda a, b: stats.ttest_ind(a, b, equal_var=False).pvalue,
        ),
        ("mannwhitney", "MWUnegLogP", lambda a, b: stats.mannwhitneyu(a, b).pvalue),
    ],
)
def test_volcano_from_labels_tests(labelled_rows, test, column, pvalue):
    plot = fb.VolcanoPlot.from_labels(labelled_rows, "meta_is_case", test=test)
    assert (plot.x, plot.y) == ("median_diff", column)
    assert plot.data["feature"].to_list() == ["shifted", "flat", "noisy"]
    for feat in ["shifted", "flat", "noisy"]:
        rows = labelled_rows.filter(pl.col(feat).is_not_nan())
        a = rows.filter(pl.col("meta_is_case"))[feat].to_numpy()
        b = rows.filter(~pl.col("meta_is_case"))[feat].to_numpy()
        got = plot.data.filter(pl.col("feature") == feat)[column][0]
        assert got == pytest.approx(-np.log10(pvalue(a, b)))
    assert plot.data.sort(column)["feature"][-1] == "shifted"


@pytest.mark.parametrize(
    "effect,column,expected",
    [
        ("median", "median_diff", lambda a, b: np.median(a) - np.median(b)),
        ("mean", "mean_diff", lambda a, b: a.mean() - b.mean()),
        (
            "cohens_d",
            "cohens_d",
            lambda a, b: (
                (a.mean() - b.mean())
                / np.sqrt(
                    ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                    / (len(a) + len(b) - 2)
                )
            ),
        ),
    ],
)
def test_volcano_from_labels_effects(labelled_rows, effect, column, expected):
    plot = fb.VolcanoPlot.from_labels(
        labelled_rows, "meta_is_case", features=["noisy"], effect=effect
    )
    assert (plot.x, plot.y) == (column, "KSnegLogP")
    rows = labelled_rows.filter(pl.col("noisy").is_not_nan())
    a = rows.filter(pl.col("meta_is_case"))["noisy"].to_numpy()
    b = rows.filter(~pl.col("meta_is_case"))["noisy"].to_numpy()
    assert plot.data[column][0] == pytest.approx(expected(a, b))
    assert plot.data.row(0, named=True)["n_positive"] == len(a)
    assert plot.data.row(0, named=True)["n_negative"] == len(b)


def test_volcano_from_labels_several_labels(labelled_rows):
    plot = fb.VolcanoPlot.from_labels(
        labelled_rows, ["meta_is_case", "meta_other"], alpha=0.05
    )
    assert plot.data.height == 6
    assert plot.data["label"].unique(maintain_order=True).to_list() == [
        "meta_is_case",
        "meta_other",
    ]
    # the two null labels are left out of meta_other's comparison
    flat = plot.data.filter(
        pl.col("label") == "meta_other", pl.col("feature") == "flat"
    )
    assert flat["n_positive"][0] + flat["n_negative"][0] == labelled_rows.height - 2
    assert plot.significance_threshold() == pytest.approx(-np.log10(0.05 / 6))
    _, ax = plot.layer(pl.col("label") == "meta_is_case", label="case").plot()
    assert len(ax.collections[0].get_offsets()) == 3


def test_volcano_from_labels_errors(labelled_rows):
    from_labels = fb.VolcanoPlot.from_labels
    with pytest.raises(ValueError, match="boolean or 0/1"):
        from_labels(labelled_rows, "meta_id")
    with pytest.raises(ValueError, match="both true and false"):
        from_labels(
            labelled_rows.with_columns(meta_is_case=pl.lit(True)), "meta_is_case"
        )
    with pytest.raises(ValueError, match="Unknown test"):
        from_labels(labelled_rows, "meta_is_case", test="anova")
    with pytest.raises(ValueError, match="Unknown effect"):
        from_labels(labelled_rows, "meta_is_case", effect="log2fc")
    with pytest.raises(ValueError, match="No feature"):
        from_labels(labelled_rows.select("meta_is_case"), "meta_is_case")


def test_heatmap_correlation_squared(profiles):
    cols = ["feature_0", "feature_1", "feature_2"]
    r = fb.Heatmap.correlation(profiles, cols).matrix()
    plot = fb.Heatmap.correlation(profiles, cols, squared=True)
    assert np.allclose(plot.matrix().to_numpy(), r.to_numpy() ** 2)
    assert plot.heatmap_kw["vmin"] == 0 and plot.heatmap_kw["cmap"] == "Blues"


@pytest.mark.parametrize("method", ["pearson", "spearman", "cosine"])
def test_batch_correlation_methods_and_squared(replicate_scores, method):
    kw = {
        "batch": "experiment",
        "label": "variant",
        "score": "test_auroc",
        "method": method,
    }
    plain = fb.BatchCorrelationHeatmap(replicate_scores, **kw)
    squared = fb.BatchCorrelationHeatmap(replicate_scores, squared=True, **kw)
    expected = fb.Heatmap.correlation(
        replicate_scores.pivot(on="experiment", index="variant", values="test_auroc"),
        ["T1_R1", "T1_R2", "T2_R1"],
        method=method,
    ).matrix()
    assert np.allclose(plain.matrix().to_numpy(), expected.to_numpy(), equal_nan=True)
    assert np.allclose(
        squared.matrix().to_numpy(), expected.to_numpy() ** 2, equal_nan=True
    )
    pairs = squared.pairs().drop_nans("r")
    assert np.allclose(pairs["r_squared"], pairs["r"] ** 2)
    squared.plot()


def test_batch_correlation_rejects_unknown_method(replicate_scores):
    with pytest.raises(ValueError, match="cosine"):
        fb.BatchCorrelationHeatmap(
            replicate_scores,
            batch="experiment",
            label="variant",
            score="test_auroc",
            method="kendall",
        )


def _cbar_label(ax):
    return ax.collections[0].colorbar.ax.get_ylabel()


@pytest.mark.parametrize(
    ("method", "squared", "label"),
    [
        ("pearson", False, "Pearson r"),
        ("spearman", True, "Spearman ρ²"),
        ("cosine", False, "Cosine similarity"),
    ],
)
def test_correlation_heatmaps_label_colorbar(
    profiles, replicate_scores, method, squared, label
):
    _, ax = fb.Heatmap.correlation(
        profiles, ["feature_0", "feature_1"], method=method, squared=squared
    ).plot()
    assert _cbar_label(ax) == label
    _, ax = fb.BatchCorrelationHeatmap(
        replicate_scores,
        batch="experiment",
        label="variant",
        score="test_auroc",
        method=method,
        squared=squared,
    ).plot()
    assert _cbar_label(ax) == label


def test_correlation_colorbar_label_can_be_overridden(profiles):
    _, ax = fb.Heatmap.correlation(
        profiles,
        ["feature_0", "feature_1"],
        cbar_kws={"label": "custom", "shrink": 0.5},
    ).plot()
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
        df,
        x="meta_notebook_umap_1",
        y="meta_notebook_umap_2",
        hue="meta_cluster_idx",
        kind="hexbin",
        palette="tab20",
    )
    plot = base.highlight(
        pl.col("meta_old_cluster_idx").is_not_null(),
        hue="meta_old_cluster_idx",
        label="Synonymous",
    )
    _, ax = plot.plot()
    subset = df.filter(pl.col("meta_old_cluster_idx").is_not_null())
    expected = [to_rgba(base.palette[str(c)]) for c in subset["meta_old_cluster_idx"]]
    np.testing.assert_allclose(_last_scatter(ax).get_facecolors(), expected)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["Synonymous"]


def test_highlight_hue_explicit_palette_and_missing(profiles):
    plot = fb.EmbeddingPlot(
        profiles, x="meta_notebook_umap_1", y="meta_notebook_umap_2"
    ).highlight(
        pl.col("meta_variant_type") != "Single Missense",
        hue="meta_variant_type",
        palette={"Synonymous": "green"},
    )
    _, ax = plot.plot()
    subset = profiles.filter(pl.col("meta_variant_type") != "Single Missense")
    expected = [
        to_rgba("green" if v == "Synonymous" else "white")
        for v in subset["meta_variant_type"]
    ]
    np.testing.assert_allclose(_last_scatter(ax).get_facecolors(), expected)


def test_highlight_hue_defaults_to_fisseq_palette(profiles):
    _, ax = (
        fb.EmbeddingPlot(profiles, x="meta_notebook_umap_1", y="meta_notebook_umap_2")
        .highlight(pl.col("meta_variant_type") == "Frameshift", hue="meta_variant_type")
        .plot()
    )
    colors = _last_scatter(ax).get_facecolors()
    np.testing.assert_allclose(
        colors, [to_rgba(fisseq.PALETTE["Frameshift"])] * len(colors)
    )


def test_highlight_hue_rejects_float_column(profiles):
    plot = fb.EmbeddingPlot(
        profiles, x="meta_notebook_umap_1", y="meta_notebook_umap_2"
    )
    with pytest.raises(TypeError, match="categorical"):
        plot.highlight(pl.lit(True), hue="meta_impact_score").plot()


# ----- PairPlot -----------------------------------------------------------------------


def test_pairplot_uses_fisseq_palette(profiles):
    plot = fb.PairPlot(
        profiles,
        vars=["feature_0", "feature_1", "feature_2"],
        hue="meta_variant_type",
        title="PCs",
    )
    fig, grid = plot.plot()
    assert grid.axes.shape == (3, 3)
    assert plot.hue_order == ["Synonymous", "Single Missense", "Frameshift"]
    assert plot.palette == {k: fisseq.VARIANT_TYPE_PALETTE[k] for k in plot.hue_order}
    legend = [t.get_text() for t in grid.legend.get_texts()]
    assert legend == plot.hue_order
    assert fig._suptitle.get_text() == "PCs"


def test_pairplot_natural_cluster_order_and_selector(profiles):
    import polars.selectors as cs

    plot = fb.PairPlot(
        profiles,
        vars=cs.starts_with("feature_") & cs.matches("[01]$"),
        hue="meta_cluster_idx",
        corner=True,
    )
    assert plot.vars == ["feature_0", "feature_1"]
    assert plot.hue_order == [str(i) for i in range(12)]
    _, grid = plot.plot()
    assert grid.axes[0, 1] is None  # corner=True


def test_pairplot_is_figure_level(profiles):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    with pytest.raises(TypeError, match="figure-level"):
        fb.PairPlot(profiles, vars=["feature_0"]).plot(ax=ax)
