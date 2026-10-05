import numpy as np
import polars as pl
import pytest
from matplotlib.lines import Line2D

import fisseqborn as fb


@pytest.fixture
def pca_profiles() -> fb.Profiles:
    rng = np.random.default_rng(3)
    n, k = 40, 6
    latent = rng.normal(size=(n, 2))
    x = latent @ rng.normal(size=(2, k)) + rng.normal(0, 0.3, (n, k))
    labels = [f"A{i}A" if i < 10 else f"D{i}V" for i in range(n)]
    return fb.Profiles(
        pl.DataFrame(
            {"meta_aa_changes": labels} | {f"f{i}_median": x[:, i] for i in range(k)}
        )
    ).variant_type()


def test_pca_reduce_keeps_every_components_variance(pca_profiles):
    reduced = pca_profiles.pca_reduce(variance=0.8)
    ev = reduced.pca_explained_variance
    assert ev.height == 6
    assert ev.columns == [
        "component",
        "explained_variance_ratio",
        "cumulative_explained_variance",
        "retained",
    ]
    np.testing.assert_allclose(
        ev["cumulative_explained_variance"], np.cumsum(ev["explained_variance_ratio"])
    )
    assert ev["cumulative_explained_variance"][-1] == pytest.approx(1.0)
    assert ev["retained"].sum() == len(reduced.features)
    assert ev.filter("retained")["component"].to_list() == reduced.features
    assert reduced.pca_noise_floor is None
    # chained transforms drop the stale fit, like pca_loadings
    assert (
        reduced.with_columns(pl.lit(1).alias("meta_x")).pca_explained_variance is None
    )


def test_pca_reduce_noise_floor_is_stored(pca_profiles):
    reduced = pca_profiles.pca_reduce(noise_floor=True)
    ratios = reduced.pca_explained_variance["explained_variance_ratio"].to_numpy()
    assert 0 < reduced.pca_noise_floor < 1
    assert len(reduced.features) == max(
        1, int((ratios > reduced.pca_noise_floor).sum())
    )


def test_impact_scores_match_pca_reduce(pca_profiles):
    variances = [0.5, 0.9, 1.0]
    scored = pca_profiles.impact_scores(variances)
    assert scored.features == pca_profiles.features
    df = scored.df
    for v in variances:
        col = f"meta_impact_score_{v}"
        expected = (
            pca_profiles.pca_reduce(variance=v).impact_score(output_col=col).df[col]
        )
        np.testing.assert_allclose(df[col], expected)
    assert scored.pca_explained_variance.height == 6
    fb.RocPlot(
        scored,
        label="meta_variant_type",
        positive="Single Missense",
        score=[f"meta_impact_score_{v}" for v in variances],
    ).plot()


def test_impact_scores_requires_variant_type(pca_profiles):
    with pytest.raises(ValueError, match="variant_type"):
        pca_profiles.drop("meta_is_control").impact_scores([0.9])


def _dashed_lines(ax):
    return [ln for ln in ax.get_lines() if ln.get_linestyle() == "--"]


@pytest.mark.parametrize("kind", ["cumulative", "both"])
def test_explained_variance_plot_marks_thresholds(pca_profiles, kind):
    reduced = pca_profiles.pca_reduce(variance=0.9, noise_floor=True)
    plot = fb.ExplainedVariancePlot(reduced, kind=kind, thresholds=[0.5, 0.9])
    fig, ax = plot.plot()
    curve = ax.get_lines()[0]
    np.testing.assert_allclose(
        curve.get_ydata(),
        reduced.pca_explained_variance["cumulative_explained_variance"],
    )
    labels = ax.get_legend_handles_labels()[1]
    k = plot.n_components(0.9)
    assert f"0.9: {k} PCs" in labels
    assert any(label.startswith("noise floor") for label in labels)
    assert len(_dashed_lines(ax)) == 4  # a horizontal and a vertical line per threshold
    if kind == "both":
        assert len(fig.axes) == 2 and len(fig.axes[1].patches) == 6


def test_explained_variance_plot_scree_and_fraction(pca_profiles):
    ev = pca_profiles.pca_reduce().pca_explained_variance
    _, ax = fb.ExplainedVariancePlot(
        ev, kind="scree", x="fraction", noise_floor=0.05
    ).plot()
    assert len(ax.patches) == 6
    assert ax.patches[-1].get_x() + ax.patches[-1].get_width() / 2 == pytest.approx(1.0)
    assert any(
        isinstance(ln, Line2D) and ln.get_ydata()[0] == 0.05 for ln in ax.get_lines()
    )


def test_explained_variance_plot_needs_a_fit(pca_profiles):
    with pytest.raises(ValueError, match="pca_reduce"):
        fb.ExplainedVariancePlot(pca_profiles)
