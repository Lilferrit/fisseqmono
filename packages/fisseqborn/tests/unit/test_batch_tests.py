import matplotlib
import numpy as np
import polars as pl
import pytest
from scipy import stats

import fisseqborn as fb
from fisseqborn.batch_tests import _permanova_pair

matplotlib.use("Agg")


@pytest.fixture
def profiles() -> pl.DataFrame:
    """T1_R1 and T1_R2 replicate v0-v29 (T1_R2 shifted on f0); T1_R3 is T1_R1 plus noise;
    T2_R1 has disjoint variants."""
    rng = np.random.default_rng(0)
    n, k = 30, 4
    base = rng.normal(size=(n, k))
    shift = np.zeros(k)
    shift[0] = 1.5

    def batch(name: str, x: np.ndarray, prefix: str = "v") -> pl.DataFrame:
        return pl.DataFrame(
            {
                "meta_experiment": [name] * len(x),
                "meta_aa_changes": [f"{prefix}{i}" for i in range(len(x))],
                **{f"f{j}": x[:, j] for j in range(k)},
            }
        )

    return pl.concat(
        [
            batch("T1_R1", base + rng.normal(0, 0.3, (n, k))),
            batch("T1_R2", base + shift + rng.normal(0, 0.3, (n, k))),
            batch("T1_R3", base + rng.normal(0, 0.3, (n, k))),
            batch("T2_R1", rng.normal(size=(n, k)), prefix="w"),
        ]
    )


def _pair(df: pl.DataFrame, a: str, b: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    xa = df.filter(pl.col("meta_experiment") == a).sort("meta_aa_changes")[col]
    xb = df.filter(pl.col("meta_experiment") == b).sort("meta_aa_changes")[col]
    return xa.to_numpy(), xb.to_numpy()


def test_anova_is_paired_t_squared(profiles):
    res = fb.BatchTestMatrix.anova(
        profiles, batch="meta_experiment", label="meta_aa_changes", score="f0"
    )
    assert res.batches == ["T1_R1", "T1_R2", "T1_R3", "T2_R1"]
    a, b = _pair(profiles, "T1_R1", "T1_R2", "f0")
    t = stats.ttest_rel(a, b)
    i, j = 0, 1
    assert res.f[i, j] == pytest.approx(t.statistic**2)
    assert res.p[i, j] == pytest.approx(t.pvalue)
    assert res.f[j, i] == res.f[i, j]
    assert np.isnan(np.diag(res.f)).all()
    assert np.isnan(res.f[0, 3]) and res.n_shared[0, 3] == 0
    assert res.n_shared[0, 0] == 30 and res.n_shared[0, 1] == 30


def test_permanova_on_one_feature_matches_anova(profiles):
    kw = {"batch": "meta_experiment", "label": "meta_aa_changes"}
    anova = fb.BatchTestMatrix.anova(profiles, score="f0", **kw)
    perm = fb.BatchTestMatrix.permanova(
        profiles, features=["f0"], metric="euclidean", permutations=99, **kw
    )
    finite = np.isfinite(anova.f)
    assert np.array_equal(finite, np.isfinite(perm.f))
    assert np.allclose(perm.f[finite], anova.f[finite])


def test_permanova_detects_shift(profiles):
    res = fb.BatchTestMatrix.permanova(profiles, permutations=199)
    pos = {b: i for i, b in enumerate(res.batches)}
    shifted = res.p[pos["T1_R1"], pos["T1_R2"]]
    same = res.p[pos["T1_R1"], pos["T1_R3"]]
    assert shifted == pytest.approx(1 / 200)
    assert same > 0.05
    assert res.f[pos["T1_R1"], pos["T1_R2"]] > res.f[pos["T1_R1"], pos["T1_R3"]]
    assert np.allclose(res.f, res.f.T, equal_nan=True)
    assert np.isnan(res.f[pos["T1_R1"], pos["T2_R1"]])
    assert res.params["metric"] == "euclidean" and res.params["n_features"] == 4


def test_permanova_reproducible_across_jobs(profiles):
    one = fb.BatchTestMatrix.permanova(profiles, permutations=99, seed=3)
    two = fb.BatchTestMatrix.permanova(profiles, permutations=99, seed=3, n_jobs=2)
    assert np.array_equal(one.p, two.p, equal_nan=True)


def test_permanova_metric_and_min_shared(profiles):
    res = fb.BatchTestMatrix.permanova(
        profiles, metric="pearson", permutations=19, min_shared=31
    )
    assert np.isnan(res.f).all()
    res = fb.BatchTestMatrix.permanova(profiles, metric="pearson", permutations=19)
    assert np.isfinite(res.f[0, 1]) and res.f[0, 1] >= 0
    assert res.params["metric"] == "pearson" and res.params["sqrt"] is True


def test_permanova_from_profiles(profiles):
    ds = fb.Profiles(profiles)
    res = fb.BatchTestMatrix.permanova(ds, permutations=19)
    assert res.params["batch"] == "meta_experiment"
    assert res.params["n_features"] == 4


def test_permanova_pair_identical_rows_is_nan():
    x = np.arange(10.0)[:, None]
    d = np.abs(np.vstack([x, x]) - np.vstack([x, x]).T)
    f, p = _permanova_pair(d, 10, 9, np.random.default_rng(0))
    assert np.isnan(f) and np.isnan(p)


def test_save_load_round_trip(profiles, tmp_path):
    res = fb.BatchTestMatrix.permanova(profiles, permutations=19)
    path = tmp_path / "sub" / "permanova.parquet"
    assert res.save(path) is res
    back = fb.BatchTestMatrix.load(path)
    assert back.batches == res.batches and back.test == "permanova"
    assert back.params == res.params
    assert np.array_equal(back.f, res.f, equal_nan=True)
    assert np.array_equal(back.p, res.p, equal_nan=True)
    assert np.array_equal(back.n_shared, res.n_shared)
    assert back.pairs().equals(res.pairs())


def test_matrix_views(profiles):
    res = fb.BatchTestMatrix.permanova(profiles, permutations=19)
    f = res.matrix("f")
    assert f.metric.label == "pseudo-F" and f.symmetric
    assert np.array_equal(f.n_shared, res.n_shared)
    logp = res.matrix("-log10p").values
    assert np.allclose(logp, -np.log10(res.p), equal_nan=True)
    with pytest.raises(ValueError, match="value"):
        res.matrix("q")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", ["f", "p", "-log10p"])
def test_heatmaps_draw(profiles, tmp_path, value):
    perm = fb.BatchPermanovaHeatmap(profiles, permutations=19, value=value)
    _, ax = perm.plot()
    mat = perm.matrix()
    assert list(mat.index) == ["T1_R1", "T1_R2", "T1_R3", "T2_R1"]
    assert len(ax.texts) == int(np.isfinite(mat.to_numpy()).sum())
    assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)

    perm.result.save(tmp_path / "p.parquet")
    again = fb.BatchPermanovaHeatmap(
        fb.BatchTestMatrix.load(tmp_path / "p.parquet"), value=value
    )
    assert np.allclose(again.matrix(), mat, equal_nan=True)

    anova = fb.BatchAnovaHeatmap(
        profiles,
        batch="meta_experiment",
        label="meta_aa_changes",
        score="f0",
        value=value,
    )
    anova.plot()
    assert anova.pairs().columns == ["batch_a", "batch_b", "f", "p", "n_shared"]


def test_heatmap_rejects_wrong_result(profiles):
    anova = fb.BatchTestMatrix.anova(
        profiles, batch="meta_experiment", label="meta_aa_changes", score="f0"
    )
    with pytest.raises(ValueError, match="permanova"):
        fb.BatchPermanovaHeatmap(anova)
    with pytest.raises(ValueError, match="score"):
        fb.BatchAnovaHeatmap(profiles, batch="meta_experiment", label="meta_aa_changes")
