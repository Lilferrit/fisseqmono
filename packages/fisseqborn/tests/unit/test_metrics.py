import numpy as np
import polars as pl
import pytest
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import kneighbors_graph

import fisseqborn as fb
from fisseqborn.metrics import (
    Cosine,
    CosineDistance,
    Euclidean,
    Metric,
    MetricMatrix,
    Pearson,
    Spearman,
    names,
)


@pytest.fixture
def x() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(25, 6))


def test_names_resolve():
    assert names() == [
        "euclidean",
        "cityblock",
        "cosine_distance",
        "cosine",
        "pearson",
        "spearman",
        "cosine_squared",
        "pearson_squared",
        "spearman_squared",
    ]
    for name in names():
        assert Metric.get(name).name == name
    assert Metric.get("pearson_squared") == Pearson(squared=True)
    assert Metric.get(Spearman()) == Spearman()
    with pytest.raises(ValueError, match="Unknown metric 'kendall'.*euclidean"):
        Metric.get("kendall")


def test_labels_and_kinds():
    assert Spearman(squared=True).label == "Spearman ρ²"
    assert Spearman(squared=True).symbol == "ρ²"
    assert Euclidean().kind == CosineDistance().kind == "distance"
    assert Cosine().kind == "similarity"


def _reference(name: str, a: np.ndarray, b: np.ndarray) -> float:
    base = name.removesuffix("_squared")
    value = {
        "euclidean": lambda: np.linalg.norm(a - b),
        "cityblock": lambda: np.abs(a - b).sum(),
        "cosine_distance": lambda: 1 - a @ b / np.linalg.norm(a) / np.linalg.norm(b),
        "cosine": lambda: a @ b / np.linalg.norm(a) / np.linalg.norm(b),
        "pearson": lambda: stats.pearsonr(a, b).statistic,
        "spearman": lambda: stats.spearmanr(a, b).statistic,
    }[base]()
    return value**2 if name.endswith("_squared") else value


@pytest.mark.parametrize("name", names())
def test_value_and_pairwise_match_references(x, name):
    metric = Metric.get(name)
    assert metric.value(x[:, 0], x[:, 1]) == pytest.approx(
        _reference(name, x[:, 0], x[:, 1])
    )
    matrix = metric.pairwise(x)
    expected = np.array(
        [[_reference(name, x[:, i], x[:, j]) for j in range(6)] for i in range(6)]
    )
    np.testing.assert_allclose(matrix.values, expected, atol=1e-12)
    assert matrix.symmetric and (matrix.n_shared == 25).all()


@pytest.mark.parametrize(
    ("name", "scipy_name", "rank"),
    [
        ("euclidean", "euclidean", False),
        ("cityblock", "cityblock", False),
        ("cosine_distance", "cosine", False),
        ("cosine", "cosine", False),
        ("pearson", "correlation", False),
        ("spearman", "correlation", True),
    ],
)
def test_between_rows_matches_pdist(x, name, scipy_name, rank):
    data = stats.rankdata(x, axis=1) if rank else x
    expected = squareform(pdist(data, scipy_name))
    d = Metric.get(name).between_rows(x).to_distance()
    np.testing.assert_allclose(d.values, expected, atol=1e-12)
    np.testing.assert_allclose(d.condensed(), pdist(data, scipy_name), atol=1e-12)


def test_squared_distance(x):
    r = Pearson().between_rows(x).values
    d = Pearson(squared=True).between_rows(x).to_distance().values
    np.testing.assert_allclose(d, 1 - r**2, atol=1e-12)
    assert Pearson().between_rows(x).squared().metric == Pearson(squared=True)


def test_zero_variance_is_nan():
    assert np.isnan(Pearson().value(np.ones(5), np.arange(5.0)))
    assert np.isnan(Cosine().value(np.zeros(5), np.arange(5.0)))
    matrix = Pearson().between_rows(np.vstack([np.ones(4), np.arange(4.0)]))
    assert np.isnan(matrix.values[0]).all()
    with pytest.raises(ValueError, match="NaN"):
        matrix.condensed()


def test_pairwise_skips_missing_per_pair():
    rng = np.random.default_rng(3)
    m = rng.normal(size=(40, 3))
    m[:20, 0] = np.nan
    m[25:, 2] = np.nan
    for metric in (Pearson(), Spearman(), Cosine(), Euclidean()):
        matrix = metric.pairwise(m, min_shared=2)
        ok = np.isfinite(m[:, 0]) & np.isfinite(m[:, 1])
        assert matrix.values[0, 1] == pytest.approx(metric.value(m[ok, 0], m[ok, 1]))
        assert matrix.n_shared[0, 2] == 5
        cross = metric.cross(m, m, min_shared=2)
        np.testing.assert_allclose(cross.values, matrix.values, equal_nan=True)
        assert (cross.n_shared == matrix.n_shared).all()
    assert np.isnan(Pearson().pairwise(m, min_shared=10).values[0, 2])


def test_pairwise_from_frame_labels():
    df = pl.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 1, 4, 3], "s": ["x"] * 4})
    matrix = Pearson().pairwise(df, min_shared=2)
    assert matrix.row_labels == matrix.col_labels == ["a", "b"]
    pairs = matrix.pairs()
    assert pairs.columns == ["a", "b", "value", "n_shared"]
    assert pairs.height == 3


@pytest.mark.parametrize(
    "name", ["euclidean", "cityblock", "cosine", "pearson", "spearman"]
)
def test_native_neighbors_match_precomputed(x, name):
    metric = Metric.get(name)
    rows, base = metric.native(x)
    native = kneighbors_graph(rows, n_neighbors=5, metric=base).toarray()
    d = metric.between_rows(x).to_distance().values
    pre = kneighbors_graph(d, n_neighbors=5, metric="precomputed").toarray()
    assert (native == pre).all()


@pytest.mark.parametrize(
    "name", ["cosine_squared", "pearson_squared", "spearman_squared"]
)
def test_squared_metrics_have_no_native_form(x, name):
    assert Metric.get(name).native(x) is None


def test_from_array_validates():
    with pytest.raises(ValueError, match="square"):
        MetricMatrix.from_array(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="symmetric"):
        MetricMatrix.from_array(np.array([[0.0, 1.0], [2.0, 0.0]]))
    rect = MetricMatrix.from_array(np.zeros((2, 3)), ["a", "b"], col_labels="xyz")
    assert not rect.symmetric and rect.col_labels == ["x", "y", "z"]
    sim = MetricMatrix.from_array(np.eye(2), kind="similarity")
    np.testing.assert_allclose(sim.to_distance().values, 1 - np.eye(2))


def test_align_and_for_rows():
    values = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    matrix = MetricMatrix.from_array(values, ["x", "y", "z"], key="id")
    aligned = matrix.align(["z", "x"])
    np.testing.assert_allclose(aligned.values, [[0.0, 2.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="aren't in the matrix"):
        matrix.align(["w"])
    rows = matrix.for_rows(pl.DataFrame({"id": ["y", "z", "x"]}))
    assert rows.row_labels == ["y", "z", "x"] and rows.values[0, 1] == 3.0
    with pytest.raises(ValueError, match="keyed by 'id'"):
        matrix.for_rows(pl.DataFrame({"other": ["x"]}))
    unkeyed = MetricMatrix.from_array(values)
    with pytest.raises(ValueError, match="3 rows but the data 2"):
        unkeyed.for_rows(pl.DataFrame({"a": [1, 2]}))


def test_save_load_round_trip(tmp_path, x):
    matrix = Spearman(squared=True).between_rows(
        pl.DataFrame({"id": [f"v{i}" for i in range(25)]}).with_columns(
            pl.Series(f"f{j}", x[:, j]) for j in range(6)
        ),
        key="id",
    )
    path = tmp_path / "d" / "m.parquet"
    assert matrix.save(path) is matrix
    loaded = MetricMatrix.load(path)
    assert loaded.metric == Spearman(squared=True)
    assert loaded.key == "id" and loaded.symmetric
    assert loaded.row_labels == matrix.row_labels == loaded.col_labels
    np.testing.assert_allclose(loaded.values, matrix.values)

    custom = MetricMatrix.from_array(np.zeros((2, 2)), ["a", "b"], label="Mine")
    custom.save(tmp_path / "c.parquet")
    again = MetricMatrix.load(tmp_path / "c.parquet")
    assert again.metric.label == "Mine" and again.metric.kind == "distance"


def test_from_frame():
    df = pl.DataFrame({"id": ["a", "b"], "a": [0.0, 1.0], "b": [1.0, 0.0]})
    matrix = MetricMatrix.from_frame(df, "id")
    assert matrix.symmetric and matrix.row_labels == ["a", "b"]
    rect = MetricMatrix.from_frame(df.rename({"b": "c"}), "id")
    assert not rect.symmetric and rect.col_labels == ["a", "c"]


def test_profiles_distances(profiles):
    p = fb.Profiles(profiles).drop_nonfinite()
    d = p.distances("pearson")
    assert d.key == "meta_aa_changes" and d.shape == (300, 300)
    x = p.df.select(p.values).to_numpy()
    np.testing.assert_allclose(
        d.to_distance().values, squareform(pdist(x, "correlation")), atol=1e-12
    )
    np.testing.assert_allclose(p.distances("euclidean").values, cdist(x, x), atol=1e-12)
