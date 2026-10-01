import numpy as np
import polars as pl
import pytest
from conftest import PIPELINE_BATCHES, PIPELINE_FEATURES, PIPELINE_VARIANTS

import fisseqborn as fb

AREA = "AreaShape_Area_median"


@pytest.fixture
def raw(pipeline_dir) -> fb.Profiles:
    return fb.Profiles.from_pipeline(
        pipeline_dir, types=["median", "KS"], passthrough=["KSnegLogP"]
    ).variant_type()


def _read(pipeline_dir, batch, folder, stat):
    path = pipeline_dir / "feature_select_batchwise" / batch / folder / f"{stat}.parquet"
    return pl.read_parquet(path)


def test_from_pipeline_tags_batches_in_natural_order(raw):
    df = raw.df
    assert df["meta_experiment"].unique(maintain_order=True).to_list() == [
        "T1_R1",
        "T2_R1",
        "T10_R1",
    ]
    assert df.height == len(PIPELINE_VARIANTS) * len(PIPELINE_BATCHES)
    assert len(raw.features) == 3 * len(PIPELINE_FEATURES)
    assert raw.passthrough == ("KSnegLogP",)
    assert raw.values == [c for c in raw.features if not c.endswith("_KSnegLogP")]


def test_from_pipeline_batches_and_missing_files(pipeline_dir):
    one = fb.Profiles.from_pipeline(pipeline_dir, batches=["T2_R1"])
    assert one.df["meta_experiment"].unique().to_list() == ["T2_R1"]
    with pytest.raises(FileNotFoundError, match="T9_R9"):
        fb.Profiles.from_pipeline(pipeline_dir, batches=["T9_R9"])
    with pytest.raises(FileNotFoundError, match="AUROC.parquet"):
        fb.Profiles.from_pipeline(pipeline_dir, types=["AUROC"])
    with pytest.raises(FileNotFoundError, match="feature_select_batchwise"):
        fb.Profiles.from_pipeline(pipeline_dir / "nope")


def test_normalize_per_batch_matches_formula(pipeline_dir, raw):
    normed = raw.normalize(by="meta_experiment").df
    for batch in PIPELINE_BATCHES:
        values = _read(pipeline_dir, batch, "aggregates", "median")[AREA].to_numpy()
        controls = values[:3]  # A1A, C2C and G6G are the synonymous variants
        expected = (values - controls.mean()) / controls.std(ddof=1)
        got = normed.filter(pl.col("meta_experiment") == batch)[AREA].to_numpy()
        np.testing.assert_allclose(got, expected)


def test_normalize_zero_std_is_null_and_skips_passthrough(raw):
    normed = raw.normalize(by="meta_experiment").df
    assert normed["Constant_median"].is_null().all()
    assert normed["AreaShape_Area_KSnegLogP"].equals(raw.df["AreaShape_Area_KSnegLogP"])


def test_normalize_global_and_types(raw):
    df = raw.df
    only_ks = raw.normalize(types=["KS"]).df
    assert only_ks[AREA].equals(df[AREA])
    values = df["AreaShape_Area_KS"].to_numpy()
    controls = values[df["meta_is_control"].to_numpy()]
    np.testing.assert_allclose(
        only_ks["AreaShape_Area_KS"].to_numpy(),
        (values - controls.mean()) / controls.std(ddof=1),
    )


def test_normalize_requires_variant_type(pipeline_dir):
    with pytest.raises(ValueError, match="variant_type"):
        fb.Profiles.from_pipeline(pipeline_dir).normalize()


def test_median_across_batches(raw):
    per_batch = raw.df
    med = raw.median_across_batches().df
    assert med.height == len(PIPELINE_VARIANTS)
    assert med["meta_n_experiments"].to_list() == [3] * len(PIPELINE_VARIANTS)
    assert "meta_experiment" not in med.columns
    assert med["meta_variant_type"].to_list()[:3] == ["Synonymous"] * 3
    expected = per_batch.group_by("meta_aa_changes").agg(pl.col(AREA).median())
    assert med.join(expected, on="meta_aa_changes")[[AREA, f"{AREA}_right"]].pipe(
        lambda d: np.allclose(d[AREA], d[f"{AREA}_right"])
    )


def test_paired_median_keeps_pvalue_from_the_same_batch():
    rows = pl.DataFrame(
        {
            "meta_aa_changes": ["A1V"] * 4 + ["C2V"] * 2 + ["D3V"] * 2,
            "meta_experiment": ["b1", "b2", "b3", "b4", "b1", "b2", "b1", "b2"],
            "x_median": [3.0, 1.0, 2.0, 4.0, None, 5.0, None, None],
            "x_KSnegLogP": [30.0, 10.0, 20.0, 40.0, 1.0, 50.0, 7.0, 8.0],
        }
    )
    med = fb.Profiles(rows).median_across_batches(paired={"_median": "_KSnegLogP"}).df
    # A1V: lower-middle of [1, 2, 3, 4] is 2, from b3; C2V: nulls are skipped
    assert med["x_median"].to_list() == [2.0, 5.0, None]
    assert med["x_KSnegLogP"].to_list() == [20.0, 50.0, None]


def test_paired_median_breaks_ties_independently_of_batch_order():
    rows = pl.DataFrame(
        {
            "meta_aa_changes": ["A1V"] * 3,
            "meta_experiment": ["b1", "b2", "b3"],
            "x_median": [1.0, 1.0, 1.0],
            "x_KSnegLogP": [3.0, 1.0, 2.0],
        }
    )
    picks = {
        fb.Profiles(rows.sample(fraction=1, shuffle=True, seed=seed))
        .median_across_batches(paired={"_median": "_KSnegLogP"})
        .df["x_KSnegLogP"]
        .item()
        for seed in range(5)
    }
    assert picks == {2.0}


def test_keep_and_drop_features(raw):
    kept = raw.keep_features(["AreaShape_Area_median", "not_a_column"])
    assert kept.features == ["AreaShape_Area_median", "AreaShape_Area_KSnegLogP"]
    assert set(kept.meta) == set(raw.meta)
    globbed = raw.drop_features("*_CH1_*")
    assert not any("CH1" in c for c in globbed.features)
    assert len(globbed.features) == 6


def test_drop_nonfinite():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "C2V"],
            "ok": [1.0, 2.0],
            "nan": [1.0, float("nan")],
            "inf": [float("inf"), 1.0],
            "null": [1.0, None],
        }
    )
    assert fb.Profiles(df).drop_nonfinite().features == ["ok"]


def _selectable() -> fb.Profiles:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 40))
    df = pl.DataFrame(
        {
            "meta_aa_changes": [f"A{i}V" for i in range(40)],
            "A_median": a,
            "A_KSnegLogP": rng.uniform(size=40),
            "Twin_median": 2 * a + 1 + 0.2 * rng.normal(size=40),
            "Twin_KSnegLogP": rng.uniform(size=40),
            "B_median": b,
            "Constant_median": np.ones(40),
        }
    )
    return fb.Profiles(df, passthrough=["KSnegLogP"])


def test_feature_select_drops_correlated_and_constant_features():
    profiles = _selectable()
    selected = profiles.feature_select()
    assert selected.values == ["A_median", "B_median"]
    assert selected.features == ["A_median", "A_KSnegLogP", "B_median"]
    assert selected.meta == profiles.meta
    assert selected.df["A_median"].equals(profiles.df["A_median"])
    assert selected.feature_selection.to_dict(as_series=False) == {
        "feature": ["A_median", "Twin_median", "B_median", "Constant_median"],
        "kept": [True, False, True, False],
    }
    assert selected.filter(pl.lit(True)).feature_selection is None
    assert profiles.feature_selection is None


def test_feature_select_operations_and_thresholds():
    profiles = _selectable()
    only_variance = profiles.feature_select(["variance_threshold"])
    assert only_variance.values == ["A_median", "Twin_median", "B_median"]
    varying = profiles.drop_features("Constant_median")
    assert "Twin_median" not in varying.feature_select(["correlation_threshold"]).values
    loose = varying.feature_select(["correlation_threshold"], corr_threshold=0.999)
    assert "Twin_median" in loose.values
    with pytest.raises(ValueError, match="not supported"):
        profiles.feature_select(["no_such_operation"])


def test_impact_score_matches_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(6, 4))
    df = pl.DataFrame(
        {"meta_aa_changes": ["A1A", "C2C", "G3G", "D4V", "E5K", "F6L"]}
        | {f"f{i}": x[:, i] for i in range(4)}
    )
    scored = fb.Profiles(df).variant_type().impact_score().df["meta_impact_score"].to_numpy()
    ref = np.median(x[:3], axis=0)
    cosine = x @ ref / (np.linalg.norm(x, axis=1) * np.linalg.norm(ref))
    np.testing.assert_allclose(scored, (1 - cosine) / 2)


def test_impact_score_skips_missing_values_per_row():
    df = pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "C2C", "D3V"],
            "f0": [1.0, 1.0, 1.0],
            "f1": [0.0, 0.0, float("nan")],
        }
    )
    scored = fb.Profiles(df).variant_type().impact_score().df["meta_impact_score"]
    assert scored.to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_pca_and_pca_reduce(raw):
    med = raw.normalize(by="meta_experiment").median_across_batches().drop_nonfinite()
    with_pcs = med.pca(n_components=2)
    assert {"meta_pc_1", "meta_pc_2"} <= set(with_pcs.columns)
    assert with_pcs.pca_loadings.columns[:2] == ["component", "explained_variance_ratio"]
    assert with_pcs.features == med.features

    reduced = med.pca_reduce(variance=0.5)
    ratios = reduced.pca_loadings["explained_variance_ratio"].to_numpy()
    assert reduced.features == [f"X_{i}" for i in range(len(ratios))]
    assert ratios.sum() >= 0.5 and (len(ratios) == 1 or ratios[:-1].sum() < 0.5)
    assert set(reduced.meta) == set(med.meta)
    assert med.pca_reduce(noise_floor=True).features[0] == "X_0"


def test_embeddings_require_finite_values(raw):
    with pytest.raises(ValueError, match="drop_nonfinite"):
        raw.normalize(by="meta_experiment").pca()


def test_kmeans_cluster(profiles):
    clustered = fb.Profiles(profiles).cluster("kmeans", n_clusters=3).df
    assert clustered["meta_cluster_idx"].dtype == pl.String
    assert set(clustered["meta_cluster_idx"]) == {"0", "1", "2"}
    with pytest.raises(ValueError, match="n_clusters"):
        fb.Profiles(profiles).cluster("kmeans")


def test_leiden_cluster(profiles):
    pytest.importorskip("leidenalg")
    clustered = fb.Profiles(profiles).cluster(n_neighbors=10).df
    assert clustered["meta_cluster_idx"].n_unique() > 1
    with pytest.raises(ValueError, match="resolution"):
        fb.Profiles(profiles).cluster(n_clusters=3)


def test_umap(profiles):
    pytest.importorskip("umap")
    embedded = fb.Profiles(profiles.drop("meta_notebook_umap_1", "meta_notebook_umap_2")).umap(
        n_neighbors=10
    )
    assert {"meta_notebook_umap_1", "meta_notebook_umap_2"} <= set(embedded.columns)


def test_feature_info():
    info = fb.Profiles(
        pl.DataFrame(
            schema={
                "meta_aa_changes": pl.String,
                "AreaShape_Area_median": pl.Float64,
                "Mean_NucleiExpanded_Texture_SumAverage_CH1_20_01_256_KS": pl.Float64,
                "Correlation_RWC_CH1_CH3_KSnegLogP": pl.Float64,
                "X_0": pl.Float64,
            }
        )
    ).feature_info()
    assert info.rows() == [
        ("AreaShape_Area_median", "AreaShape_Area", "median", None, "AreaShape", None),
        (
            "Mean_NucleiExpanded_Texture_SumAverage_CH1_20_01_256_KS",
            "Mean_NucleiExpanded_Texture_SumAverage_CH1_20_01_256",
            "KS",
            "NucleiExpanded",
            "Texture",
            "CH1",
        ),
        (
            "Correlation_RWC_CH1_CH3_KSnegLogP",
            "Correlation_RWC_CH1_CH3",
            "KSnegLogP",
            None,
            "Correlation",
            "CH1+CH3",
        ),
        ("X_0", "X_0", None, None, None, None),
    ]


def test_volcano_from_profiles(raw):
    med = raw.normalize(by="meta_experiment").median_across_batches(
        paired={"_median": "_KSnegLogP"}
    )
    plot = fb.VolcanoPlot.from_wide(med)
    assert set(plot.data["feature"]) == set(PIPELINE_FEATURES)


def test_from_global(tmp_path):
    path = tmp_path / "global" / "main" / "feature_select"
    path.mkdir(parents=True)
    pl.DataFrame({"meta_aa_changes": ["A1A"], "f_median": [1.0]}).write_parquet(
        path / "aggregate.parquet"
    )
    assert fb.Profiles.from_global(tmp_path, "main").features == ["f_median"]
