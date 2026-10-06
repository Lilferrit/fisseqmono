"""CORRELATE_FEATURES -- per-dimension Pearson r between two split halves."""

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
import scipy.stats
from omegaconf import OmegaConf

import fisseq_common.stages.correlatefeatures as m

LABEL = "meta_aa_changes"


def _half(values: "dict[str, list[float]]", labels: "list[str]") -> pl.DataFrame:
    return pl.DataFrame({LABEL: labels, **values})


def test_matches_scipy_pearsonr() -> None:
    rng = np.random.default_rng(0)
    labels = [f"V{i}" for i in range(20)]
    a = rng.normal(size=20)
    b = 0.7 * a + rng.normal(size=20) * 0.3

    result = m.compute_feature_correlations(
        _half({"emb_0000_KS": a.tolist()}, labels),
        _half({"emb_0000_KS": b.tolist()}, labels),
        LABEL,
    )

    expected = scipy.stats.pearsonr(a, b).statistic
    assert result["r"][0] == pytest.approx(expected)
    assert result["r_squared"][0] == pytest.approx(expected**2)


def test_aligns_on_the_label_not_row_order() -> None:
    """The two halves are aggregated independently, so their row order is
    not guaranteed to agree -- the join on the label is what makes the
    correlation meaningful rather than a comparison of unrelated pairs."""
    labels = ["A", "B", "C", "D"]
    a = [1.0, 2.0, 3.0, 4.0]

    in_order = m.compute_feature_correlations(
        _half({"emb_0000_KS": a}, labels),
        _half({"emb_0000_KS": a}, labels),
        LABEL,
    )
    reversed_ = m.compute_feature_correlations(
        _half({"emb_0000_KS": a}, labels),
        _half({"emb_0000_KS": a[::-1]}, labels[::-1]),
        LABEL,
    )

    assert in_order["r"][0] == pytest.approx(1.0)
    assert reversed_["r"][0] == pytest.approx(1.0)


def test_perfect_and_anticorrelated_dimensions() -> None:
    labels = ["A", "B", "C", "D"]
    a = [1.0, 2.0, 3.0, 4.0]
    result = m.compute_feature_correlations(
        _half({"good": a, "bad": a}, labels),
        _half({"good": a, "bad": a[::-1]}, labels),
        LABEL,
    ).sort("feature")

    assert result["feature"].to_list() == ["bad", "good"]
    assert result["r"].to_list() == pytest.approx([-1.0, 1.0])


def test_constant_dimension_gives_null_r() -> None:
    """Zero variance in a half -- Polars' corr is null, and BLOCKLIST reads
    that as 'not reproducible', which is the intended verdict."""
    labels = ["A", "B", "C"]
    result = m.compute_feature_correlations(
        _half({"flat": [1.0, 1.0, 1.0]}, labels),
        _half({"flat": [1.0, 2.0, 3.0]}, labels),
        LABEL,
    )
    assert result["r"][0] is None
    assert result["r_squared"][0] is None


def test_uses_feature_selector_so_suffixed_columns_are_covered() -> None:
    """AGGREGATE_HALF writes emb_0000_KS, which EMBEDDING_SELECTOR's
    ^emb_\\d+$ would NOT match -- getting this wrong would silently
    correlate nothing."""
    labels = ["A", "B", "C"]
    values = [1.0, 2.0, 3.0]
    result = m.compute_feature_correlations(
        _half({"emb_0000_KS": values, "emb_0001_AUROC": values}, labels),
        _half({"emb_0000_KS": values, "emb_0001_AUROC": values}, labels),
        LABEL,
    )
    assert sorted(result["feature"].to_list()) == ["emb_0000_KS", "emb_0001_AUROC"]


def test_meta_columns_are_not_correlated() -> None:
    labels = ["A", "B", "C"]
    df = _half({"emb_0000_KS": [1.0, 2.0, 3.0]}, labels).with_columns(
        meta_num_cells=pl.Series([10, 20, 30])
    )
    result = m.compute_feature_correlations(df, df, LABEL)
    assert result["feature"].to_list() == ["emb_0000_KS"]


def test_no_feature_columns_returns_empty_table() -> None:
    df = pl.DataFrame({LABEL: ["A", "B"]})
    result = m.compute_feature_correlations(df, df, LABEL)
    assert result.height == 0
    assert result.columns == ["feature", "r", "r_squared"]


def test_main_writes_the_correlations_of_two_half_files(tmp_path) -> None:
    """The entry point, python -m fisseq_common.stages.correlatefeatures."""
    labels = ["A", "B", "C"]
    df1 = _half({"f1_mean": [1.0, 2.0, 4.0]}, labels)
    df2 = _half({"f1_mean": [2.0, 5.0, 1.0]}, labels)
    p1, p2 = tmp_path / "half1.parquet", tmp_path / "half2.parquet"
    df1.write_parquet(p1)
    df2.write_parquet(p2)
    cfg = m.CorrelateFeaturesParams(
        output_dir=str(tmp_path / "out"),
        half1_file=str(p1),
        half2_file=str(p2),
        output_name="mean",
    )
    with patch("fisseq_common.stages.config.setup_logging"):
        m.main.__wrapped__(OmegaConf.structured(cfg))

    result = pl.read_parquet(tmp_path / "out" / "mean.parquet")
    assert result.equals(m.compute_feature_correlations(df1, df2, LABEL))
