from __future__ import annotations

from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.aggregate as m
from fisseq_common.schema import IMPACT_SCORE_COL
from fisseq_data_pipeline.aggregate import AggregateConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def toy_norm_df() -> pl.DataFrame:
    """Cell-level dataset: WT cells are controls, A1B cells are variants."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "A1B", "A1B", "WT", "WT", "A1B", "A1B"],
            "meta_is_control": [True, True, False, False, True, True, False, False],
            "f1": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 13.0, 14.0],
            "f2": [5.0, 7.0, 6.0, 6.0, 1.0, 3.0, 0.0, 2.0],
        }
    )


@pytest.fixture
def simple_df() -> pl.DataFrame:
    """Two variant groups with no control rows."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B", "B"],
            "meta_is_control": [False, False, False, False, False, False],
            "f1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "f2": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        }
    )


def _get_row(df: pl.DataFrame, label: str) -> dict:
    return df.filter(pl.col("meta_aa_changes") == label).to_dicts().pop()


# ---------------------------------------------------------------------------
# KSAggregator / AUROCAggregator / QQCorrelationAggregator — native vs.
# scipy/sklearn ground truth
# ---------------------------------------------------------------------------


@pytest.fixture
def native_stats_df() -> pl.DataFrame:
    """
    Reference pool (WT, continuous) plus three variant groups exercising
    different value shapes: RANDOM (continuous, no ties), TIES (repeated
    integer values), SINGLE (one distinct value repeated).
    """
    rng = np.random.default_rng(0)
    ref_vals = rng.standard_normal(40).tolist()
    random_vals = rng.standard_normal(12).tolist()
    tie_vals = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0]
    single_vals = [5.0] * 4

    labels = (
        ["WT"] * len(ref_vals)
        + ["RANDOM"] * len(random_vals)
        + ["TIES"] * len(tie_vals)
        + ["SINGLE"] * len(single_vals)
    )
    values = ref_vals + random_vals + tie_vals + single_vals
    return pl.DataFrame(
        {
            "meta_aa_changes": labels,
            "meta_is_control": [lbl == "WT" for lbl in labels],
            "f1": values,
        }
    )


def _group_and_ref(df: pl.DataFrame, label: str) -> tuple[list[float], list[float]]:
    ref = df.filter(pl.col("meta_is_control"))["f1"].to_list()
    group = df.filter(pl.col("meta_aa_changes") == label)["f1"].to_list()
    return group, ref


def _signed_ks_stat(group: list[float], ref: list[float]) -> float:
    """Independent numpy reference for the signed KS statistic: same
    combined-sort/signed-weight construction as SignedKSAggregator, kept
    separate so it's a genuine cross-check rather than a restatement."""
    group_arr = np.asarray(group, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    n1, n2 = len(group_arr), len(ref_arr)
    combined = np.concatenate([group_arr, ref_arr])
    weights = np.concatenate([np.full(n1, 1.0 / n1), np.full(n2, -1.0 / n2)])
    order = np.argsort(combined, kind="stable")
    val_sorted = combined[order]
    w_sorted = weights[order]
    cumsum = np.cumsum(w_sorted)
    is_last = np.ones(len(val_sorted), dtype=bool)
    is_last[:-1] = val_sorted[:-1] != val_sorted[1:]
    masked = np.where(is_last, cumsum, np.nan)
    pos_max, neg_min = np.nanmax(masked), np.nanmin(masked)
    return pos_max if abs(pos_max) >= abs(neg_min) else neg_min


# ---------------------------------------------------------------------------
# Null-value fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def null_df() -> pl.DataFrame:
    """Variant group A has one null per feature; group B is all-null for f1."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B"],
            "meta_is_control": [False, False, False, False, False],
            "f1": pl.Series([1.0, None, 3.0, None, None], dtype=pl.Float64),
            "f2": pl.Series([4.0, 5.0, None, 7.0, 8.0], dtype=pl.Float64),
        }
    )


@pytest.fixture
def null_ref_df() -> pl.DataFrame:
    """Control rows: f1 has one null; f2 is entirely null."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "WT"],
            "meta_is_control": [True, True, True],
            "f1": pl.Series([1.0, None, 3.0], dtype=pl.Float64),
            "f2": pl.Series([None, None, None], dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Native aggregators — edge cases
# ---------------------------------------------------------------------------


@pytest.fixture
def single_value_group_df() -> pl.DataFrame:
    """Group A has three values; group B has exactly one (std edge case)."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B"],
            "meta_is_control": [False, False, False, False],
            "f1": [1.0, 2.0, 3.0, 5.0],
        }
    )


@pytest.fixture
def nan_inf_group_df() -> pl.DataFrame:
    """Group A mixes finite values with None, NaN, and Inf."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "A"],
            "meta_is_control": [False, False, False, False],
            "f1": pl.Series([1.0, None, float("nan"), float("inf")], dtype=pl.Float64),
        }
    ).vstack(
        pl.DataFrame(
            {
                "meta_aa_changes": ["A"],
                "meta_is_control": [False],
                "f1": pl.Series([2.0], dtype=pl.Float64),
            }
        )
    )


# ---------------------------------------------------------------------------
# feature chunking
# ---------------------------------------------------------------------------


@pytest.fixture
def chunking_df() -> pl.DataFrame:
    """
    Cell-level frame wide enough to span several chunks, with the null/NaN/
    Inf and degenerate-group cases that make the per-chunk null handling
    worth checking.
    """
    rng = np.random.default_rng(0)
    n_ctrl, n_a, n_b = 25, 20, 15
    labels = ["WT"] * n_ctrl + ["A1B"] * n_a + ["C2D"] * n_b
    data: dict[str, object] = {
        "meta_aa_changes": labels,
        "meta_is_control": [True] * n_ctrl + [False] * (n_a + n_b),
    }
    n = len(labels)
    for i in range(7):
        vals = rng.normal(size=n).tolist()
        if i == 3:  # nulls, NaN and Inf in one feature
            vals[0], vals[n_ctrl] = None, float("nan")
            vals[n_ctrl + 1] = float("inf")
        if i == 5:  # constant in one variant group -> null QQ/std
            for j in range(n_ctrl + n_a, n):
                vals[j] = 1.0
        data[f"f{i}"] = vals
    return pl.DataFrame(data)


def test_chunking_with_all_features_block_listed(chunking_df: pl.DataFrame) -> None:
    """
    An empty feature list still yields one row per label. The chunk loop has
    to run once over an empty chunk to produce that, rather than falling
    through and returning nothing.
    """
    result = m.aggregate(
        chunking_df.lazy(),
        "meta_aa_changes",
        "mean",
        block_list={f"f{i}_mean" for i in range(7)},
        feature_chunk_size=2,
    ).collect()
    assert result.columns == ["meta_aa_changes"]
    assert sorted(result["meta_aa_changes"].to_list()) == ["A1B", "C2D"]


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def make_agg_cfg(
    tmp_path,
    *,
    output_root=None,
    save_normalizer=False,
    aggregator="mean",
    block_list_file=None,
    compute_impact_score=True,
) -> OmegaConf:
    """Return a DictConfig for AggregateConfig with sensible test defaults."""
    return OmegaConf.structured(
        AggregateConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            save_normalizer=save_normalizer,
            aggregator=aggregator,
            block_list_file=block_list_file,
            compute_impact_score=compute_impact_score,
        )
    )


def write_agg_input_parquet(tmp_path, *, with_barcode: bool = False) -> None:
    """Write cell-level test parquet with WT controls, synonymous and missense variants."""
    data = {
        "meta_aa_changes": [
            "WT",
            "WT",
            "WT",
            "A1A",
            "A1A",
            "A1A",
            "A2A",
            "A2A",
            "A2A",
            "A1B",
            "A1B",
            "A1B",
        ],
        "meta_is_control": [
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        "f1": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0],
        "f2": [0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 30.0, 30.0, 30.0],
    }
    if with_barcode:
        data["meta_barcode"] = [
            "bc1",
            "bc2",
            "bc3",
            "bc1",
            "bc2",
            "bc1",
            "bc1",
            "bc1",
            "bc2",
            "bc3",
            "bc3",
            "bc3",
        ]
    pl.DataFrame(data).write_parquet(tmp_path / "input.parquet")


def test_main_creates_output_file(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    assert (tmp_path / "out" / "input.parquet").exists()


def test_main_output_contains_label_column(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_aa_changes" in result.columns


def test_main_synonymous_rows_normalized_to_zero_mean(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    # A1A and A2A are synonymous; after normalization their mean should be ~0
    syn_rows = result.filter(pl.col("meta_aa_changes").is_in(["A1A", "A2A"]))
    assert syn_rows["f1_mean"].mean() == pytest.approx(0.0, abs=1e-6)


def test_main_output_root_naming(tmp_path):
    write_agg_input_parquet(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


def test_main_saves_normalizer_when_configured(tmp_path):
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, save_normalizer=True))
    assert (tmp_path / "out" / "normalizer.parquet").exists()


# ---------------------------------------------------------------------------
# Null handling — reference-based aggregators
# ---------------------------------------------------------------------------


def _ref_based_null_df() -> pl.DataFrame:
    """Variant group A with one null; reference with one null. f2 all-null in variant."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT", "WT", "WT", "A1B", "A1B", "A1B"],
            "meta_is_control": [True, True, True, False, False, False],
            "f1": pl.Series([1.0, None, 3.0, 10.0, None, 30.0], dtype=pl.Float64),
            "f2": pl.Series([5.0, 6.0, 7.0, None, None, None], dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Block list
# ---------------------------------------------------------------------------


def test_aggregate_blocked_feature_excluded(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_mean"},
    ).collect()
    assert "f1_mean" not in result.columns


def test_aggregate_unblocked_feature_included(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_mean"},
    ).collect()
    assert "f2_mean" in result.columns


def test_aggregate_none_block_list_no_effect(simple_df: pl.DataFrame) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list=None,
    ).collect()
    assert {"f1_mean", "f2_mean"}.issubset(set(result.columns))


def test_aggregate_unknown_feature_in_block_list_ignored(
    simple_df: pl.DataFrame,
) -> None:
    result = m.aggregate(
        simple_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name="mean",
        block_list={"f1_does_not_exist"},
    ).collect()
    assert {"f1_mean", "f2_mean"}.issubset(set(result.columns))


@pytest.mark.parametrize(
    "aggregator_name",
    ["KS", "signedKS", "QQ", "AUROC", "KSnegLogP", "AUROCnegLogP"],
)
def test_aggregate_block_list_with_reference_based_aggregator(
    toy_norm_df: pl.DataFrame, aggregator_name: str
) -> None:
    result = m.aggregate(
        toy_norm_df.lazy(),
        label_col="meta_aa_changes",
        aggregator_name=aggregator_name,
        block_list={f"f1_{aggregator_name}"},
    ).collect()
    assert f"f1_{aggregator_name}" not in result.columns
    assert f"f2_{aggregator_name}" in result.columns


def test_ks_neg_log_p_stat_suffix_matches_expected_block_list_name() -> None:
    assert f"f1{m.KSNegLogPValueAggregator._stat_suffix}" == "f1_KSnegLogP"


def test_auroc_neg_log_p_stat_suffix_matches_expected_block_list_name() -> None:
    assert f"f1{m.AUROCNegLogPValueAggregator._stat_suffix}" == "f1_AUROCnegLogP"


def test_main_block_list_file_excludes_features(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    bl_path = tmp_path / "block_list.parquet"
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean"], "feature_ok": [False, True]}
    ).write_parquet(bl_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, block_list_file=str(bl_path)))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f1_mean" not in result.columns
    assert "f2_mean" in result.columns


def test_main_output_contains_meta_num_cells(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_num_cells" in result.columns


def test_main_meta_num_cells_reflects_cell_level_counts(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    counts = dict(
        zip(result["meta_aa_changes"].to_list(), result["meta_num_cells"].to_list())
    )
    assert counts["A1B"] == 3


def test_main_barcode_metadata_serializes_to_parquet(tmp_path) -> None:
    write_agg_input_parquet(tmp_path, with_barcode=True)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_barcode_num_unique" in result.columns
    assert "meta_barcode_counts" in result.columns
    assert result["meta_barcode_counts"].null_count() == 0


# ---------------------------------------------------------------------------
# get_aggregate_meta_data
# ---------------------------------------------------------------------------


@pytest.fixture
def meta_lf_no_barcode() -> pl.LazyFrame:
    """Three cells for label A, two for label B; no barcode column."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B"],
            "meta_is_control": [False, False, False, False, False],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).lazy()


@pytest.fixture
def meta_lf_with_barcode() -> pl.LazyFrame:
    """Three cells for label A (two unique barcodes), two for label B (one unique)."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B"],
            "meta_is_control": [False, False, False, False, False],
            "meta_barcode": ["bc1", "bc1", "bc2", "bc3", "bc3"],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    ).lazy()


# ---------------------------------------------------------------------------
# compute_impact_score — main() integration
# ---------------------------------------------------------------------------


def write_agg_input_parquet_asymmetric(tmp_path) -> None:
    """Cell-level data with 3 asymmetric synonymous controls.

    Three synonymous variants (A1A, A2A, A3A) with unevenly spaced feature
    values ensure the control median after Z-score normalization is non-zero,
    avoiding NaN impact scores.
    """
    pl.DataFrame(
        {
            "meta_aa_changes": (
                ["WT"] * 3 + ["A1A"] * 3 + ["A2A"] * 3 + ["A3A"] * 3 + ["A1B"] * 3
            ),
            "meta_is_control": [True] * 3 + [False] * 12,
            "f1": [0.0] * 3 + [1.0] * 3 + [2.0] * 3 + [6.0] * 3 + [20.0] * 3,
            "f2": [0.0] * 3 + [1.0] * 3 + [4.0] * 3 + [1.0] * 3 + [30.0] * 3,
        }
    ).write_parquet(tmp_path / "input.parquet")


def test_main_impact_score_column_present_by_default(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL in result.columns


def test_main_impact_score_column_absent_when_disabled(tmp_path) -> None:
    write_agg_input_parquet(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path, compute_impact_score=False))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL not in result.columns


def test_main_impact_score_values_are_finite(tmp_path) -> None:
    # Asymmetric synonymous controls guarantee a non-zero control median after
    # Z-score normalization, so impact scores are finite rather than NaN.
    write_agg_input_parquet_asymmetric(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_impact_score_in_unit_interval(tmp_path) -> None:
    write_agg_input_parquet_asymmetric(tmp_path)
    with patch("fisseq_data_pipeline.aggregate.setup_logging"):
        m.main.__wrapped__(make_agg_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    scores = result[IMPACT_SCORE_COL]
    assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# downsample_control
# ---------------------------------------------------------------------------


def _control_df(n_control: int, n_variant: int = 2) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "meta_aa_changes": ["WT"] * n_control + ["A1B"] * n_variant,
            "meta_is_control": [True] * n_control + [False] * n_variant,
            "row_id": list(range(n_control + n_variant)),
        }
    )
