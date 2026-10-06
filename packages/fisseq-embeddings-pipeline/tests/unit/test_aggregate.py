"""Tests for AGGREGATE_EMBEDDINGS's aggregator classes.

Covers BaseAggregator/ReferenceBasedAggregator/MeanAggregator/
MedianAggregator/KSAggregator/AUROCAggregator/KSNegLogPValueAggregator/
AUROCNegLogPValueAggregator and the _AGGREGATORS
registry, aggregate_embeddings() combination/backward-compat, and the
Hydra `main()` CLI end-to-end. Ground-truth numerical tests are adapted
from fisseq-data-pipeline's tests/unit/test_aggregate.py, retargeted from
its CellProfiler-style `f1`/`f2` feature columns to this pipeline's
`emb_0000`/`emb_0001` embedding columns (the only columns
EMBEDDING_SELECTOR matches).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fisseq_embeddings_pipeline.aggregate as m
from fisseq_embeddings_pipeline.filter import JOIN_KEYS, filter_and_fit_normalizer

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
            "emb_0000": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 13.0, 14.0],
            "emb_0001": [5.0, 7.0, 6.0, 6.0, 1.0, 3.0, 0.0, 2.0],
        }
    )


@pytest.fixture
def simple_df() -> pl.DataFrame:
    """Two variant groups with no control rows."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B", "B"],
            "meta_is_control": [False, False, False, False, False, False],
            "emb_0000": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "emb_0001": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        }
    )


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
            "emb_0000": values,
        }
    )


def _get_row(df: pl.DataFrame, label: str) -> dict:
    return df.filter(pl.col("meta_aa_changes") == label).to_dicts().pop()


def _group_and_ref(df: pl.DataFrame, label: str) -> tuple[list[float], list[float]]:
    ref = df.filter(pl.col("meta_is_control"))["emb_0000"].to_list()
    group = df.filter(pl.col("meta_aa_changes") == label)["emb_0000"].to_list()
    return group, ref


# ---------------------------------------------------------------------------
# aggregate_embeddings() -- combination & backward-compat
# ---------------------------------------------------------------------------


@pytest.fixture
def agg_embeddings_lf() -> pl.LazyFrame:
    """A1A is control (excluded from output); M1K and WT are reportable
    variants. meta_barcode/meta_batch are present so get_aggregate_meta_data
    produces its optional columns too."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A1A", "M1K", "M1K", "M1K", "WT", "WT"],
            "meta_is_control": [True, True, False, False, False, False, False],
            "meta_barcode": ["bc0", "bc1", "bc2", "bc2", "bc3", "bc4", "bc4"],
            "meta_batch": ["batch1"] * 7,
            "emb_0000": [0.0, 1.0, 10.0, 11.0, 12.0, 20.0, 21.0],
            "emb_0001": [0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    ).lazy()


# ---------------------------------------------------------------------------
# feature_selector -- CellProfiler-shaped columns via FEATURE_SELECTOR,
# reused (unforked) by AGGREGATE_CP_FEATURES (aggregate_cp_features.py).
# ---------------------------------------------------------------------------


@pytest.fixture
def cp_style_df() -> pl.DataFrame:
    """Same shape as simple_df, but with CellProfiler-style feature-column
    names (uppercase, underscore-separated) instead of emb_%04d -- these
    are NOT matched by EMBEDDING_SELECTOR, only by FEATURE_SELECTOR."""
    return pl.DataFrame(
        {
            "meta_aa_changes": ["A", "A", "A", "B", "B", "B"],
            "meta_is_control": [False, False, False, False, False, False],
            "Cells_AreaShape_Area": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "Cells_Intensity_MeanIntensity_DNA": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        }
    )


# ---------------------------------------------------------------------------
# main() -- CLI end-to-end (subprocess, mirroring test_filter.py's pattern)
# ---------------------------------------------------------------------------


def _write_cli_fixture(tmp_path: Path) -> "tuple[Path, Path, Path]":
    """Build embeddings.parquet/filtered_keys.parquet/normalizer.parquet the
    way FILTER_EMBEDDINGS would, for a realistic end-to-end input
    to AGGREGATE_EMBEDDINGS's CLI. A1A is synonymous+untagged (control);
    M1K/WT are reportable variants; every cell passes QC."""
    embeddings_df = pl.DataFrame(
        {
            "meta_batch": ["batch1"] * 7,
            "meta_well": ["well1"] * 7,
            "meta_tile": ["tile0x0y"] * 7,
            "meta_cell_index": list(range(7)),
            "meta_barcode": [f"bc{i}" for i in range(7)],
            "meta_aa_changes": ["A1A", "A1A", "M1K", "M1K", "M1K", "WT", "WT"],
            "meta_edit_distance": [0] * 7,
            "emb_0000": [0.0, 1.0, 10.0, 11.0, 12.0, 20.0, 21.0],
            "emb_0001": [0.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )
    qc_passed_df = embeddings_df.select(JOIN_KEYS)

    embeddings_path = tmp_path / "embeddings.parquet"
    embeddings_df.write_parquet(embeddings_path)

    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        embeddings_df.lazy(), qc_passed_df.lazy(), "meta_aa_changes"
    )
    filtered_keys_path = tmp_path / "filtered_keys.parquet"
    filtered_keys_lf.collect().write_parquet(filtered_keys_path)
    normalizer_path = tmp_path / "normalizer.parquet"
    normalizer.save(normalizer_path)

    return embeddings_path, filtered_keys_path, normalizer_path


def _run_aggregate(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "fisseq_embeddings_pipeline.aggregate", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )


def test_main_runs_end_to_end_via_cli_default(tmp_path: Path) -> None:
    """AggregateEmbeddingsConfig's default is `["median", "KS", "AUROC"]`
    (not bare `["median"]`) -- so the CLI's own default run produces
    suffixed columns for all three methods, not bare `emb_0000`. See
    test_main_runs_end_to_end_via_cli_explicit_median_only below for the
    bare-column case."""
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_aggregate(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "aggregate.parquet")
    assert {"emb_0000_median", "emb_0000_KS", "emb_0000_AUROC"}.issubset(
        set(agg.columns)
    )
    assert "emb_0000" not in agg.columns
    assert "A1A" not in agg["meta_aa_changes"].to_list()
    assert set(agg["meta_aa_changes"].to_list()) == {"M1K", "WT"}


def test_main_runs_end_to_end_via_cli_explicit_median_only(tmp_path: Path) -> None:
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_aggregate(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        "aggregators=[median]",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "aggregate.parquet")
    assert "emb_0000" in agg.columns
    assert "emb_0000_median" not in agg.columns
    assert "A1A" not in agg["meta_aa_changes"].to_list()
    assert set(agg["meta_aa_changes"].to_list()) == {"M1K", "WT"}


def test_main_runs_end_to_end_via_cli_multi_method(tmp_path: Path) -> None:
    embeddings_path, filtered_keys_path, normalizer_path = _write_cli_fixture(tmp_path)
    output_dir = tmp_path / "out"

    result = _run_aggregate(
        tmp_path,
        f"output_dir={output_dir}",
        f"embeddings_file={embeddings_path}",
        f"filtered_keys_file={filtered_keys_path}",
        f"normalizer_file={normalizer_path}",
        "aggregators=[mean,median]",
    )
    assert result.returncode == 0, result.stderr

    agg = pl.read_parquet(output_dir / "aggregate.parquet")
    assert "emb_0000_mean" in agg.columns
    assert "emb_0000_median" in agg.columns
    assert "emb_0000" not in agg.columns


def test_main_is_hydra_entry_point() -> None:
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(m.main)


# ---------------------------------------------------------------------------
# AggregateEmbeddingsConfig -- dropped-fields regression: no per_barcode
# pooling option, no WT-null-bootstrap machinery, and no in-stage blocklist.
#
# Reproducibility filtering DOES exist in this pipeline now, but as the
# separate GENERATE_SPLIT -> ... -> FILTER_AGGREGATE chain, not as a field on
# this stage's config -- aggregate.py still computes every dimension and
# FILTER_AGGREGATE drops the ones the blocklist condemns. See
# test_filter_aggregate.py.
# ---------------------------------------------------------------------------


def test_aggregate_embeddings_config_default_aggregators_is_median_ks_auroc():
    cfg = m.AggregateEmbeddingsConfig(
        embeddings_file="x", filtered_keys_file="y", normalizer_file="z"
    )
    assert cfg.aggregators == ["median", "KS", "AUROC"]


def test_aggregate_config_omits_per_barcode_and_wt_null_fields():
    cfg = m.AggregateEmbeddingsConfig(
        embeddings_file="x", filtered_keys_file="y", normalizer_file="z"
    )
    for dropped in (
        "per_barcode",
        "block_list",
        "barcode_column",
        "wt_null_aggregate",
        "wt_null_blocklist",
    ):
        assert not hasattr(cfg, dropped)
