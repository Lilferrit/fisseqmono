"""Reading either pipeline's run through fisseq_common.layout."""

import pathlib

import polars as pl
import pytest

import fisseqborn as fb
from fisseq_common.layout import DataPipelineLayout, EmbeddingsPipelineLayout
from fisseqborn import _pipeline
from fisseqborn.profiles import _method_columns

VARIANTS = ["A1C", "A1D", "A1E"]


def _write(path: pathlib.Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


@pytest.fixture
def emb_run(tmp_path) -> pathlib.Path:
    """An embeddings-pipeline run: one aggregate.parquet holding median and KS columns."""
    layout = EmbeddingsPipelineLayout()
    for i, batch in enumerate(["b1", "b2"]):
        _write(
            tmp_path / layout.aggregate(batch),
            pl.DataFrame(
                {
                    "meta_aa_changes": VARIANTS,
                    "emb_0000_median": [1.0 + i, 2.0, 3.0],
                    "emb_0000_KS": [0.1, 0.2, 0.3],
                    "meta_num_cells": [5, 6, 7],
                }
            ),
        )
        _write(
            tmp_path / layout.ovwt_results(batch),
            pl.DataFrame(
                {"meta_aa_changes": VARIANTS, "auroc_pooled": [0.5, 0.6, 0.7]}
            ),
        )
        (tmp_path / layout.filtered_keys(batch)).parent.mkdir(parents=True)
    return tmp_path


def test_embeddings_run_is_detected(emb_run):
    assert _pipeline.source(emb_run).layout == EmbeddingsPipelineLayout()


def test_embeddings_profiles_take_one_methods_columns(emb_run):
    df = fb.Profiles.from_pipeline(emb_run, types=["median"], metadata=True).collect()
    assert [c for c in df.columns if not c.startswith("meta_")] == ["emb_0000_median"]
    assert df.filter(pl.col("meta_experiment") == "b2")["meta_num_cells"].to_list() == [
        5,
        6,
        7,
    ]
    both = fb.Profiles.from_pipeline(emb_run, types=["median", "KS"]).collect()
    assert {"emb_0000_median", "emb_0000_KS"} <= set(both.columns)


def test_embeddings_ovwt_scores(emb_run):
    scores = fb.OvwtScores.from_pipeline(emb_run).collect()
    assert scores.height == 6


def test_explicit_layout_overrides_detection(emb_run):
    with pytest.raises(FileNotFoundError, match="aggregates"):
        fb.Profiles.from_pipeline(emb_run, layout="data").collect()


def test_run_without_markers_reads_as_data_pipeline(tmp_path):
    (tmp_path / "feature_select_batchwise" / "b1").mkdir(parents=True)
    assert _pipeline.source(tmp_path).layout == DataPipelineLayout()


def test_bad_layout_raises(tmp_path):
    with pytest.raises(ValueError, match="layout must be"):
        _pipeline.source(tmp_path, layout="nope").layout


def test_method_columns():
    path = pathlib.Path("aggregate.parquet")
    assert _method_columns(["meta_aa_changes", "f_KS", "f_median"], ["KS"], path) == [
        "f_KS"
    ]
    # A median-only run's bare columns.
    assert _method_columns(["meta_aa_changes", "f", "g"], ["median"], path) == [
        "f",
        "g",
    ]
    with pytest.raises(ValueError, match="no 'AUROC' columns"):
        _method_columns(["meta_aa_changes", "f_KS"], ["AUROC"], path)
    with pytest.raises(ValueError, match="no 'median' columns"):
        _method_columns(["meta_aa_changes", "f_KS"], ["median"], path)
