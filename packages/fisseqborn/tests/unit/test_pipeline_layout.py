"""Reading either pipeline's run through fisseq_common.layout."""

import pathlib

import polars as pl
import pytest

import fisseqborn as fb
from fisseq_common.layout import DataPipelineLayout, EmbeddingsPipelineLayout
from fisseqborn import _pipeline

VARIANTS = ["A1C", "A1D", "A1E"]


def _write(path: pathlib.Path, df: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


@pytest.fixture
def emb_run(tmp_path) -> pathlib.Path:
    """An embeddings-pipeline run: per-method aggregates and output.parquet for the
    Cell-DINO track, median aggregates for the CellProfiler track."""
    layout = EmbeddingsPipelineLayout()
    cp_layout = EmbeddingsPipelineLayout("cp_features")
    for i, batch in enumerate(["b1", "b2"]):
        for method, values in (
            ("median", [1.0 + i, 2.0, 3.0]),
            ("KS", [0.1, 0.2, 0.3]),
        ):
            _write(
                tmp_path / layout.aggregate(batch, method),
                pl.DataFrame(
                    {"meta_aa_changes": VARIANTS, f"emb_0000_{method}": values}
                ),
            )
        _write(
            tmp_path / layout.selected(batch),
            pl.DataFrame(
                {
                    "meta_aa_changes": VARIANTS,
                    "emb_0000_median": [1.0 + i, 2.0, 3.0],
                    "meta_num_cells": [5, 6, 7],
                }
            ),
        )
        _write(
            tmp_path / cp_layout.aggregate(batch, "median"),
            pl.DataFrame({"meta_aa_changes": VARIANTS, "Cells_Area_median": [i, 1, 2]}),
        )
        _write(
            tmp_path / layout.ovwt_results(batch),
            pl.DataFrame(
                {"meta_aa_changes": VARIANTS, "auroc_pooled": [0.5, 0.6, 0.7]}
            ),
        )
        (tmp_path / layout.metadata(batch)).parent.mkdir(parents=True)
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


def test_cp_features_track(emb_run):
    df = fb.Profiles.from_pipeline(emb_run, track="cp_features").collect()
    assert [c for c in df.columns if not c.startswith("meta_")] == ["Cells_Area_median"]
    with pytest.raises(ValueError, match="no output.parquet"):
        fb.Profiles.from_pipeline(emb_run, track="cp_features", metadata=True)


def test_embeddings_ovwt_scores(emb_run):
    scores = fb.OvwtScores.from_pipeline(emb_run).collect()
    assert scores.height == 6


def test_explicit_layout_overrides_detection(emb_run):
    assert _pipeline.source(emb_run, layout="data").layout == DataPipelineLayout()


def test_run_without_markers_reads_as_data_pipeline(tmp_path):
    (tmp_path / "feature_select_batchwise" / "b1").mkdir(parents=True)
    assert _pipeline.source(tmp_path).layout == DataPipelineLayout()


def test_bad_layout_raises(tmp_path):
    with pytest.raises(ValueError, match="layout must be"):
        _pipeline.source(tmp_path, layout="nope").layout
