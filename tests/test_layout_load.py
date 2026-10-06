"""Both pipelines' per-experiment outputs are where ``fisseq_common.layout`` says, and load in
fisseqborn through it.

The runs are the saved reference outputs of each pipeline's integration fixture
(``tests/reference``), which ``test_reference_outputs`` keeps equal to a fresh run.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

import fisseqborn as fb
from _fixtures import REFERENCE_DIR
from fisseq_common.layout import (
    DataPipelineLayout,
    EmbeddingsPipelineLayout,
    PipelineLayout,
    detect,
)

DATA_RUN = REFERENCE_DIR / "data"
EMB_RUN = REFERENCE_DIR / "embeddings_two"
EMB_METHODS = ["median", "KS", "AUROC"]

# Placeholders for a layout's arguments, turned into one-path-segment regex groups.
_ARGS = {
    "batch": "\x00b\x00",
    "rep": "\x00r\x00",
    "half": "\x00h\x00",
    "method": "\x00m\x00",
}


def _patterns(layout: PipelineLayout) -> list[re.Pattern[str]]:
    """A regex per path the layout knows, every argument matching one path segment."""
    b, r, h, m = (_ARGS[k] for k in ("batch", "rep", "half", "method"))
    paths = [
        layout.filtered_cells(b),
        layout.barcode_counts(b),
        layout.variants_per_barcode(b),
        layout.filtered_keys(b),
        layout.normalizer(b),
        layout.ovwt_results(b),
        layout.ovwt_cell_scores(b),
        layout.ovwt_models(b),
        layout.aggregate(b, m),
        layout.passthrough_aggregate(b, m),
        layout.blocklist(b),
        layout.method_blocklist(b, m),
        layout.split(b, r, h),
        layout.half_aggregate(b, r, h, m),
        layout.correlations(b, r, m),
        layout.selected(b),
    ]
    if isinstance(layout, DataPipelineLayout):
        paths += [layout.input(b), layout.pca_components(b)]
    else:
        paths += [
            layout.cell_table(b),
            layout.tiles(b),
            layout.metadata(b),
            layout.features(b),
            layout.aggregate_with_passthrough(b),
        ]
    patterns = []
    for path in paths:
        if path is None:
            continue
        regex = re.escape(path)
        for placeholder in _ARGS.values():
            regex = regex.replace(re.escape(placeholder), "[^/]+")
        patterns.append(re.compile(regex))
    return patterns


def _published(run: Path) -> list[str]:
    return sorted(
        p.relative_to(run).as_posix()
        for p in run.rglob("*")
        if p.is_file()
        and p.name != "MANIFEST.json"
        and "global" not in p.relative_to(run).parts
    )


@pytest.mark.parametrize(
    "run, layouts",
    [
        (DATA_RUN, [DataPipelineLayout()]),
        (
            EMB_RUN,
            [
                EmbeddingsPipelineLayout("embeddings"),
                EmbeddingsPipelineLayout("cp_features"),
            ],
        ),
    ],
    ids=["data", "embeddings"],
)
def test_every_published_file_is_in_the_layout(run, layouts):
    patterns = [p for layout in layouts for p in _patterns(layout)]
    unknown = [f for f in _published(run) if not any(p.fullmatch(f) for p in patterns)]
    assert unknown == []


def test_detect():
    assert detect(DATA_RUN) == DataPipelineLayout()
    assert detect(EMB_RUN) == EmbeddingsPipelineLayout("embeddings")
    assert detect(EMB_RUN, track="cp_features") == EmbeddingsPipelineLayout(
        "cp_features"
    )


# --- data pipeline ----------------------------------------------------------------------


def test_data_profiles_load():
    profiles = fb.Profiles.from_pipeline(
        DATA_RUN, types=["median", "KS"], passthrough=[], metadata=["meta_num_cells"]
    )
    df = profiles.collect()
    assert sorted(df["meta_experiment"].unique()) == ["batch1", "batch2"]
    expected = pl.read_parquet(
        DATA_RUN / DataPipelineLayout().aggregate("batch1", "KS")
    )
    batch1 = df.filter(pl.col("meta_experiment") == "batch1")
    ks_cols = [c for c in expected.columns if c.endswith("_KS")]
    assert set(ks_cols) <= set(df.columns)
    assert batch1.height == expected.height
    assert batch1["meta_num_cells"].null_count() == 0


def test_data_blocklists_and_ovwt_load():
    blocklists = fb.Blocklists.from_pipeline(DATA_RUN).collect()
    assert sorted(blocklists["meta_feature_type"].unique()) == sorted(
        p.stem
        for p in (
            DATA_RUN / "feature_select_batchwise" / "batch1" / "blocklists"
        ).iterdir()
    )
    scores = fb.OvwtScores.from_pipeline(DATA_RUN).collect()
    assert {"auroc_pooled", "auroc_median_barcode"} <= set(scores.columns)
    assert sorted(scores["meta_experiment"].unique()) == ["batch1", "batch2"]


# --- embeddings pipeline, Cell-DINO track --------------------------------------------------


def test_embeddings_profiles_load():
    profiles = fb.Profiles.from_pipeline(
        EMB_RUN,
        types=EMB_METHODS,
        passthrough=["KSnegLogP"],
        metadata=["meta_num_cells"],
    )
    df = profiles.collect()
    aggregate = pl.read_parquet(
        EMB_RUN / EmbeddingsPipelineLayout().aggregate("batch1")
    )
    for method in [*EMB_METHODS, "KSnegLogP"]:
        assert any(c.endswith(f"_{method}") for c in df.columns), method
    feature_cols = [c for c in aggregate.columns if not c.startswith("meta_")]
    assert set(feature_cols) <= set(df.columns)
    batch1 = df.filter(pl.col("meta_experiment") == "batch1").sort("meta_aa_changes")
    assert (
        batch1["emb_0000_KS"].to_list()
        == aggregate.sort("meta_aa_changes")["emb_0000_KS"].to_list()
    )
    assert batch1["meta_num_cells"].null_count() == 0


def test_embeddings_profiles_select_one_method():
    df = fb.Profiles.from_pipeline(EMB_RUN, types=["KS"]).collect()
    features = [c for c in df.columns if not c.startswith("meta_")]
    assert features and all(c.endswith("_KS") for c in features)


def test_embeddings_blocklists_and_ovwt_load():
    blocklists = fb.Blocklists.from_pipeline(EMB_RUN).collect()
    assert sorted(blocklists["meta_feature_type"].unique()) == sorted(EMB_METHODS)
    scores = fb.OvwtScores.from_pipeline(EMB_RUN).collect()
    assert sorted(scores["meta_experiment"].unique()) == ["batch1", "batch2"]


# --- embeddings pipeline, CellProfiler track ------------------------------------------------


def test_cp_features_profiles_load():
    df = fb.Profiles.from_pipeline(
        EMB_RUN, types=["median"], track="cp_features"
    ).collect()
    aggregate = pl.read_parquet(
        EMB_RUN / EmbeddingsPipelineLayout("cp_features").aggregate("batch1")
    )
    features = [c for c in aggregate.columns if not c.startswith("meta_")]
    # A median-only run writes bare feature columns.
    assert features and set(features) <= set(df.columns)
    assert sorted(df["meta_experiment"].unique()) == ["batch1", "batch2"]


def test_cp_features_ovwt_load():
    scores = fb.OvwtScores.from_pipeline(EMB_RUN, track="cp_features").collect()
    expected = pl.read_parquet(
        EMB_RUN / EmbeddingsPipelineLayout("cp_features").ovwt_results("batch1")
    )
    assert (
        scores.filter(pl.col("meta_experiment") == "batch1").height == expected.height
    )


def test_cp_features_track_has_no_blocklists():
    with pytest.raises(ValueError, match="no blocklists"):
        fb.Blocklists.from_pipeline(EMB_RUN, track="cp_features")
