import pytest

from fisseq_common.layout import (
    DataPipelineLayout,
    EmbeddingsPipelineLayout,
    detect,
    detect_from_names,
)

DATA = DataPipelineLayout()
EMB = EmbeddingsPipelineLayout()
CP = EmbeddingsPipelineLayout("cp_features")


@pytest.mark.parametrize(
    "layout, expected",
    [
        (DATA, "normalization/b1/filtered_keys.parquet"),
        (EMB, "normalization/b1/filtered_keys.parquet"),
        (CP, "normalization_cp_features/b1/filtered_keys.parquet"),
    ],
)
def test_filtered_keys(layout, expected):
    assert layout.filtered_keys("b1") == expected


@pytest.mark.parametrize(
    "layout, expected",
    [
        (DATA, "ovwt_batchwise/b1/results.parquet"),
        (EMB, "ovwt_batchwise/b1/results.parquet"),
        (CP, "ovwt_batchwise_cp_features/b1/results.parquet"),
    ],
)
def test_ovwt_results(layout, expected):
    assert layout.ovwt_results("b1") == expected


def test_qc_filter_is_shared_by_both_tracks():
    assert (
        EMB.filtered_cells("b1") == CP.filtered_cells("b1") == DATA.filtered_cells("b1")
    )
    assert DATA.filtered_cells("b1") == "qc_filter/b1/filtered_cells.parquet"


def test_aggregates_are_one_file_per_method():
    for layout in (DATA, EMB):
        assert (
            layout.aggregate("b1", "KS")
            == "feature_select_batchwise/b1/aggregates/KS.parquet"
        )
    assert (
        CP.aggregate("b1", "median")
        == "feature_select_batchwise_cp_features/b1/aggregates/median.parquet"
    )


@pytest.mark.parametrize("layout", [DATA, EMB], ids=["data", "embeddings"])
def test_reproducibility_paths(layout):
    """Both pipelines publish the bootstrap feature selection in one layout."""
    assert (
        layout.split("b", 2, 1)
        == "feature_select_batchwise/b/splits/bootstrap_2/half1.parquet"
    )
    assert (
        layout.half_aggregate("b", 2, 1, "KS")
        == "feature_select_batchwise/b/half_aggregates/bootstrap_2/KS/half1_agg.parquet"
    )
    assert (
        layout.correlations("b", 2, "KS")
        == "feature_select_batchwise/b/correlations/KS/bootstrap_2.parquet"
    )
    assert layout.blocklist("b") == "feature_select_batchwise/b/blocklist.parquet"
    assert (
        layout.method_blocklist("b", "KS")
        == "feature_select_batchwise/b/blocklists/KS.parquet"
    )
    assert (
        layout.passthrough_aggregate("b", "KSnegLogP")
        == "feature_select_batchwise/b/passthrough_aggregates/KSnegLogP.parquet"
    )
    assert layout.selected("b") == "feature_select_batchwise/b/output.parquet"


def test_cp_features_track_has_no_reproducibility_outputs():
    assert CP.blocklist("b") is None
    assert CP.method_blocklist("b", "median") is None
    assert CP.passthrough_aggregate("b", "KSnegLogP") is None
    assert CP.split("b", 1, 1) is None
    assert CP.half_aggregate("b", 1, 1, "median") is None
    assert CP.correlations("b", 1, "median") is None
    assert CP.selected("b") is None


def test_stage_dirs_by_track():
    assert EMB.stage_dir("filter") == "normalization"
    assert EMB.stage_dir("feature_select") == "feature_select_batchwise"
    assert CP.stage_dir("filter") == "normalization_cp_features"
    assert CP.stage_dir("ovwt") == "ovwt_batchwise_cp_features"
    assert CP.stage_dir("feature_select") == "feature_select_batchwise_cp_features"


def test_data_input():
    assert DATA.input("b1") == "input/b1.parquet"


def test_track_features():
    assert EMB.features("b") == "embeddings/b/embeddings.parquet"
    assert CP.features("b") == "cp_features/b/cp_features.parquet"


def test_unknown_track_and_stage_raise():
    with pytest.raises(ValueError, match="track"):
        EmbeddingsPipelineLayout("nope")
    with pytest.raises(ValueError, match="Unknown stage"):
        DATA.stage_dir("nope")


def test_layouts_compare_by_track():
    assert EmbeddingsPipelineLayout() == EMB
    assert EMB != CP


def test_detect_from_names():
    assert isinstance(
        detect_from_names(["input", "qc_filter", "normalization"]), DataPipelineLayout
    )
    assert detect_from_names(["qc_filter", "normalization", "embeddings"]) == EMB
    assert detect_from_names(["cell_metadata"], track="cp_features") == CP
    assert detect_from_names(["cp_features"]) == EMB
    with pytest.raises(ValueError, match="neither pipeline"):
        detect_from_names(["qc_filter", "normalization"])
    with pytest.raises(ValueError, match="both pipelines"):
        detect_from_names(["input", "cell_images"])


def test_detect_and_list_batches(tmp_path):
    (tmp_path / "input").mkdir()
    for batch in ("b2", "b1"):
        (tmp_path / "normalization" / batch).mkdir(parents=True)
        (tmp_path / "ovwt_batchwise" / batch).mkdir(parents=True)
    (tmp_path / "ovwt_batchwise" / "stray.txt").write_text("")
    layout = detect(tmp_path)
    assert isinstance(layout, DataPipelineLayout)
    assert layout.list_batches(tmp_path, "ovwt") == ["b1", "b2"]
    assert layout.list_batches(tmp_path, "feature_select") == []
    with pytest.raises(FileNotFoundError):
        detect(tmp_path / "missing")
