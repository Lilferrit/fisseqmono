from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
import yaml
from omegaconf import OmegaConf

import fisseq_element_data_pipeline.input as m
from fisseq_element_data_pipeline.input import (
    InputStageConfig,
    element_identity_columns,
    load_and_concat,
    load_and_tag,
    load_feature_patterns,
    select_output_columns,
)


def _element_frame(targets, variants) -> pl.LazyFrame:
    """A small Element cell table: bookkeeping columns, identity columns and features."""
    n = len(targets)
    return pl.LazyFrame(
        {
            "cell_id_global": list(range(n)),
            "tile": [f"t{i % 2}" for i in range(n)],
            "ImageNumber": [1] * n,
            "ObjectNumber": list(range(n)),
            "AreaShape_Area": [float(i) for i in range(n)],
            "Mean_Cytoplasm_Intensity_MeanIntensity_CH0": [0.5] * n,
            "origin_y": [0] * n,
            "origin_x": [0] * n,
            "MostAbundantTarget": targets,
            "variant": variants,
            "MostAbundantTargetCount": pl.Series([3] * n, dtype=pl.Int32),
            "TotalCount": [4] * n,
            "Confidence": [0.75] * n,
        }
    )


# ---------------------------------------------------------------------------
# load_and_tag / load_and_concat
# ---------------------------------------------------------------------------


def test_load_and_tag_parquet_adds_origin_columns(tmp_path):
    p = tmp_path / "cells.parquet"
    pl.DataFrame({"variant": ["WT", "M1K"]}).write_parquet(p)

    result = load_and_tag(str(p)).collect()

    assert result["origin_file"].to_list() == [str(p), str(p)]
    assert result["origin_row_idx"].to_list() == [0, 1]


def test_load_and_tag_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "cells.txt"
    p.write_text("variant\nWT\n")

    with pytest.raises(ValueError):
        load_and_tag(str(p)).collect()


def test_load_and_concat_stacks_files(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.csv"
    pl.DataFrame({"x": [1, 2]}).write_parquet(a)
    pl.DataFrame({"x": [3]}).write_csv(b)

    result = load_and_concat([str(a), str(b)]).collect()

    assert result["x"].to_list() == [1, 2, 3]
    assert result["origin_row_idx"].to_list() == [0, 1, 0]


def test_load_feature_patterns_skips_blank_lines(tmp_path):
    p = tmp_path / "patterns.txt"
    p.write_text("AreaShape_*\n\n  Location_*  \n")

    assert load_feature_patterns(str(p)) == ["AreaShape_*", "Location_*"]


# ---------------------------------------------------------------------------
# element_identity_columns
# ---------------------------------------------------------------------------


def test_barcode_is_the_whole_target_name():
    lf = _element_frame(
        ["W498A_0007", "A502L_R644H", "W498._0003", "WT_0001"],
        ["W498A", "A502L|R644H", "W498*", "WT"],
    )

    result = element_identity_columns(lf).collect()

    assert result["upBarcode"].to_list() == [
        "W498A_0007",
        "A502L_R644H",
        "W498._0003",
        "WT_0001",
    ]
    assert result["aaChanges"].to_list() == ["W498A", "A502L|R644H", "W498*", "WT"]


def test_edit_distance_is_zero_int64():
    result = element_identity_columns(
        _element_frame(["M1K_0001", "WT_0001"], ["M1K", "WT"])
    ).collect()

    assert result.schema["editDistance"] == pl.Int64
    assert result["editDistance"].to_list() == [0, 0]
    assert result.schema["upBarcode"] == pl.String
    assert result.schema["aaChanges"] == pl.String


def test_null_target_or_variant_dropped():
    lf = _element_frame(["M1K_0001", None, "M2L_0001"], ["M1K", "M1K", None])

    result = element_identity_columns(lf).collect()

    assert result["upBarcode"].to_list() == ["M1K_0001"]


def test_wt_label_relabelled():
    lf = _element_frame(["silent_0001", "M1K_0001"], ["silent", "M1K"])

    result = element_identity_columns(lf, wt_label="silent").collect()

    assert result["aaChanges"].to_list() == ["WT", "M1K"]


def test_missing_identity_column_raises():
    lf = _element_frame(["M1K_0001"], ["M1K"]).drop("MostAbundantTarget")

    with pytest.raises(ValueError, match="MostAbundantTarget"):
        element_identity_columns(lf)


def test_renamed_identity_columns():
    lf = _element_frame(["M1K_0001"], ["M1K"]).rename(
        {"MostAbundantTarget": "target", "variant": "aa"}
    )

    result = element_identity_columns(lf, target_col="target", variant_col="aa")

    assert result.collect()["upBarcode"].to_list() == ["M1K_0001"]


# ---------------------------------------------------------------------------
# select_output_columns
# ---------------------------------------------------------------------------


def test_select_output_columns_splits_features_and_metadata():
    lf = _element_frame(["M1K_0001"], ["M1K"]).with_columns(
        pl.lit("f").alias("origin_file"), pl.lit(0).alias("origin_row_idx")
    )

    result = select_output_columns(lf).collect()

    assert result.columns == [
        "AreaShape_Area",
        "Mean_Cytoplasm_Intensity_MeanIntensity_CH0",
        "upBarcode",
        "editDistance",
        "aaChanges",
        "meta_cell_id_global",
        "meta_tile",
        "meta_ImageNumber",
        "meta_ObjectNumber",
        "meta_origin_y",
        "meta_origin_x",
        "meta_MostAbundantTargetCount",
        "meta_TotalCount",
        "meta_Confidence",
        "meta_origin_file",
        "meta_origin_row_idx",
    ]


def test_unknown_string_column_is_metadata():
    lf = _element_frame(["M1K_0001"], ["M1K"]).with_columns(pl.lit("x").alias("well"))

    result = select_output_columns(lf).collect()

    assert "meta_well" in result.columns
    assert "well" not in result.columns


def test_feature_allowlist_then_blocklist():
    lf = _element_frame(["M1K_0001"], ["M1K"]).with_columns(
        pl.lit(1.0).alias("AreaShape_Perimeter")
    )

    result = select_output_columns(
        lf, feature_allowlist=["AreaShape_*"], feature_blocklist=["*_Perimeter"]
    ).collect()

    assert "AreaShape_Area" in result.columns
    assert "AreaShape_Perimeter" not in result.columns
    assert "Mean_Cytoplasm_Intensity_MeanIntensity_CH0" not in result.columns


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path, config_path, output_root=None):
    return OmegaConf.structured(
        InputStageConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            log_level="info",
            config_path=str(config_path),
        )
    )


def test_main_writes_element_input(tmp_path):
    source = tmp_path / "source.parquet"
    _element_frame(
        ["M1K_0001", "A2L_R4H", "silent_0001"], ["M1K", "A2L|R4H", "silent"]
    ).collect().write_parquet(source)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"input_paths": [str(source)], "wt_label": "silent"})
    )

    with patch("fisseq_element_data_pipeline.input.setup_logging"):
        m.main.__wrapped__(_make_cfg(tmp_path, config_path, output_root="b1"))

    result = pl.read_parquet(tmp_path / "out" / "b1.output.parquet")
    assert result["upBarcode"].to_list() == ["M1K_0001", "A2L_R4H", "silent_0001"]
    assert result["aaChanges"].to_list() == ["M1K", "A2L|R4H", "WT"]
    assert result["editDistance"].to_list() == [0, 0, 0]
    assert result["meta_origin_file"].to_list() == [str(source)] * 3
