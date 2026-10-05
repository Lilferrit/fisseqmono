"""NORMALIZE's entry point: QC-passed keys plus the wildtype-fitted normalizer."""

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_data_pipeline.cells import CellsInput, load_cells
from fisseq_data_pipeline.normalize import NormalizeConfig, main


def write_qc_cells(tmp_path) -> str:
    """A minimal QC_FILTER filtered_cells.parquet."""
    path = tmp_path / "filtered_cells.parquet"
    pl.DataFrame(
        {
            "meta_cell_index": pl.Series([0, 1, 2, 3, 4], dtype=pl.Int64),
            "meta_variant_tag": pl.Series([None] * 5, dtype=pl.String),
            "meta_aa_changes": ["WT", "WT", "WT", "M1A", "M1A"],
            "f1": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
            "f2": pl.Series([10.0, 20.0, 30.0, 40.0, 50.0], dtype=pl.Float64),
        }
    ).write_parquet(path)
    return str(path)


def run_main(tmp_path, **overrides):
    cfg = NormalizeConfig(
        output_dir=str(tmp_path / "out"), input_file=write_qc_cells(tmp_path)
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    with patch("fisseq_data_pipeline.normalize.setup_logging"):
        main.__wrapped__(OmegaConf.structured(cfg))
    return tmp_path / "out"


def test_main_writes_keys_and_normalizer_only(tmp_path):
    out = run_main(tmp_path, batch_name="batch1")
    assert sorted(p.name for p in out.glob("*.parquet")) == [
        "filtered_keys.parquet",
        "normalizer.parquet",
    ]


def test_keys_carry_meta_columns_batch_and_control_but_no_features(tmp_path):
    out = run_main(tmp_path, batch_name="batch1")
    keys = pl.read_parquet(out / "filtered_keys.parquet")
    assert not {"f1", "f2"} & set(keys.columns)
    assert keys["meta_batch"].unique().to_list() == ["batch1"]
    assert keys["meta_is_control"].to_list() == [True, True, True, False, False]


def test_batch_name_defaults_to_input_stem(tmp_path):
    out = run_main(tmp_path)
    keys = pl.read_parquet(out / "filtered_keys.parquet")
    assert keys["meta_batch"].unique().to_list() == ["filtered_cells"]


def test_normalizer_is_fitted_on_wildtype(tmp_path):
    out = run_main(tmp_path)
    normalizer = Normalizer.load(out / "normalizer.parquet")
    assert normalizer.means["f1"][0] == pytest.approx(2.0)


def test_control_sample_query_is_configurable(tmp_path):
    out = run_main(tmp_path, control_sample_query="meta_aa_changes = 'M1A'")
    keys = pl.read_parquet(out / "filtered_keys.parquet")
    assert keys["meta_is_control"].to_list() == [False, False, False, True, True]


def test_output_root_prefixes_both_files(tmp_path):
    out = run_main(tmp_path, output_root="b1")
    assert (out / "b1.filtered_keys.parquet").exists()
    assert (out / "b1.normalizer.parquet").exists()


def test_load_cells_rebuilds_wildtype_normalized_cells(tmp_path):
    out = run_main(tmp_path, batch_name="batch1")
    cells, _ = load_cells(
        CellsInput(
            cells_file=str(tmp_path / "filtered_cells.parquet"),
            filtered_keys_file=str(out / "filtered_keys.parquet"),
            normalizer_file=str(out / "normalizer.parquet"),
        )
    )
    df = cells.collect()
    assert df["meta_cell_index"].to_list() == [0, 1, 2, 3, 4]
    assert df.filter(pl.col("meta_aa_changes") == "WT")["f1"].mean() == pytest.approx(
        0.0, abs=1e-9
    )
    assert df["meta_batch"].unique().to_list() == ["batch1"]


def test_load_cells_accepts_deprecated_input_file(tmp_path):
    path = tmp_path / "batch7.parquet"
    pl.DataFrame({"meta_aa_changes": ["WT"], "f1": [0.0]}).write_parquet(path)
    with pytest.warns(DeprecationWarning):
        cells, stem = load_cells(CellsInput(input_file=str(path)))
    assert stem == "batch7"
    assert cells.collect()["meta_batch"].to_list() == ["batch7"]


def test_load_cells_rejects_mixed_inputs(tmp_path):
    with pytest.raises(ValueError):
        load_cells(CellsInput(input_file="a.parquet", cells_file="b.parquet"))
    with pytest.raises(ValueError):
        load_cells(CellsInput(cells_file="b.parquet"))
