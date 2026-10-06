"""The shared filter stage: control marking, key filtering and normalized-table rebuilding."""

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf

from fisseq_common.normalizer import Normalizer
from fisseq_common.stages.config import EMBEDDINGS_JOIN_KEYS, CellsInput
from fisseq_common.stages.filter import (
    SYNONYMOUS_CONTROL,
    WT_CONTROL,
    FilterParams,
    filter_and_fit_normalizer,
    load_cells,
    load_filtered_cells,
    main,
    mark_controls,
    variant_classification,
)

LABEL = "meta_aa_changes"


def _cells() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "meta_cell_index": [0, 1, 2, 3, 4, 5, 2],
            "meta_variant_tag": [None, None, None, None, None, None, "downsample-1"],
            LABEL: ["WT", "WT", "A1A", "C2C", "M1K", "M1K", "A1A:downsample-1"],
            "f1": [1.0, 3.0, 10.0, 20.0, 5.0, 7.0, 10.0],
            "f2": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 4.0],
        }
    ).lazy()


KEYS = ["meta_cell_index", "meta_variant_tag"]


def test_sql_control_marks_matching_rows():
    out = mark_controls(_cells(), "meta_aa_changes = 'WT'", LABEL).collect()
    assert out["meta_is_control"].to_list() == [True, True] + [False] * 5


def test_synonymous_control_marks_untagged_synonymous_rows():
    out = mark_controls(_cells(), SYNONYMOUS_CONTROL, LABEL).collect()
    assert out["meta_is_control"].to_list() == [False, False, True, True] + [False] * 3


def test_synonymous_control_is_variant_classification():
    a = mark_controls(_cells(), SYNONYMOUS_CONTROL, LABEL).collect()
    b = variant_classification(_cells(), LABEL).collect()
    assert a.equals(b)


def test_custom_sql_control():
    out = mark_controls(_cells(), "meta_aa_changes = 'M1K'", LABEL).collect()
    assert out["meta_is_control"].sum() == 2


def test_filter_keeps_only_qc_passed_cells_and_no_features():
    qc_passed = _cells().filter(pl.col("meta_cell_index") != 5)
    keys, _ = filter_and_fit_normalizer(
        _cells(), qc_passed, LABEL, KEYS, control="meta_aa_changes = 'WT'"
    )
    keys = keys.collect()
    assert 5 not in keys["meta_cell_index"].to_list()
    assert keys.height == 6
    assert not {"f1", "f2"} & set(keys.columns)
    assert "meta_is_control" in keys.columns


def test_null_join_keys_match():
    """Untagged cells have a null meta_variant_tag; they must still join."""
    keys, _ = filter_and_fit_normalizer(_cells(), _cells(), LABEL, KEYS)
    assert keys.collect().height == 7


def test_normalizer_is_fitted_on_controls_only():
    _, normalizer = filter_and_fit_normalizer(
        _cells(), _cells(), LABEL, KEYS, control="meta_aa_changes = 'WT'"
    )
    assert normalizer.means["f1"][0] == pytest.approx(2.0)


def test_load_filtered_cells_rebuilds_the_normalized_table():
    cells = _cells().with_columns(pl.lit("b1").alias("meta_batch"))
    keys, normalizer = filter_and_fit_normalizer(
        cells, cells, LABEL, KEYS, control="meta_aa_changes = 'WT'"
    )
    out = load_filtered_cells(_cells(), keys, normalizer, KEYS, sort_by=KEYS).collect()
    expected = normalizer.apply(
        mark_controls(_cells(), "meta_aa_changes = 'WT'", LABEL)
    ).collect()
    expected = expected.sort(KEYS, nulls_last=False)
    assert out.select(expected.columns).equals(expected)
    # a column only the keys carry comes through the join
    assert out["meta_batch"].unique().to_list() == ["b1"]


def test_load_filtered_cells_does_not_suffix_shared_columns():
    keys, normalizer = filter_and_fit_normalizer(_cells(), _cells(), LABEL, KEYS)
    out = load_filtered_cells(_cells(), keys, normalizer, KEYS).collect()
    assert not [c for c in out.columns if c.endswith("_right")]


def test_load_filtered_cells_sorts_when_asked():
    keys, normalizer = filter_and_fit_normalizer(_cells(), _cells(), LABEL, KEYS)
    shuffled = _cells().collect().sample(fraction=1.0, shuffle=True, seed=3).lazy()
    out = load_filtered_cells(shuffled, keys, normalizer, KEYS, sort_by=KEYS).collect()
    assert out.select(KEYS).equals(out.select(KEYS).sort(KEYS, nulls_last=False))


def test_normalizer_round_trip_type():
    _, normalizer = filter_and_fit_normalizer(_cells(), _cells(), LABEL, KEYS)
    assert isinstance(normalizer, Normalizer)


def test_default_control_is_wildtype():
    assert FilterParams().control == WT_CONTROL
    _, normalizer = filter_and_fit_normalizer(_cells(), _cells(), LABEL, KEYS)
    assert normalizer.means["f1"][0] == pytest.approx(2.0)


def test_meta_columns_come_from_the_qc_side():
    """The cell table's own ``meta_*`` columns are ignored: QC_FILTER's side is the source of
    every label, so a stale label in the cell table can't leak into the keys."""
    stale = _cells().with_columns(pl.lit("stale").alias(LABEL))
    keys, _ = filter_and_fit_normalizer(stale, _cells(), LABEL, KEYS)
    keys = keys.collect()
    assert "stale" not in keys[LABEL].to_list()
    assert keys.height == 7


# ---------------------------------------------------------------------------
# QC pseudo-variant rows
# ---------------------------------------------------------------------------


def test_pseudo_variant_rows_are_not_duplicated_with_data_join_keys():
    """A pseudo-variant row shares its source cell's ``meta_cell_index`` but has its own
    ``meta_variant_tag``: each of the two rows joins only itself."""
    keys, normalizer = filter_and_fit_normalizer(
        _cells(), _cells(), LABEL, KEYS, sort_by=KEYS
    )
    keys = keys.collect()
    assert keys.height == _cells().collect().height
    assert keys.select(KEYS).is_duplicated().sum() == 0

    out = load_filtered_cells(_cells(), keys.lazy(), normalizer, KEYS).collect()
    assert out.height == keys.height
    assert sorted(out[LABEL].to_list()) == sorted(_cells().collect()[LABEL].to_list())


def _embedding_cells() -> pl.LazyFrame:
    """The embeddings pipeline's cell table: one row per cell, keyed by its tile, no labels."""
    return pl.DataFrame(
        {
            "meta_batch": ["b"] * 4,
            "meta_well": ["w"] * 4,
            "meta_tile": ["t0", "t0", "t1", "t1"],
            "meta_cell_index": [0, 1, 0, 1],
            "emb_0000": [1.0, 3.0, 10.0, 20.0],
        }
    ).lazy()


def _embedding_qc_passed() -> pl.LazyFrame:
    """QC_FILTER's output for :func:`_embedding_cells`, with a pseudo-variant copy of cell
    (t1, 0) under its own label and tag."""
    return pl.DataFrame(
        {
            "meta_batch": ["b"] * 5,
            "meta_well": ["w"] * 5,
            "meta_tile": ["t0", "t0", "t1", "t1", "t1"],
            "meta_cell_index": [0, 1, 0, 1, 0],
            "meta_variant_tag": [None, None, None, None, "downsample-1"],
            LABEL: ["WT", "WT", "A1A", "M1K", "A1A:downsample-1"],
        }
    ).lazy()


def test_embeddings_pseudo_variant_row_gets_its_source_cells_features():
    join_keys = list(EMBEDDINGS_JOIN_KEYS)
    keys, normalizer = filter_and_fit_normalizer(
        _embedding_cells(), _embedding_qc_passed(), LABEL, join_keys
    )
    out = load_filtered_cells(
        _embedding_cells(),
        keys,
        normalizer,
        join_keys,
        sort_by=join_keys + ["meta_variant_tag"],
    ).collect()

    assert out.height == 5
    source = out.filter(pl.col(LABEL) == "A1A")
    pseudo = out.filter(pl.col(LABEL) == "A1A:downsample-1")
    assert pseudo.height == source.height == 1
    assert pseudo["meta_variant_tag"].to_list() == ["downsample-1"]
    assert pseudo["emb_0000"].to_list() == source["emb_0000"].to_list()
    # WT (1.0, 3.0) -> mean 2, std sqrt(2)
    assert pseudo["emb_0000"][0] == pytest.approx((10.0 - 2.0) / 2**0.5)


# ---------------------------------------------------------------------------
# The entry point (python -m fisseq_common.stages.filter), as the data pipeline runs it:
# QC_FILTER's filtered_cells is both the cell table and the QC side.
# ---------------------------------------------------------------------------


def write_qc_cells(tmp_path) -> str:
    """A minimal QC_FILTER filtered_cells.parquet."""
    path = tmp_path / "filtered_cells.parquet"
    pl.DataFrame(
        {
            "meta_cell_index": pl.Series([0, 1, 2, 3, 4], dtype=pl.Int64),
            "meta_variant_tag": pl.Series([None] * 5, dtype=pl.String),
            LABEL: ["WT", "WT", "WT", "M1A", "M1A"],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    ).write_parquet(path)
    return str(path)


def run_main(tmp_path, **overrides):
    qc_cells = write_qc_cells(tmp_path)
    cfg = FilterParams(
        output_dir=str(tmp_path / "out"),
        cells_file=qc_cells,
        qc_passed_file=qc_cells,
        **overrides,
    )
    with patch("fisseq_common.stages.config.setup_logging"):
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


def test_no_batch_name_adds_no_batch_column(tmp_path):
    out = run_main(tmp_path)
    keys = pl.read_parquet(out / "filtered_keys.parquet")
    assert "meta_batch" not in keys.columns


def test_main_fits_the_normalizer_on_wildtype(tmp_path):
    out = run_main(tmp_path)
    normalizer = Normalizer.load(out / "normalizer.parquet")
    assert normalizer.means["f1"][0] == pytest.approx(2.0)


def test_main_control_is_configurable(tmp_path):
    out = run_main(tmp_path, control="meta_aa_changes = 'M1A'")
    keys = pl.read_parquet(out / "filtered_keys.parquet")
    assert keys["meta_is_control"].to_list() == [False, False, False, True, True]


def test_main_output_root_prefixes_both_files(tmp_path):
    out = run_main(tmp_path, output_root="b1")
    assert (out / "b1.filtered_keys.parquet").exists()
    assert (out / "b1.normalizer.parquet").exists()


def test_load_cells_rebuilds_wildtype_normalized_cells(tmp_path):
    out = run_main(tmp_path, batch_name="batch1")
    df = load_cells(
        CellsInput(
            cells_file=str(tmp_path / "filtered_cells.parquet"),
            filtered_keys_file=str(out / "filtered_keys.parquet"),
            normalizer_file=str(out / "normalizer.parquet"),
        )
    ).collect()
    assert df["meta_cell_index"].to_list() == [0, 1, 2, 3, 4]
    assert df.filter(pl.col(LABEL) == "WT")["f1"].mean() == pytest.approx(0.0, abs=1e-9)
    assert df["meta_batch"].unique().to_list() == ["batch1"]
