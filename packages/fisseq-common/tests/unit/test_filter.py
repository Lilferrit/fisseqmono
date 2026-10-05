"""The shared filter stage: control marking, key filtering and normalized-table rebuilding."""

import polars as pl
import pytest

from fisseq_common.normalizer import Normalizer
from fisseq_common.stages.filter import (
    SYNONYMOUS_CONTROL,
    filter_and_fit_normalizer,
    load_filtered_cells,
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
