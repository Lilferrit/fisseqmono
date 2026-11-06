import logging

import polars as pl
import pytest
from structview_testdata import SEQUENCE

from fisseq_structview.aggregate import RESIDUE, VALUE, WT
from fisseq_structview.mapping import MappingError, match_positions

RESIDUES = {i + 1: aa for i, aa in enumerate(SEQUENCE)}  # 1-8: MKTAYIAK


def _table(rows):
    return pl.DataFrame(rows, schema=[RESIDUE, VALUE, WT], orient="row")


def test_unscored_residues_are_listed():
    match = match_positions(_table([(2, 0.7, "K"), (4, 0.9, "A")]), RESIDUES)
    assert match.values == {2: 0.7, 4: 0.9}
    assert match.unscored == (1, 3, 5, 6, 7, 8)
    assert match.outside == ()


def test_positions_outside_the_structure_are_dropped_with_a_warning(caplog):
    table = _table([(2, 0.7, "K"), (8, 0.1, "K"), (9, 0.5, None), (12, 0.6, None)])
    with caplog.at_level(logging.WARNING):
        match = match_positions(table, RESIDUES)
    assert match.values == {2: 0.7, 8: 0.1}
    assert match.outside == (9, 12)
    assert "2 of 4 scored positions (9-12) are not in the structure" in caplog.text


def test_no_overlap_names_both_ranges_and_the_offset():
    with pytest.raises(MappingError) as err:
        match_positions(
            _table([(102, 0.7, "K"), (104, 0.9, "A")]), RESIDUES, offset=100
        )
    message = str(err.value)
    assert (
        "None of the 2 scored positions (residues 102-104 after offset 100)" in message
    )
    assert "residues are 1-8" in message
    assert "--offset" in message


def test_wild_type_mismatch_is_an_error():
    # Shifted by one: label K2 lands on T3, A4 on Y5.
    table = _table([(3, 0.7, "K"), (5, 0.9, "A")])
    with pytest.raises(MappingError, match=r"2 of 2 .*3: label K, structure T"):
        match_positions(table, RESIDUES, offset=1)
    match = match_positions(table, RESIDUES, check_wt=False)
    assert match.wt_mismatches == ((3, "K", "T"), (5, "A", "Y"))


def test_a_few_mismatches_are_tolerated():
    rows = [(i, 0.5, aa) for i, aa in RESIDUES.items()]
    rows[0] = (1, 0.5, "W")  # 1 of 8 = 12.5%
    with pytest.raises(MappingError):
        match_positions(_table(rows), RESIDUES)
    match = match_positions(_table(rows), RESIDUES, max_mismatch=0.2)
    assert match.wt_mismatches == ((1, "W", "M"),)


def test_empty_table_is_not_an_error():
    match = match_positions(_table([]), RESIDUES)
    assert match.values == {} and match.unscored == tuple(RESIDUES)
