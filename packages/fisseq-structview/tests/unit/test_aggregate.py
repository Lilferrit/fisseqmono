import polars as pl
import pytest

from fisseq_structview.aggregate import (
    N_VARIANTS,
    RESIDUE,
    VALUE,
    WT,
    apply_offset,
    filter_types,
    per_position,
)
from fisseq_structview.data import POSITION, SCORE, VARIANT, VARIANT_TYPE


@pytest.fixture
def variants() -> pl.DataFrame:
    return pl.DataFrame(
        {
            VARIANT: ["K2E", "K2R", "K2N", "A4V", "Y5F", "A4*", "K2K", "WT"],
            VARIANT_TYPE: ["Single Missense"] * 5 + ["Nonsense", "Synonymous", "WT"],
            POSITION: [2, 2, 2, 4, 5, 4, 2, None],
            WT: ["K", "K", "K", "A", "Y", None, "K", None],
            SCORE: [0.7, 0.6, 0.9, 0.8, None, 0.95, 0.5, 0.5],
        }
    )


def _values(df: pl.DataFrame) -> dict[int, float]:
    return dict(zip(df[RESIDUE].to_list(), df[VALUE].to_list()))


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("median", {2: 0.7, 4: 0.8}),
        ("mean", {2: (0.7 + 0.6 + 0.9) / 3, 4: 0.8}),
        ("min", {2: 0.6, 4: 0.8}),
        ("max", {2: 0.9, 4: 0.8}),
        ("frac_deleterious", {2: 2 / 3, 4: 1.0}),
    ],
)
def test_methods(variants, method, expected):
    missense = apply_offset(filter_types(variants, ["Single Missense"]), 0)
    got = _values(per_position(missense, method, threshold=0.65))
    assert got.keys() == expected.keys()  # Y5F has no score: no row for residue 5
    for residue, value in expected.items():
        assert got[residue] == pytest.approx(value)


def test_frac_deleterious_threshold_is_strict(variants):
    df = apply_offset(filter_types(variants, ["Single Missense"]), 0)
    assert _values(per_position(df, "frac_deleterious", threshold=0.7))[2] == (
        pytest.approx(1 / 3)
    )


def test_frac_deleterious_absolute():
    df = pl.DataFrame(
        {
            VARIANT: ["K2E", "K2R", "K2N", "K2D"],
            VARIANT_TYPE: ["Single Missense"] * 4,
            POSITION: [2] * 4,
            WT: ["K"] * 4,
            SCORE: [2.5, -3.0, 1.0, -1.5],
        }
    )
    df = apply_offset(df, 0)
    assert _values(per_position(df, "frac_deleterious", threshold=2.0))[2] == 0.25
    got = per_position(df, "frac_deleterious", threshold=2.0, absolute=True)
    assert _values(got)[2] == 0.5


def test_type_filter_and_counts(variants):
    df = apply_offset(filter_types(variants, ["Single Missense", "Nonsense"]), 0)
    out = per_position(df, "max")
    assert _values(out) == {2: 0.9, 4: 0.95}
    assert dict(zip(out[RESIDUE], out[N_VARIANTS])) == {2: 3, 4: 2}
    assert dict(zip(out[RESIDUE], out[WT])) == {2: "K", 4: "A"}


def test_details_best_first(variants):
    df = apply_offset(filter_types(variants, ["Single Missense"]), 0)
    out = per_position(df).filter(pl.col(RESIDUE) == 2)
    assert out["details"][0].to_list() == ["K2N 0.9", "K2E 0.7", "K2R 0.6"]


def test_unknown_method(variants):
    with pytest.raises(ValueError, match="Unknown aggregation 'mode'"):
        per_position(apply_offset(variants, 0), "mode")


@pytest.mark.parametrize("offset", [0, 24, -1])
def test_offset_shifts_residues(variants, offset):
    out = per_position(
        apply_offset(filter_types(variants, ["Single Missense"]), offset)
    )
    assert out[RESIDUE].to_list() == [2 + offset, 4 + offset]
