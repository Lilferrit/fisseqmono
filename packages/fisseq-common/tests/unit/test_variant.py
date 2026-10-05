import polars as pl
import pytest

from fisseq_common.variant import (
    VARIANT_TYPES,
    classify_variant,
    control_expr,
    variant_type_expr,
)

# ---------------------------------------------------------------------------
# classify_variant
# ---------------------------------------------------------------------------


class TestClassifyVariant:
    @pytest.mark.parametrize(
        "v,expected",
        [
            ("WT", "WT"),
            ("A1A", "Synonymous"),
            ("A1B", "Single Missense"),
            ("A1fs", "Frameshift"),
            ("A1X", "Nonsense"),
            ("A1*", "Nonsense"),
            ("A5-|A6-", "3nt Deletion"),
            ("A5-|A9-", "Other"),
            ("A5-", "3nt Deletion"),
            ("garbage", "Other"),
            ("M1K:downsampled-half", "Single Missense"),
            ("A1A:sometag", "Synonymous"),
            ("M1K:tag:extra", "Single Missense"),
        ],
    )
    def test_classify(self, v, expected):
        assert classify_variant(v) == expected
        assert expected in VARIANT_TYPES


# Labels whose tag would change the result if it weren't stripped: the tag contains "fs",
# "X", "*", "WT", or ends in "-".
LABELS = [
    "WT",
    "A1A",
    "A1B",
    "A1fs",
    "A1X",
    "A1*",
    "A5-|A6-",
    "A5-|A9-",
    "A5-",
    "garbage",
    "M1K:downsampled-half",
    "A1A:sometag",
    "M1K:tag:extra",
    "A1A:offs",
    "A1A:X",
    "A1A:*",
    "A1A:WT",
    "M1K:minus-",
    "WT:tag",
]


def test_variant_type_expr_matches_classify_variant() -> None:
    df = pl.DataFrame({"v": LABELS}).with_columns(variant_type_expr("v").alias("t"))
    assert df["t"].to_list() == [classify_variant(v) for v in LABELS]


def test_variant_type_expr_null_label_is_null() -> None:
    df = pl.DataFrame({"v": ["A1A", None]}, schema={"v": pl.String})
    assert df.select(variant_type_expr("v").alias("t"))["t"].to_list() == [
        "Synonymous",
        None,
    ]


def test_control_is_untagged_synonymous() -> None:
    df = pl.DataFrame({"v": ["A1A", "A1A:downsampled-half", "M1K", "WT"]})
    is_control = df.select(control_expr(variant_type_expr("v"), "v").alias("c"))["c"]
    assert is_control.to_list() == [True, False, False, False]
