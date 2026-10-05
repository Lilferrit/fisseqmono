"""FILTER_AGGREGATE -- aggregates -> filtered aggregates, plus passthrough."""

from pathlib import Path

import polars as pl
import pytest

import fisseq_common.stages.filter_aggregate as m

LABEL = "meta_aa_changes"


def _aggregate() -> pl.DataFrame:
    return pl.DataFrame(
        {
            LABEL: ["M1K", "L2P"],
            "emb_0000_median": [1.0, 2.0],
            "emb_0001_median": [3.0, 4.0],
            "emb_0000_KS": [0.1, 0.2],
            "meta_num_cells": [10, 20],
        }
    )


def _blocklist(ok: "dict[str, bool]") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": list(ok),
            "median_r": [0.9 if v else 0.1 for v in ok.values()],
            "feature_ok": list(ok.values()),
        }
    )


def test_blocked_columns_are_dropped() -> None:
    out = m.apply_blocklist(
        _aggregate(),
        _blocklist(
            {"emb_0000_median": True, "emb_0001_median": False, "emb_0000_KS": True}
        ),
    )
    assert "emb_0001_median" not in out.columns
    assert "emb_0000_median" in out.columns
    assert "emb_0000_KS" in out.columns


def test_metadata_columns_are_never_dropped() -> None:
    out = m.apply_blocklist(_aggregate(), _blocklist({"emb_0000_median": False}))
    assert LABEL in out.columns
    assert "meta_num_cells" in out.columns


def test_columns_absent_from_the_blocklist_are_kept() -> None:
    """A partial blocklist must never silently delete data -- only columns
    explicitly marked not-ok go."""
    out = m.apply_blocklist(_aggregate(), _blocklist({"emb_0000_median": False}))
    assert "emb_0001_median" in out.columns
    assert "emb_0000_KS" in out.columns


def test_blocklist_entries_absent_from_the_aggregate_are_ignored() -> None:
    out = m.apply_blocklist(_aggregate(), _blocklist({"emb_9999_median": False}))
    assert out.columns == _aggregate().columns


def test_null_feature_ok_counts_as_blocked() -> None:
    bl = pl.DataFrame(
        {
            "feature": ["emb_0000_median"],
            "median_r": [None],
            "feature_ok": [None],
        },
        schema={"feature": pl.String, "median_r": pl.Float64, "feature_ok": pl.Boolean},
    )
    assert m.blocked_features(bl) == {"emb_0000_median"}


def test_passthrough_is_left_joined(tmp_path: Path) -> None:
    """A variant missing from a passthrough aggregate surfaces as null, not
    as a vanished row."""
    d = tmp_path / "pt"
    d.mkdir()
    pl.DataFrame({LABEL: ["M1K"], "emb_0000_KSnegLogP": [7.0]}).write_parquet(
        d / "KSnegLogP.parquet"
    )

    out = m.join_passthrough(_aggregate(), f"{d}/*.parquet", LABEL).sort(LABEL)

    assert out.height == 2
    assert out["emb_0000_KSnegLogP"].to_list() == [None, 7.0]


def test_passthrough_none_is_a_no_op() -> None:
    out = m.join_passthrough(_aggregate(), None, LABEL)
    assert out.columns == _aggregate().columns


def test_passthrough_empty_glob_warns_rather_than_raising(tmp_path: Path) -> None:
    """An empty aggregate_methods_passthrough is the default, so a pattern
    matching nothing is normal -- contrast BLOCKLIST's declared inputs."""
    out = m.join_passthrough(_aggregate(), f"{tmp_path}/none/*.parquet", LABEL)
    assert out.columns == _aggregate().columns


def test_passthrough_column_collision_raises(tmp_path: Path) -> None:
    d = tmp_path / "pt"
    d.mkdir()
    pl.DataFrame({LABEL: ["M1K"], "emb_0000_KS": [7.0]}).write_parquet(d / "KS.parquet")

    with pytest.raises(ValueError, match="collides with existing column"):
        m.join_passthrough(_aggregate(), f"{d}/*.parquet", LABEL)
