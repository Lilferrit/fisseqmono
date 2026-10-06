"""FINALIZE_FEATURE_SELECT (``python -m fisseq_common.stages.finalize``): per-method
aggregates -> the blocklisted, synonymous-normalized per-variant table, plus the impact score,
per-variant metadata and passthrough aggregates. The entry-point tests moved from the data
pipeline's ``featureselect`` wrapper."""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_common.stages.finalize as m
from fisseq_common.schema import IMPACT_SCORE_COL, META_BARCODE_COL, META_BATCH_COL

LABEL = "meta_aa_changes"


# ---------------------------------------------------------------------------
# apply_blocklist / join_passthrough
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# join_feature_type_files
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, df: pl.DataFrame) -> str:
    path = tmp_path / name
    df.write_parquet(path)
    return str(path)


def test_join_feature_type_files_single_file(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "mean.parquet",
        pl.DataFrame({LABEL: ["A", "B"], "f1_mean": [1.0, 2.0]}),
    )
    result = m.join_feature_type_files([path], LABEL)
    assert result.columns == [LABEL, "f1_mean"]
    assert result["f1_mean"].to_list() == [1.0, 2.0]


def test_join_feature_type_files_joins_on_label_column(tmp_path: Path) -> None:
    mean_path = _write(
        tmp_path,
        "mean.parquet",
        pl.DataFrame({LABEL: ["A", "B"], "f1_mean": [1.0, 2.0]}),
    )
    std_path = _write(
        tmp_path, "std.parquet", pl.DataFrame({LABEL: ["B", "A"], "f1_std": [0.2, 0.1]})
    )
    result = m.join_feature_type_files([mean_path, std_path], LABEL)
    assert result.columns == [LABEL, "f1_mean", "f1_std"]
    assert result.height == 2
    row_a = result.filter(pl.col(LABEL) == "A")
    assert row_a["f1_mean"][0] == 1.0
    assert row_a["f1_std"][0] == 0.1


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def write_filtered_keys(tmp_path: Path) -> None:
    """The filter stage's keys, the source of the per-variant metadata: 4 cells per label
    over two barcodes, one batch."""
    n = 4
    labels = [v for v in ["WT", "A1A", "A2A", "A3A", "A1B", "A1C"] for _ in range(n)]
    pl.DataFrame(
        {
            "meta_cell_index": list(range(len(labels))),
            "meta_variant_tag": pl.Series([None] * len(labels), dtype=pl.String),
            LABEL: labels,
            "meta_is_control": [label == "WT" for label in labels],
            META_BARCODE_COL: ["bc_0", "bc_1"] * (len(labels) // 2),
            META_BATCH_COL: ["batch1"] * len(labels),
        }
    ).write_parquet(tmp_path / "filtered_keys.parquet")


def write_feature_type_aggregate(tmp_path: Path) -> None:
    """One method's aggregates. A1A, A2A and A3A are all Synonymous (classify_variant), so
    together they are the normalization reference; their non-uniformly-spaced values keep the
    synonymous median (used by compute_impact_score) non-degenerate."""
    ft_dir = tmp_path / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            LABEL: ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
        }
    ).write_parquet(ft_dir / "mean.parquet")


def write_blocklist(tmp_path: Path, ok: "dict[str, bool] | None" = None) -> None:
    ok = ok or {"f1_mean": True, "f2_mean": True}
    pl.DataFrame({"feature": list(ok), "feature_ok": list(ok.values())}).write_parquet(
        tmp_path / "blocklist.parquet"
    )


def make_cfg(tmp_path: Path, **overrides) -> OmegaConf:
    fields = dict(
        output_dir=str(tmp_path / "out"),
        feature_type_files=str(tmp_path / "ft" / "*.parquet"),
        block_list_file=str(tmp_path / "blocklist.parquet"),
        filtered_keys_file=str(tmp_path / "filtered_keys.parquet"),
    )
    fields.update(overrides)
    return OmegaConf.structured(m.FinalizeConfig(**fields))


def _run(tmp_path: Path, *, blocklist=None, **overrides) -> pl.DataFrame:
    """Write the default fixtures, run the entry point and return its output."""
    write_filtered_keys(tmp_path)
    write_feature_type_aggregate(tmp_path)
    write_blocklist(tmp_path, blocklist)
    with patch("fisseq_common.stages.config.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, **overrides))
    return pl.read_parquet(tmp_path / "out" / "output.parquet")


def test_main_writes_output_parquet(tmp_path: Path) -> None:
    _run(tmp_path)
    assert [p.name for p in (tmp_path / "out").glob("*.parquet")] == ["output.parquet"]


def test_main_output_root_and_name(tmp_path: Path) -> None:
    write_filtered_keys(tmp_path)
    write_feature_type_aggregate(tmp_path)
    write_blocklist(tmp_path)
    with patch("fisseq_common.stages.config.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path, output_root="b1", output_name="selected"))
    assert (tmp_path / "out" / "b1.selected.parquet").exists()


def test_main_output_is_one_row_per_variant_sorted_by_label(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result[LABEL].to_list() == ["A1A", "A1B", "A1C", "A2A", "A3A"]


def test_main_keeps_meta_is_control_marking_the_synonymous_variants(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)
    marked = result.filter(pl.col("meta_is_control"))[LABEL].to_list()
    assert marked == ["A1A", "A2A", "A3A"]


def test_main_blocked_feature_absent_from_output(tmp_path: Path) -> None:
    result = _run(tmp_path, blocklist={"f1_mean": True, "f2_mean": False})
    assert "f2_mean" not in result.columns
    assert "f1_mean" in result.columns


def test_main_joins_multiple_feature_type_files(tmp_path: Path) -> None:
    write_filtered_keys(tmp_path)
    write_feature_type_aggregate(tmp_path)
    pl.DataFrame(
        {
            LABEL: ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_std": [0.0, 1.0, 3.0, 2.0, 5.0],
        }
    ).write_parquet(tmp_path / "ft" / "std.parquet")
    write_blocklist(tmp_path, {"f1_mean": True, "f2_mean": True, "f1_std": True})
    with patch("fisseq_common.stages.config.setup_logging"):
        m.main.__wrapped__(make_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "output.parquet")
    assert {"f1_mean", "f2_mean", "f1_std"} <= set(result.columns)


def test_main_raises_on_empty_feature_type_glob(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No files matched"):
        _run(tmp_path, feature_type_files=str(tmp_path / "nonexistent" / "*.parquet"))


def test_main_output_features_are_synonymous_normalized(tmp_path: Path) -> None:
    """Output features are z-scored against the synonymous (A1A/A2A/A3A) variants' mean and
    std, checked against an independent numpy computation."""
    result = _run(tmp_path)

    control_f1 = np.array([0.0, 1.0, 4.0])
    control_f2 = np.array([0.0, 2.0, 8.0])
    f1_mean, f1_std = control_f1.mean(), control_f1.std(ddof=1)
    f2_mean, f2_std = control_f2.mean(), control_f2.std(ddof=1)

    raw_values = {
        "A1A": (0.0, 0.0),
        "A2A": (1.0, 2.0),
        "A3A": (4.0, 8.0),
        "A1B": (5.0, 6.0),
        "A1C": (10.0, 12.0),
    }
    for label, (raw_f1, raw_f2) in raw_values.items():
        row = result.filter(pl.col(LABEL) == label)
        assert row["f1_mean"][0] == pytest.approx((raw_f1 - f1_mean) / f1_std, abs=1e-9)
        assert row["f2_mean"][0] == pytest.approx((raw_f2 - f2_mean) / f2_std, abs=1e-9)


def test_main_impact_score_is_on_by_default(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_impact_score_absent_when_disabled(tmp_path: Path) -> None:
    result = _run(tmp_path, compute_impact_score=False)
    assert IMPACT_SCORE_COL not in result.columns


def test_main_synonymous_median_has_zero_impact_score(tmp_path: Path) -> None:
    """compute_impact_score's reference is the synonymous variants' *median*: with raw f1
    0.0, 1.0, 4.0, A2A is the middle value, so its normalized vector is the reference and
    its impact score is 0."""
    result = _run(tmp_path)
    median_row = result.filter(pl.col(LABEL) == "A2A")
    assert median_row[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_main_metadata_comes_from_filtered_keys(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert (result["meta_num_cells"] == 4).all()
    assert (result[f"{META_BARCODE_COL}_num_unique"] == 2).all()
    assert (result[f"{META_BATCH_COL}_num_unique"] == 1).all()


# Deliberately lopsided: A1C's value is an order of magnitude off the rest, so
# a synonymous-baseline z-score would be unmistakable if one were ever applied.
PASSTHROUGH_VALUES = [0.0, 0.0, 0.0, 1.0, 40.0]


def write_passthrough_aggregate(
    tmp_path: Path, *, column: str = "f1_KSnegLogP"
) -> None:
    """A passthrough method's aggregates, in their own directory: Nextflow publishes these
    to passthrough_aggregates/, not aggregates/."""
    pt_dir = tmp_path / "pt"
    pt_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {LABEL: ["A1A", "A2A", "A3A", "A1B", "A1C"], column: PASSTHROUGH_VALUES}
    ).write_parquet(pt_dir / "KSnegLogP.parquet")


def _run_with_passthrough(tmp_path: Path, **overrides) -> pl.DataFrame:
    write_passthrough_aggregate(tmp_path)
    return _run(
        tmp_path,
        passthrough_feature_type_files=str(tmp_path / "pt" / "*.parquet"),
        **overrides,
    )


def test_main_passthrough_values_are_not_normalized(tmp_path: Path) -> None:
    result = _run_with_passthrough(tmp_path)
    expected = pl.read_parquet(tmp_path / "pt" / "KSnegLogP.parquet").sort(LABEL)
    assert result["f1_KSnegLogP"].to_list() == pytest.approx(
        expected["f1_KSnegLogP"].to_list()
    )
    # ... while the selected features of the same run are, so the raw passthrough values
    # are not an artifact of a no-op normalizer.
    ft_values = pl.read_parquet(tmp_path / "ft" / "mean.parquet").sort(LABEL)
    assert result["f1_mean"].to_list() != pytest.approx(ft_values["f1_mean"].to_list())


def test_main_passthrough_column_survives_the_blocklist(tmp_path: Path) -> None:
    """A passthrough column named in the blocklist is still kept: the blocklist is a
    reproducibility verdict, and passthrough methods never had a bootstrap to earn one."""
    result = _run_with_passthrough(
        tmp_path, blocklist={"f1_mean": True, "f2_mean": True, "f1_KSnegLogP": False}
    )
    assert "f1_KSnegLogP" in result.columns


def test_main_passthrough_excluded_from_impact_score(tmp_path: Path) -> None:
    baseline = _run(tmp_path)
    with_passthrough = _run_with_passthrough(tmp_path)
    assert with_passthrough[IMPACT_SCORE_COL].to_list() == pytest.approx(
        baseline[IMPACT_SCORE_COL].to_list()
    )


def test_main_empty_passthrough_glob_warns_and_continues(
    tmp_path: Path, caplog
) -> None:
    """An empty passthrough list reaches the stage as an empty staging dir: a warning, not
    the ValueError an empty feature_type_files glob raises."""
    (tmp_path / "pt").mkdir()
    with caplog.at_level(logging.WARNING):
        result = _run(
            tmp_path, passthrough_feature_type_files=str(tmp_path / "pt" / "*.parquet")
        )
    assert "f1_mean" in result.columns
    assert any("passthrough glob" in record.message for record in caplog.records)


def test_main_passthrough_column_collision_raises(tmp_path: Path) -> None:
    """A method in both lists would collide on the join; Nextflow rejects that up front,
    and this is the backstop for a direct invocation."""
    write_passthrough_aggregate(tmp_path, column="f1_mean")
    with pytest.raises(ValueError, match="collide"):
        _run(
            tmp_path, passthrough_feature_type_files=str(tmp_path / "pt" / "*.parquet")
        )
