from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

import fisseq_data_pipeline.featureselect as m
from fisseq_common.schema import (
    IMPACT_SCORE_COL,
    META_BARCODE_COL,
    META_BATCH_COL,
)


def write_feat_input_parquet(tmp_path) -> None:
    """Raw cell-level parquet used only for the metadata join in main()."""
    n = 4
    labels = (
        ["WT"] * n + ["A1A"] * n + ["A2A"] * n + ["A3A"] * n + ["A1B"] * n + ["A1C"] * n
    )
    pl.DataFrame(
        {
            "meta_aa_changes": labels,
            "meta_is_control": [label == "WT" for label in labels],
            META_BARCODE_COL: ["bc_0", "bc_1"] * (len(labels) // 2),
        }
    ).write_parquet(tmp_path / "input.parquet")


def write_feature_type_aggregate(tmp_path) -> None:
    """Per-feature-type aggregate fixture matching write_feat_input_parquet's
    cell-level means exactly. A1A, A2A, and A3A are all Synonymous
    (classify_variant), so together they form the synonymous-baseline
    normalization reference; their non-uniformly-spaced values keep the
    control group's median (used by compute_impact_score) non-degenerate."""
    ft_dir = tmp_path / "ft"
    ft_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            "f1_mean": [0.0, 1.0, 4.0, 5.0, 10.0],
            "f2_mean": [0.0, 2.0, 8.0, 6.0, 12.0],
        }
    ).write_parquet(ft_dir / "mean.parquet")


def write_blocklist(tmp_path, *, block_f2: bool = False) -> None:
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean"], "feature_ok": [True, not block_f2]}
    ).write_parquet(tmp_path / "blocklist.parquet")


def make_feat_cfg(
    tmp_path,
    *,
    output_root=None,
    feature_type_files=None,
    block_list_file=None,
    compute_impact_score: bool = True,
    passthrough_feature_type_files=None,
    random_seed=42,
) -> OmegaConf:
    """Return a DictConfig for FinalizeFeatureSelectConfig with test defaults."""
    if feature_type_files is None:
        feature_type_files = str(tmp_path / "ft" / "*.parquet")
    if block_list_file is None:
        block_list_file = str(tmp_path / "blocklist.parquet")
    return OmegaConf.structured(
        m.FinalizeFeatureSelectConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            input_file=str(tmp_path / "input.parquet"),
            feature_type_files=feature_type_files,
            block_list_file=block_list_file,
            compute_impact_score=compute_impact_score,
            passthrough_feature_type_files=passthrough_feature_type_files,
            random_seed=random_seed,
        )
    )


def _write_default_fixtures(tmp_path, *, block_f2: bool = False) -> None:
    write_feat_input_parquet(tmp_path)
    write_feature_type_aggregate(tmp_path)
    write_blocklist(tmp_path, block_f2=block_f2)


def test_main_creates_output_file(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    assert (tmp_path / "out" / "input.parquet").exists()


def test_main_output_contains_label_column(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "meta_aa_changes" in result.columns


def test_main_output_root_names_output_file(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    root = str(tmp_path / "run1")
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path, output_root=root))
    assert (tmp_path / "run1.input.parquet").exists()


def test_main_blocked_feature_absent_from_output(tmp_path) -> None:
    _write_default_fixtures(tmp_path, block_f2=True)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f2_mean" not in result.columns


def test_main_unblocked_feature_present_in_output(tmp_path) -> None:
    _write_default_fixtures(tmp_path, block_f2=True)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f1_mean" in result.columns


def test_main_joins_multiple_feature_type_files(tmp_path) -> None:
    write_feat_input_parquet(tmp_path)
    write_feature_type_aggregate(tmp_path)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A1B", "A1C"],
            "f1_std": [0.0, 0.0, 0.0],
        }
    ).write_parquet(tmp_path / "ft" / "std.parquet")
    write_blocklist(tmp_path)
    pl.DataFrame(
        {"feature": ["f1_mean", "f2_mean", "f1_std"], "feature_ok": [True, True, True]}
    ).write_parquet(tmp_path / "blocklist.parquet")

    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert {"f1_mean", "f2_mean", "f1_std"}.issubset(set(result.columns))


def test_main_raises_on_empty_feature_type_glob(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with pytest.raises(ValueError):
            m.main.__wrapped__(
                make_feat_cfg(
                    tmp_path,
                    feature_type_files=str(tmp_path / "nonexistent" / "*.parquet"),
                )
            )


def test_main_impact_score_column_present_by_default(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL in result.columns


def test_main_impact_score_column_absent_when_disabled(tmp_path) -> None:
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path, compute_impact_score=False))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert IMPACT_SCORE_COL not in result.columns


def test_main_impact_score_values_are_finite(tmp_path) -> None:
    # A1A/A2A/A3A are the synonymous control group; their normalized feature
    # vectors are non-zero, so compute_impact_score produces finite scores
    # for all rows.
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert result[IMPACT_SCORE_COL].is_finite().all()


def test_main_synonymous_median_has_zero_impact_score(tmp_path) -> None:
    # variant_classification marks A1A/A2A/A3A (all Synonymous) as the
    # control group. compute_impact_score's reference vector is their
    # *median*, not their mean — with three non-uniformly-spaced values
    # (raw f1_mean = 0.0, 1.0, 4.0), A2A (raw f1_mean=1.0) is the actual
    # middle value, so its normalized vector exactly equals the control
    # median and its impact score should be 0.
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path))
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    median_row = result.filter(pl.col("meta_aa_changes") == "A2A")
    assert median_row[IMPACT_SCORE_COL][0] == pytest.approx(0.0, abs=1e-9)


def test_main_output_features_are_synonymous_normalized(tmp_path) -> None:
    # Output feature values should be z-scored against the synonymous
    # (A1A/A2A/A3A) control group's mean/std, not left as raw aggregate
    # values. Verify against an independent numpy computation from the
    # fixture's known raw values (write_feature_type_aggregate).
    result = _run_main(tmp_path)

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
        row = result.filter(pl.col("meta_aa_changes") == label)
        assert row["f1_mean"][0] == pytest.approx((raw_f1 - f1_mean) / f1_std, abs=1e-9)
        assert row["f2_mean"][0] == pytest.approx((raw_f2 - f2_mean) / f2_std, abs=1e-9)


def _run_main(tmp_path, **kwargs) -> pl.DataFrame:
    """Run main() and return the output parquet."""
    _write_default_fixtures(tmp_path)
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(make_feat_cfg(tmp_path, **kwargs))
    return pl.read_parquet(tmp_path / "out" / "input.parquet")


def test_main_output_contains_meta_num_cells(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert "meta_num_cells" in result.columns


def test_main_meta_num_cells_correct(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert (result["meta_num_cells"] == 4).all()


def test_main_output_contains_barcode_num_unique(tmp_path) -> None:
    result = _run_main(tmp_path)
    assert f"{META_BARCODE_COL}_num_unique" in result.columns


def test_main_meta_barcode_num_unique_correct(tmp_path) -> None:
    # write_feat_input_parquet alternates bc_0 / bc_1 → 2 unique per variant
    result = _run_main(tmp_path)
    assert (result[f"{META_BARCODE_COL}_num_unique"] == 2).all()


def test_main_output_contains_batch_num_unique(tmp_path) -> None:
    # meta_batch is added by load_batches from the filename stem
    result = _run_main(tmp_path)
    assert f"{META_BATCH_COL}_num_unique" in result.columns


def test_main_meta_batch_num_unique_is_one_for_single_file(tmp_path) -> None:
    # single input file → all cells share the same batch label
    result = _run_main(tmp_path)
    assert (result[f"{META_BATCH_COL}_num_unique"] == 1).all()


# Deliberately lopsided: A1C's value is an order of magnitude off the rest, so
# a synonymous-baseline z-score would be unmistakable if one were ever applied.
PASSTHROUGH_VALUES = [0.0, 0.0, 0.0, 1.0, 40.0]


def write_passthrough_aggregate(tmp_path, *, column: str = "f1_KSnegLogP") -> None:
    """A passthrough per-feature-type aggregate, in its own directory — the
    Nextflow side publishes these to passthrough_aggregates/, not aggregates/."""
    pt_dir = tmp_path / "pt"
    pt_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "meta_aa_changes": ["A1A", "A2A", "A3A", "A1B", "A1C"],
            column: PASSTHROUGH_VALUES,
        }
    ).write_parquet(pt_dir / "KSnegLogP.parquet")


def _run_main_with_passthrough(tmp_path, **kwargs) -> pl.DataFrame:
    write_passthrough_aggregate(tmp_path)
    return _run_main(
        tmp_path,
        passthrough_feature_type_files=str(tmp_path / "pt" / "*.parquet"),
        **kwargs,
    )


def test_main_passthrough_column_present_in_output(tmp_path) -> None:
    result = _run_main_with_passthrough(tmp_path)
    assert "f1_KSnegLogP" in result.columns


def test_main_passthrough_values_are_not_normalized(tmp_path) -> None:
    result = _run_main_with_passthrough(tmp_path).sort("meta_aa_changes")
    expected = pl.read_parquet(tmp_path / "pt" / "KSnegLogP.parquet").sort(
        "meta_aa_changes"
    )["f1_KSnegLogP"]
    assert result["f1_KSnegLogP"].to_list() == pytest.approx(expected.to_list())


def test_main_selected_features_are_normalized(tmp_path) -> None:
    """Control for the test above: the selected features on the same run *are*
    z-scored, so the passthrough column's raw values are not an artifact of
    the normalizer being a no-op here."""
    result = _run_main_with_passthrough(tmp_path).sort("meta_aa_changes")
    ft_values = pl.read_parquet(tmp_path / "ft" / "mean.parquet").sort(
        "meta_aa_changes"
    )["f1_mean"]
    assert result["f1_mean"].to_list() != pytest.approx(ft_values.to_list())


def test_main_passthrough_column_survives_the_blocklist(tmp_path) -> None:
    """A passthrough column named in the blocklist is still kept — the
    blocklist is a reproducibility verdict, and passthrough types never had a
    bootstrap to earn one."""
    write_feat_input_parquet(tmp_path)
    write_feature_type_aggregate(tmp_path)
    write_passthrough_aggregate(tmp_path)
    pl.DataFrame(
        {
            "feature": ["f1_mean", "f2_mean", "f1_KSnegLogP"],
            "feature_ok": [True, True, False],
        }
    ).write_parquet(tmp_path / "blocklist.parquet")
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        m.main.__wrapped__(
            make_feat_cfg(
                tmp_path,
                passthrough_feature_type_files=str(tmp_path / "pt" / "*.parquet"),
            )
        )
    result = pl.read_parquet(tmp_path / "out" / "input.parquet")
    assert "f1_KSnegLogP" in result.columns


def test_main_passthrough_excluded_from_impact_score(tmp_path) -> None:
    baseline = _run_main(tmp_path).sort("meta_aa_changes")
    with_passthrough = _run_main_with_passthrough(tmp_path).sort("meta_aa_changes")
    assert with_passthrough[IMPACT_SCORE_COL].to_list() == pytest.approx(
        baseline[IMPACT_SCORE_COL].to_list()
    )


def test_main_no_passthrough_matches_baseline(tmp_path) -> None:
    """passthrough_feature_type_files=None is exactly today's behaviour."""
    baseline = _run_main(tmp_path)
    assert "f1_KSnegLogP" not in baseline.columns


def test_main_empty_passthrough_glob_warns_and_continues(tmp_path, caplog) -> None:
    """An empty passthrough list reaches the stage as an empty staging dir —
    a warning, not the ValueError feature_type_files raises."""
    (tmp_path / "pt").mkdir(parents=True, exist_ok=True)
    with caplog.at_level(logging.WARNING):
        result = _run_main(
            tmp_path, passthrough_feature_type_files=str(tmp_path / "pt" / "*.parquet")
        )
    assert "f1_mean" in result.columns
    assert any("passthrough glob" in record.message for record in caplog.records)


def test_main_passthrough_column_collision_raises(tmp_path) -> None:
    """A type in both lists would collide on the join; Nextflow rejects that
    up front, and this is the backstop for a direct CLI invocation."""
    _write_default_fixtures(tmp_path)
    write_passthrough_aggregate(tmp_path, column="f1_mean")
    with patch("fisseq_data_pipeline.featureselect.setup_logging"):
        with pytest.raises(ValueError, match="collide"):
            m.main.__wrapped__(
                make_feat_cfg(
                    tmp_path,
                    passthrough_feature_type_files=str(tmp_path / "pt" / "*.parquet"),
                )
            )
