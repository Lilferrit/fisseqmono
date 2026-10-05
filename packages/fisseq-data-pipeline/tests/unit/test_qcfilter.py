from unittest.mock import patch

import polars as pl
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

import fisseq_data_pipeline.qcfilter as m
from fisseq_data_pipeline.qcfilter import (
    DOWNSAMPLE_TAG,
    QcFilterConfig,
)


def _write_cells(path, barcodes, aa_changes, edit_distances=None):
    n = len(barcodes)
    pl.DataFrame(
        {
            "upBarcode": barcodes,
            "aaChanges": aa_changes,
            "editDistance": edit_distances if edit_distances is not None else [0] * n,
            "Cells_AreaShape_Area": list(range(n)),
        }
    ).write_parquet(path)


def _make_qc_cfg(tmp_path, cell_files, output_root=None, **overrides):
    files = cell_files if isinstance(cell_files, list) else [cell_files]
    return OmegaConf.structured(
        QcFilterConfig(
            output_dir=str(tmp_path / "out"),
            output_root=output_root,
            cell_files=[str(p) for p in files],
            **overrides,
        )
    )


def test_main_downsample_amounts_none_matches_no_downsampling(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert result.shape[0] == 10
    assert result["meta_variant_tag"].is_null().all()


def test_main_downsample_pseudo_rows_only_from_qc_survivors(tmp_path):
    source = tmp_path / "cells.parquet"
    barcodes = [f"bc{i}" for i in range(10)] + ["bc_fail"]
    aa_changes = ["A1A"] * 11
    # bc_fail has editDistance=5, above threshold=1, so it must be dropped by
    # add_qc_queries *before* any downsampling runs on it.
    edit_distances = [0] * 10 + [5]
    _write_cells(source, barcodes, aa_changes, edit_distances)

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=1.0,
        random_seed=0,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert "bc_fail" not in result["meta_barcode"].to_list()
    # 10 QC survivors, downsampled at fraction=1.0 -> 10 originals + 10 pseudo
    assert result.shape[0] == 20
    assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-1.0").sum() == 10


def test_main_downsample_amounts_single_scalar_equivalent_to_singleton_list(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg_scalar = _make_qc_cfg(
        tmp_path / "run_scalar",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=0.5,
        random_seed=7,
    )
    qc_cfg_list = _make_qc_cfg(
        tmp_path / "run_list",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=[0.5],
        random_seed=7,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg_scalar)
        m.main.__wrapped__(qc_cfg_list)

    result_scalar = pl.read_parquet(
        tmp_path / "run_scalar" / "out" / "filtered_cells.parquet"
    )
    result_list = pl.read_parquet(
        tmp_path / "run_list" / "out" / "filtered_cells.parquet"
    )

    sort_key = ["meta_source_file", "meta_source_file_idx", "meta_aa_changes"]
    assert_frame_equal(result_scalar.sort(sort_key), result_list.sort(sort_key))


def test_main_downsample_amounts_mixed_float_and_int(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=[0.5, 5],
        random_seed=0,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert result.shape[0] == 20  # 10 originals + 5 (float) + 5 (int)
    assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-0.5").sum() == 5
    assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-5").sum() == 5


def test_main_downsample_classes_configurable(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["WT"] * 10)

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=1.0,
        downsample_classes=["WT"],
        random_seed=0,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert result.shape[0] == 20
    assert (result["meta_aa_changes"] == f"WT:{DOWNSAMPLE_TAG}-1.0").sum() == 10


def test_main_downsample_reproducible_with_fixed_seed(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg_a = _make_qc_cfg(
        tmp_path / "run_a",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=0.5,
        random_seed=7,
    )
    qc_cfg_b = _make_qc_cfg(
        tmp_path / "run_b",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=0.5,
        random_seed=7,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg_a)
        m.main.__wrapped__(qc_cfg_b)

    result_a = pl.read_parquet(tmp_path / "run_a" / "out" / "filtered_cells.parquet")
    result_b = pl.read_parquet(tmp_path / "run_b" / "out" / "filtered_cells.parquet")

    sort_key = ["meta_source_file", "meta_source_file_idx", "meta_aa_changes"]
    assert_frame_equal(result_a.sort(sort_key), result_b.sort(sort_key))


def test_main_downsample_barcode_counts_and_variants_per_barcode_exclude_pseudo_rows(
    tmp_path,
):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)

    qc_cfg_with = _make_qc_cfg(
        tmp_path / "run_with",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        downsample_amounts=1.0,
        random_seed=0,
    )
    qc_cfg_without = _make_qc_cfg(
        tmp_path / "run_without",
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg_with)
        m.main.__wrapped__(qc_cfg_without)

    for name in ("barcode_counts", "variants_per_barcode"):
        result_with = pl.read_parquet(tmp_path / "run_with" / "out" / f"{name}.parquet")
        result_without = pl.read_parquet(
            tmp_path / "run_without" / "out" / f"{name}.parquet"
        )
        assert_frame_equal(
            result_with.sort(result_with.columns),
            result_without.sort(result_without.columns),
        )


def test_main_n_variants_none_matches_no_restriction(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(
        source,
        [f"bc{i}" for i in range(6)],
        ["M1K"] * 3 + ["M2L"] * 3,
    )

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M2L"}


def test_main_n_variants_restricts_before_qc_thresholds(tmp_path):
    source = tmp_path / "cells.parquet"
    # M2L has 3 barcodes (passes variant_bc_threshold=2); M1K has only 1
    # barcode so it would fail variant_bc_threshold on its own -- but
    # n_variants=1 in "top" mode should drop M1K (fewer cells) before QC
    # thresholding even runs, so barcode_counts/variants_per_barcode never
    # see it either.
    _write_cells(
        source,
        ["bc0"] + ["bc1", "bc2", "bc3"],
        ["M1K"] + ["M2L"] * 3,
    )

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=2,
        edit_distance_threshold=1,
        n_variants=1,
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert set(result["meta_aa_changes"].to_list()) == {"M2L"}

    variants_per_barcode = pl.read_parquet(
        tmp_path / "out" / "variants_per_barcode.parquet"
    )
    assert "M1K" not in variants_per_barcode["meta_aa_changes"].to_list()


def test_main_variant_downsample_classes_configurable(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(
        source,
        [f"bc{i}" for i in range(6)],
        ["A1A"] * 3 + ["A2A"] * 3,
    )

    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
        n_variants=1,
        variant_downsample_classes=["Synonymous"],
    )

    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)

    result = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    # Both A1A/A2A are Synonymous and tied at 3 cells each; alphabetical
    # tie-break keeps A1A only.
    assert set(result["meta_aa_changes"].to_list()) == {"A1A"}


def test_main_row_order_is_deterministic(tmp_path):
    """
    add_qc_queries' inner joins are not order-preserving under polars'
    multithreaded execution, so main() sorts on meta_cell_index before writing.
    Without that, the same input yields the same rows in a different order every
    run and every downstream seeded step diverges at a fixed random_seed.
    """
    source = tmp_path / "cells.parquet"
    _write_cells(
        source,
        [f"bc{i}" for i in range(10) for _ in range(5)],
        ["A1A"] * 25 + ["M1K"] * 25,
    )

    frames = []
    for i in range(5):
        out = tmp_path / f"out{i}"
        qc_cfg = _make_qc_cfg(
            tmp_path,
            source,
            bc_threshold=1,
            variant_bc_threshold=1,
            edit_distance_threshold=1,
        )
        qc_cfg.output_dir = str(out)
        with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
            m.main.__wrapped__(qc_cfg)
        frames.append(pl.read_parquet(out / "filtered_cells.parquet"))

    assert all(frames[0].equals(f) for f in frames[1:])


def test_main_output_sorted_by_cell_index(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)
    qc_cfg = _make_qc_cfg(
        tmp_path,
        source,
        bc_threshold=1,
        variant_bc_threshold=1,
        edit_distance_threshold=1,
    )
    with patch("fisseq_data_pipeline.qcfilter.setup_logging"):
        m.main.__wrapped__(qc_cfg)
    df = pl.read_parquet(tmp_path / "out" / "filtered_cells.parquet")
    assert df["meta_cell_index"].is_sorted()
