"""QC_FILTER's entry point (``python -m fisseq_common.stages.qcfilter``) as the data pipeline
runs it: raw starcall column names, a ``meta_cell_index`` from the input row order and output
sorted on the row keys. Moved from the data pipeline's ``qcfilter`` wrapper; plus the QC
pseudo-variant rows' identity through the filter stage."""

from unittest.mock import patch

import polars as pl
import pytest
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

import fisseq_common.stages.qcfilter as m
from fisseq_common.stages.config import DATA_JOIN_KEYS, CellsInput
from fisseq_common.stages.filter import FilterParams, load_cells, run_filter
from fisseq_common.stages.qcfilter import DOWNSAMPLE_TAG, QcFilterParams

#: The data pipeline's conf/modules.config ext.args for QC_FILTER.
DATA_ARGS = dict(
    barcode_col_name="upBarcode",
    aa_changes_col_name="aaChanges",
    edit_distance_col_name="editDistance",
    sort_output_by=["meta_cell_index", "meta_variant_tag"],
    assign_cell_index=True,
)

#: Thresholds every cell of the fixtures passes.
PERMISSIVE = dict(bc_threshold=1, variant_bc_threshold=1, edit_distance_threshold=1)

KEYS = list(DATA_JOIN_KEYS)


@pytest.fixture(autouse=True)
def _no_log_files():
    with patch("fisseq_common.stages.config.setup_logging"):
        yield


def _write_cells(path, barcodes, aa_changes, edit_distances=None):
    n = len(barcodes)
    pl.DataFrame(
        {
            "upBarcode": barcodes,
            "aaChanges": aa_changes,
            "editDistance": edit_distances if edit_distances is not None else [0] * n,
            "Cells_AreaShape_Area": [float(i) for i in range(n)],
        }
    ).write_parquet(path)


def _run(out_dir, cell_files, **overrides) -> pl.DataFrame:
    """Run the entry point with the data pipeline's args; return filtered_cells."""
    files = cell_files if isinstance(cell_files, list) else [cell_files]
    fields = dict(
        output_dir=str(out_dir), cell_files=[str(p) for p in files], **DATA_ARGS
    )
    fields.update(overrides)
    m.main.__wrapped__(OmegaConf.structured(QcFilterParams(**fields)))
    return pl.read_parquet(out_dir / "filtered_cells.parquet")


@pytest.fixture
def ten_a1a_cells(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["A1A"] * 10)
    return source


def test_main_writes_three_outputs(tmp_path, ten_a1a_cells):
    _run(tmp_path / "out", ten_a1a_cells, **PERMISSIVE)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.parquet")) == [
        "barcode_counts.parquet",
        "filtered_cells.parquet",
        "variants_per_barcode.parquet",
    ]


def test_main_without_downsample_amounts_adds_no_pseudo_rows(tmp_path, ten_a1a_cells):
    result = _run(tmp_path / "out", ten_a1a_cells, **PERMISSIVE)
    assert result.height == 10
    assert result["meta_variant_tag"].is_null().all()


def test_main_renames_raw_columns_and_assigns_cell_index(tmp_path, ten_a1a_cells):
    result = _run(tmp_path / "out", ten_a1a_cells, **PERMISSIVE)
    assert {"meta_barcode", "meta_aa_changes", "Cells_AreaShape_Area"} <= set(
        result.columns
    )
    assert result["meta_cell_index"].to_list() == list(range(10))


def test_main_downsample_pseudo_rows_only_from_qc_survivors(tmp_path):
    source = tmp_path / "cells.parquet"
    # bc_fail has editDistance=5, above threshold=1, so it must be dropped by
    # add_qc_queries *before* any downsampling runs on it.
    _write_cells(
        source,
        [f"bc{i}" for i in range(10)] + ["bc_fail"],
        ["A1A"] * 11,
        [0] * 10 + [5],
    )
    result = _run(tmp_path / "out", source, downsample_amounts=1.0, **PERMISSIVE)
    assert "bc_fail" not in result["meta_barcode"].to_list()
    # 10 QC survivors, downsampled at fraction=1.0 -> 10 originals + 10 pseudo
    assert result.height == 20
    assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-1.0").sum() == 10


def test_main_pseudo_rows_carry_their_own_variant_tag(tmp_path, ten_a1a_cells):
    """A pseudo-variant row is its source cell (same meta_cell_index) under its downsample
    tag, so (meta_cell_index, meta_variant_tag) still identifies every row."""
    result = _run(
        tmp_path / "out", ten_a1a_cells, downsample_amounts=[0.5, 1.0], **PERMISSIVE
    )
    pseudo = result.filter(pl.col("meta_variant_tag").is_not_null())
    assert pseudo.height == 15
    assert (pseudo["meta_aa_changes"] == "A1A:" + pseudo["meta_variant_tag"]).all()
    assert set(pseudo["meta_variant_tag"]) == {
        f"{DOWNSAMPLE_TAG}-0.5",
        f"{DOWNSAMPLE_TAG}-1.0",
    }
    assert set(pseudo["meta_cell_index"]) <= set(range(10))
    assert not result.select(KEYS).is_duplicated().any()


def test_pseudo_rows_are_not_duplicated_by_the_filter_stage(tmp_path, ten_a1a_cells):
    """End to end: QC_FILTER's pseudo-variant rows survive NORMALIZE and the rebuilt cell
    table one row each, with their source cell's features."""
    qc = _run(tmp_path / "qc", ten_a1a_cells, downsample_amounts=1.0, **PERMISSIVE)
    qc_file = str(tmp_path / "qc" / "filtered_cells.parquet")
    norm_dir = tmp_path / "norm"
    norm_dir.mkdir()
    run_filter(
        FilterParams(
            output_dir=str(norm_dir),
            cells_file=qc_file,
            qc_passed_file=qc_file,
            control="synonymous",
        )
    )
    keys = pl.read_parquet(norm_dir / "filtered_keys.parquet")
    assert keys.height == qc.height == 20

    cells = load_cells(
        CellsInput(
            cells_file=qc_file,
            filtered_keys_file=str(norm_dir / "filtered_keys.parquet"),
            normalizer_file=str(norm_dir / "normalizer.parquet"),
        )
    ).collect()
    assert cells.height == 20
    assert not cells.select(KEYS).is_duplicated().any()
    by_index = cells.pivot(
        on="meta_variant_tag",
        index="meta_cell_index",
        values="Cells_AreaShape_Area",
    )
    assert by_index["null"].to_list() == by_index[f"{DOWNSAMPLE_TAG}-1.0"].to_list()


def test_main_downsample_amounts_scalar_equals_singleton_list(tmp_path, ten_a1a_cells):
    scalar = _run(
        tmp_path / "scalar",
        ten_a1a_cells,
        downsample_amounts=0.5,
        random_seed=7,
        **PERMISSIVE,
    )
    singleton = _run(
        tmp_path / "list",
        ten_a1a_cells,
        downsample_amounts=[0.5],
        random_seed=7,
        **PERMISSIVE,
    )
    assert_frame_equal(scalar, singleton)


def test_main_downsample_amounts_mixed_float_and_int(tmp_path, ten_a1a_cells):
    result = _run(
        tmp_path / "out", ten_a1a_cells, downsample_amounts=[0.5, 5], **PERMISSIVE
    )
    assert result.height == 20  # 10 originals + 5 (float) + 5 (int)
    assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-0.5").sum() == 5
    assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-5").sum() == 5


def test_main_downsample_classes_configurable(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(10)], ["WT"] * 10)
    result = _run(
        tmp_path / "out",
        source,
        downsample_amounts=1.0,
        downsample_classes=["WT"],
        **PERMISSIVE,
    )
    assert result.height == 20
    assert (result["meta_aa_changes"] == f"WT:{DOWNSAMPLE_TAG}-1.0").sum() == 10


def test_main_downsample_is_seeded_by_random_seed(tmp_path, ten_a1a_cells):
    runs = {
        name: _run(
            tmp_path / name,
            ten_a1a_cells,
            downsample_amounts=0.5,
            random_seed=seed,
            **PERMISSIVE,
        )
        for name, seed in (("a", 7), ("b", 7))
    }
    assert_frame_equal(runs["a"], runs["b"])


def test_main_barcode_tables_exclude_pseudo_rows(tmp_path, ten_a1a_cells):
    _run(tmp_path / "with", ten_a1a_cells, downsample_amounts=1.0, **PERMISSIVE)
    _run(tmp_path / "without", ten_a1a_cells, **PERMISSIVE)
    for name in ("barcode_counts", "variants_per_barcode"):
        with_ = pl.read_parquet(tmp_path / "with" / f"{name}.parquet")
        without = pl.read_parquet(tmp_path / "without" / f"{name}.parquet")
        assert_frame_equal(with_.sort(with_.columns), without.sort(without.columns))


def test_main_n_variants_none_is_no_restriction(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(6)], ["M1K"] * 3 + ["M2L"] * 3)
    result = _run(tmp_path / "out", source, **PERMISSIVE)
    assert set(result["meta_aa_changes"]) == {"M1K", "M2L"}


def test_main_n_variants_restricts_before_qc_thresholds(tmp_path):
    source = tmp_path / "cells.parquet"
    # M2L has 3 barcodes (passes variant_bc_threshold=2); M1K has only 1 barcode, so it
    # would fail variant_bc_threshold on its own -- but n_variants=1 in "top" mode drops
    # M1K (fewer cells) before QC thresholding runs, so barcode_counts and
    # variants_per_barcode never see it either.
    _write_cells(source, ["bc0", "bc1", "bc2", "bc3"], ["M1K"] + ["M2L"] * 3)
    result = _run(
        tmp_path / "out",
        source,
        bc_threshold=1,
        variant_bc_threshold=2,
        edit_distance_threshold=1,
        n_variants=1,
    )
    assert set(result["meta_aa_changes"]) == {"M2L"}
    variants_per_barcode = pl.read_parquet(
        tmp_path / "out" / "variants_per_barcode.parquet"
    )
    assert "M1K" not in variants_per_barcode["meta_aa_changes"].to_list()


def test_main_variant_downsample_classes_configurable(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(6)], ["A1A"] * 3 + ["A2A"] * 3)
    result = _run(
        tmp_path / "out",
        source,
        n_variants=1,
        variant_downsample_classes=["Synonymous"],
        **PERMISSIVE,
    )
    # Both are Synonymous and tied at 3 cells; the alphabetical tie-break keeps A1A.
    assert set(result["meta_aa_changes"]) == {"A1A"}


def test_main_row_order_is_deterministic(tmp_path):
    """add_qc_queries' inner joins are not order-preserving under polars' multithreaded
    execution, so the stage sorts on sort_output_by before writing. Without that, the same
    input yields the same rows in a different order every run and every downstream seeded
    step diverges at a fixed random_seed."""
    source = tmp_path / "cells.parquet"
    _write_cells(
        source,
        [f"bc{i}" for i in range(10) for _ in range(5)],
        ["A1A"] * 25 + ["M1K"] * 25,
    )
    frames = [
        _run(tmp_path / f"out{i}", source, downsample_amounts=0.5, **PERMISSIVE)
        for i in range(5)
    ]
    assert all(frames[0].equals(f) for f in frames[1:])
    assert (
        frames[0]
        .select(KEYS)
        .equals(frames[0].select(KEYS).sort(KEYS, nulls_last=False))
    )


def test_main_keeps_the_join_order_without_sort_output_by(tmp_path, ten_a1a_cells):
    """sort_output_by and assign_cell_index default off (the embeddings pipeline's input
    carries its own index); the cfg fields are what the entry point reads."""
    result = _run(
        tmp_path / "out",
        ten_a1a_cells,
        sort_output_by=None,
        assign_cell_index=False,
        **PERMISSIVE,
    )
    assert "meta_cell_index" not in result.columns
