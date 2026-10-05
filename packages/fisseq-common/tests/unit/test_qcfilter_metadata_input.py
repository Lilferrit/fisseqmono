"""QC filter cases on input shaped like the embeddings pipeline's metadata.parquet (``meta_*`` column names).

Complements test_qcfilter.py (raw starcall column names): the unrecognized-suffix error, a
single-path ``cell_files`` and the ``variant_allow_list_file`` cases of ``select_variants``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import polars as pl
import pytest

from fisseq_common.stages.qcfilter import (
    QcFilterParams,
    combine_cell_files,
    filter_columns,
    read_file,
    select_variants,
)


def _cfg(**overrides) -> QcFilterParams:
    base = dict(
        output_dir="/tmp/out",
        cell_files=[],
        bc_threshold=3,
        variant_bc_threshold=2,
        edit_distance_threshold=1,
    )
    base.update(overrides)
    return QcFilterParams(**base)


def _make_cell_df(
    barcodes: List[str],
    aa_changes: List[str],
    edit_distances: Optional[List[int]] = None,
) -> pl.DataFrame:
    """Build a minimal raw cell-level DataFrame, shaped like metadata.parquet."""
    n = len(barcodes)
    return pl.DataFrame(
        {
            "meta_barcode": barcodes,
            "meta_aa_changes": aa_changes,
            "meta_edit_distance": edit_distances
            if edit_distances is not None
            else [0] * n,
        }
    )


def _make_filtered_df(
    barcodes: List[str],
    aa_changes: List[str],
    edit_distances: Optional[List[int]] = None,
    cfg: Optional[QcFilterParams] = None,
) -> pl.DataFrame:
    """Build a cell DataFrame already passed through filter_columns."""
    raw = _make_cell_df(barcodes, aa_changes, edit_distances)
    return filter_columns(raw.lazy(), cfg or _cfg()).collect()


class TestReadFile:
    def test_csv_adds_metadata_columns(self, tmp_path: Path):
        csv_file = tmp_path / "cells.csv"
        pl.DataFrame({"a": [1, 2, 3]}).write_csv(csv_file)

        result = read_file(csv_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_parquet_adds_metadata_columns(self, tmp_path: Path):
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_meta_source_file_value(self, tmp_path: Path):
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file"][0] == str(pq_file)

    def test_meta_source_file_idx_is_sequential(self, tmp_path: Path):
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [10, 20, 30]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file_idx"].to_list() == [0, 1, 2]

    def test_unrecognized_suffix_raises(self, tmp_path: Path):
        """Upstream's if/elif has no else, so an unrecognized suffix raises
        an opaque UnboundLocalError -- this port raises a clear ValueError
        instead (module docstring's read_file fix)."""
        bogus_file = tmp_path / "cells.tsv"
        pl.DataFrame({"a": [1]}).write_csv(bogus_file)

        with pytest.raises(ValueError, match="Unrecognized cell_files suffix"):
            read_file(bogus_file)


class TestCombineCellFiles:
    def test_row_count_is_sum_of_inputs(self, tmp_path: Path):
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(f1)
        pl.DataFrame({"x": [3, 4, 5]}).write_parquet(f2)

        result = combine_cell_files([f1, f2]).collect()

        assert result.shape[0] == 5

    def test_single_file(self, tmp_path: Path):
        f1 = tmp_path / "a.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(f1)

        result = combine_cell_files([f1]).collect()

        assert result.shape[0] == 2


class TestSelectVariants:
    def test_top_mode_keeps_highest_count_variants(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(9)],
            ["M1K"] * 3 + ["M2L"] * 5 + ["M3Q"] * 1,
            cfg=cfg,
        )
        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="top",
            seed=0,
        ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
        assert result.shape[0] == 8

    def test_top_mode_tie_break_is_alphabetical(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1", "bc2"], ["M2L", "M1K"], cfg=cfg)
        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
        ).collect()

        assert result["meta_aa_changes"].to_list() == ["M1K"]

    def test_non_eligible_classes_always_kept(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(7)],
            ["A1A"] * 5 + ["M1K"] + ["M2L"],
            cfg=cfg,
        )
        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
        ).collect()

        assert (result["meta_aa_changes"] == "A1A").sum() == 5
        assert result.shape[0] == 6

    def test_random_mode_is_deterministic_across_calls(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(5)],
            ["M1K", "M2L", "M3Q", "M4R", "M5S"],
            cfg=cfg,
        )
        result1 = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=7,
        ).collect()
        result2 = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=7,
        ).collect()

        assert set(result1["meta_aa_changes"].to_list()) == set(
            result2["meta_aa_changes"].to_list()
        )
        assert result1.shape[0] == 2

    def test_random_mode_different_seed_can_change_selection(self):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(5)],
            ["M1K", "M2L", "M3Q", "M4R", "M5S"],
            cfg=cfg,
        )
        result_a = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=0,
        ).collect()
        result_b = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=1,
        ).collect()

        assert result_a.shape[0] == result_b.shape[0] == 2

    def test_invalid_mode_raises(self):
        cfg = _cfg()
        df = _make_filtered_df(["bc1"], ["M1K"], cfg=cfg)
        with pytest.raises(ValueError):
            select_variants(
                df.lazy(),
                cfg,
                variant_downsample_classes=("Single Missense",),
                n_variants=1,
                mode="bogus",
                seed=0,
            ).collect()

    def test_allow_listed_variants_bypass_cap(self, tmp_path: Path):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(10)],
            ["M1K"] * 5 + ["M2L"] * 3 + ["M3Q"] * 2,
            cfg=cfg,
        )
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M3Q"]}).write_parquet(allow_list_path)

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M3Q"}

    def test_allow_listed_not_counted_against_cap(self, tmp_path: Path):
        cfg = _cfg()
        variants = [f"M{i}K" for i in range(7)]
        barcodes = [f"bc{i}" for i in range(7)]
        df = _make_filtered_df(barcodes, variants, cfg=cfg)
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": variants[:2]}).write_parquet(allow_list_path)

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=3,
            mode="top",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        assert result["meta_aa_changes"].n_unique() == 5
        assert set(variants[:2]) <= set(result["meta_aa_changes"].to_list())

    def test_allow_list_entries_not_in_data_are_ignored(self, tmp_path: Path):
        cfg = _cfg()
        df = _make_filtered_df(["bc0", "bc1"], ["M1K", "M2L"], cfg=cfg)
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M1K", "M99Z"]}).write_parquet(
            allow_list_path
        )

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=1,
            mode="top",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
        assert result.shape[0] == 2

    def test_all_eligible_allow_listed_is_noop_with_warning(
        self, tmp_path: Path, caplog
    ):
        cfg = _cfg()
        df = _make_filtered_df(["bc0", "bc1"], ["M1K", "M2L"], cfg=cfg)
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M1K", "M2L"]}).write_parquet(allow_list_path)

        with caplog.at_level("WARNING"):
            result = select_variants(
                df.lazy(),
                cfg,
                variant_downsample_classes=("Single Missense",),
                n_variants=1,
                mode="top",
                seed=0,
                variant_allow_list_file=str(allow_list_path),
            ).collect()

        assert set(result["meta_aa_changes"].to_list()) == {"M1K", "M2L"}
        assert any("allow_list" in rec.message for rec in caplog.records)

    def test_random_mode_with_allow_list(self, tmp_path: Path):
        cfg = _cfg()
        df = _make_filtered_df(
            [f"bc{i}" for i in range(6)],
            ["M1K", "M2L", "M3Q", "M4R", "M5S", "M6T"],
            cfg=cfg,
        )
        allow_list_path = tmp_path / "allow_list.parquet"
        pl.DataFrame({"meta_aa_changes": ["M1K", "M2L"]}).write_parquet(allow_list_path)

        result = select_variants(
            df.lazy(),
            cfg,
            variant_downsample_classes=("Single Missense",),
            n_variants=2,
            mode="random",
            seed=0,
            variant_allow_list_file=str(allow_list_path),
        ).collect()

        variants = set(result["meta_aa_changes"].to_list())
        assert {"M1K", "M2L"} <= variants
        assert len(variants) == 4  # 2 allow-listed + 2 randomly selected
