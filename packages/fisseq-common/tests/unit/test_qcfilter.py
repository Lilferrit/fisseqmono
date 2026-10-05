import polars as pl
import pytest
from omegaconf import OmegaConf

from fisseq_common.stages.qcfilter import (
    DOWNSAMPLE_TAG,
    add_downsampled_pseudo_variants,
    add_qc_queries,
    combine_cell_files,
    filter_columns,
    get_barcode_counts,
    get_barcodes_per_variant,
    read_file,
    select_variants,
)


@pytest.fixture
def cfg():
    return OmegaConf.create(
        {
            "barcode_col_name": "upBarcode",
            "aa_changes_col_name": "aaChanges",
            "edit_distance_col_name": "editDistance",
            "label_column": "meta_aa_changes",
            "bc_threshold": 3,
            "variant_bc_threshold": 2,
            "edit_distance_threshold": 1,
        }
    )


def _make_cell_df(
    barcodes: list[str],
    aa_changes: list[str],
    edit_distances: list[int] | None = None,
) -> pl.DataFrame:
    """Build a minimal raw cell-level DataFrame for testing."""
    n = len(barcodes)
    return pl.DataFrame(
        {
            "upBarcode": barcodes,
            "aaChanges": aa_changes,
            "variantType": ["missense"] * n,
            "editDistance": edit_distances if edit_distances is not None else [0] * n,
            "Cells_AreaShape_Area": list(range(n, 0, -1)),
            "nuclei_intensity": [1.0] * n,  # dropped: lowercase first char
            "someExtra": ["x"] * n,  # dropped: no underscore
        }
    )


def _make_filtered_df(
    barcodes: list[str],
    aa_changes: list[str],
    edit_distances: list[int] | None = None,
    cfg=None,
) -> pl.DataFrame:
    """Build a cell DataFrame already passed through filter_columns."""
    raw = _make_cell_df(barcodes, aa_changes, edit_distances)
    return filter_columns(raw.lazy(), cfg).collect()


class TestGetBarcodeCounts:
    def test_passing_barcode_has_count(self, cfg):
        """Barcode with count >= bc_threshold gets a non-null barcode_ok."""
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 2, ["V1"] * 5, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        bc1_row = result.filter(pl.col("meta_barcode") == "bc1")
        assert bc1_row["barcode_ok"][0] == 3

    def test_failing_barcode_is_null(self, cfg):
        """Barcode with count < bc_threshold gets a null barcode_ok."""
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 2, ["V1"] * 5, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        bc2_row = result.filter(pl.col("meta_barcode") == "bc2")
        assert bc2_row["barcode_ok"][0] is None

    def test_one_row_per_barcode(self, cfg):
        """Output contains exactly one row per unique barcode."""
        df = _make_filtered_df(["bc1"] * 3 + ["bc2"] * 3, ["V1"] * 6, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect()

        assert result.shape[0] == 2

    def test_counts_are_correct(self, cfg):
        """The count column reflects the true number of cells per barcode."""
        df = _make_filtered_df(["bc1"] * 5 + ["bc2"] * 3, ["V1"] * 8, cfg=cfg)
        result = get_barcode_counts(df.lazy(), cfg).collect().sort("meta_barcode")

        assert result["count"].to_list() == [5, 3]


class TestGetBarcodesPerVariant:
    def test_passing_variant_has_count(self, cfg):
        """Variant with barcode count >= variant_bc_threshold is flagged."""
        # V1 has two distinct barcodes; V2 has one
        df = _make_filtered_df(["bc1", "bc2", "bc3"], ["V1", "V1", "V2"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        v1_row = result.filter(pl.col("meta_aa_changes") == "V1")
        assert v1_row["variant_barcode_count_ok"][0] == 2

    def test_failing_variant_is_null(self, cfg):
        """Variant with barcode count < variant_bc_threshold gets null."""
        df = _make_filtered_df(["bc1", "bc2", "bc3"], ["V1", "V1", "V2"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        v2_row = result.filter(pl.col("meta_aa_changes") == "V2")
        assert v2_row["variant_barcode_count_ok"][0] is None

    def test_one_row_per_variant(self, cfg):
        """Output contains exactly one row per unique variant."""
        df = _make_filtered_df(
            ["bc1", "bc2", "bc3", "bc4"], ["V1", "V1", "V2", "V2"], cfg=cfg
        )
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        assert result.shape[0] == 2

    def test_tagged_and_untagged_rows_pool_together(self, cfg):
        """A tagged variant (e.g. 'V1:downsampled-half') pools with its
        untagged base 'V1' under a single meta_aa_changes group, since
        tag-stripping happens in filter_columns before this function runs."""
        df = _make_filtered_df(["bc1", "bc2"], ["V1", "V1:downsampled-half"], cfg=cfg)
        result = get_barcodes_per_variant(df.lazy(), cfg).collect()

        assert result.shape[0] == 1
        assert result["meta_aa_changes"][0] == "V1"
        assert result["barcode_count"][0] == 2


class TestAddQcQueries:
    @pytest.fixture
    def cell_df(self, cfg):
        """
        bc1: 3 cells for V1, all edit_distance=0 → passes all filters
        bc2: 3 cells for V1, all edit_distance=0 → passes all filters
        bc3: 3 cells for V2, all edit_distance=0 → passes barcode filter,
             but V2 has only 1 passing barcode so fails variant filter
        bc4: 1 cell for V1, edit_distance=0 → fails bc_threshold
        bc5: 1 cell for V1, edit_distance=2 → fails edit distance filter

        After edit filter (<=1): bc5 removed
        After barcode filter (>=3): bc1, bc2, bc3 pass; bc4 removed
        After variant filter (>=2): V1 has bc1+bc2 → passes; V2 has bc3 → fails
        Final: 6 rows (bc1×3 + bc2×3), all V1
        """
        return _make_filtered_df(
            ["bc1"] * 3 + ["bc2"] * 3 + ["bc3"] * 3 + ["bc4"] + ["bc5"],
            ["V1"] * 3 + ["V1"] * 3 + ["V2"] * 3 + ["V1"] + ["V1"],
            [0] * 3 + [0] * 3 + [0] * 3 + [0] + [2],
            cfg=cfg,
        )

    def test_edit_distance_filter(self, cell_df, cfg):
        """Rows with editDistance > threshold are removed."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert result["meta_edit_distance"].max() <= cfg.edit_distance_threshold

    def test_barcode_filter_removes_rare_barcodes(self, cell_df, cfg):
        """Barcodes below bc_threshold do not appear in output."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert "bc4" not in result["meta_barcode"].to_list()
        assert "bc5" not in result["meta_barcode"].to_list()

    def test_variant_filter_removes_rare_variants(self, cell_df, cfg):
        """Variants below variant_bc_threshold do not appear in output."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert "V2" not in result["meta_aa_changes"].to_list()

    def test_passing_cells_are_retained(self, cell_df, cfg):
        """Cells from passing barcodes and variants are kept."""
        filtered, _, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = filtered.collect()
        assert set(result["meta_barcode"].to_list()) == {"bc1", "bc2"}
        assert result.shape[0] == 6  # 3 cells each from bc1 and bc2

    def test_barcode_counts_frame_shape(self, cell_df, cfg):
        """barcode_counts has one row per barcode surviving the edit filter."""
        _, barcode_counts, _ = add_qc_queries(cell_df.lazy(), cfg)
        result = barcode_counts.collect()
        # barcode_counts is built after the edit distance filter, so barcodes
        # that fail that filter (e.g. bc5 with edit_distance=2) are excluded.
        n_post_edit_filter = cell_df.filter(
            pl.col("meta_edit_distance") <= cfg.edit_distance_threshold
        )["meta_barcode"].n_unique()
        assert result.shape[0] == n_post_edit_filter

    def test_variants_per_barcode_frame_shape(self, cell_df, cfg):
        """variants_per_barcode frame has one row per unique variant."""
        _, _, variants_per_barcode = add_qc_queries(cell_df.lazy(), cfg)
        # Only barcodes passing the barcode filter feed into this frame,
        # so the variant count may be less than in the raw data.
        result = variants_per_barcode.collect()
        assert result.shape[0] >= 1


class TestFilterColumns:
    def test_meta_columns_are_created(self, cfg):
        """aaChanges, variantType, editDistance, barcode are aliased to meta_."""
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2)
        result = filter_columns(df.lazy(), cfg).collect()

        for col in (
            "meta_aa_changes",
            "meta_edit_distance",
            "meta_barcode",
        ):
            assert col in result.columns

    def test_cell_profiler_columns_retained(self, cfg):
        """Columns starting with uppercase and containing '_' are kept."""
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2)
        result = filter_columns(df.lazy(), cfg).collect()

        assert "Cells_AreaShape_Area" in result.columns

    def test_non_cell_profiler_columns_dropped(self, cfg):
        """Columns that are not CellProfiler features and not meta_ are dropped."""
        df = _make_cell_df(["bc1"] * 2, ["V1"] * 2)
        result = filter_columns(df.lazy(), cfg).collect()

        assert "nuclei_intensity" not in result.columns
        assert "someExtra" not in result.columns

    def test_tagged_variant_is_split(self, cfg):
        """A ':'-delimited tag is split off into meta_variant_tag."""
        df = _make_cell_df(["bc1"], ["M1K:downsampled-half"])
        result = filter_columns(df.lazy(), cfg).collect()

        assert result["meta_aa_changes"][0] == "M1K"
        assert result["meta_variant_tag"][0] == "downsampled-half"

    def test_untagged_variant_has_null_tag(self, cfg):
        """A variant with no ':' gets a null meta_variant_tag."""
        df = _make_cell_df(["bc1"], ["M1K"])
        result = filter_columns(df.lazy(), cfg).collect()

        assert result["meta_aa_changes"][0] == "M1K"
        assert result["meta_variant_tag"][0] is None


class TestReadFile:
    def test_csv_adds_metadata_columns(self, tmp_path):
        """Reading a CSV adds meta_source_file and meta_origin_file_idx."""
        csv_file = tmp_path / "cells.csv"
        pl.DataFrame({"a": [1, 2, 3]}).write_csv(csv_file)

        result = read_file(csv_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_parquet_adds_metadata_columns(self, tmp_path):
        """Reading a Parquet file adds meta_source_file and meta_source_file_idx."""
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert "meta_source_file" in result.columns
        assert "meta_source_file_idx" in result.columns

    def test_meta_source_file_value(self, tmp_path):
        """meta_source_file contains the path of the file that was read."""
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file"][0] == str(pq_file)

    def test_meta_source_file_idx_is_sequential(self, tmp_path):
        """meta_source_file_idx is a zero-based row index."""
        pq_file = tmp_path / "cells.parquet"
        pl.DataFrame({"a": [10, 20, 30]}).write_parquet(pq_file)

        result = read_file(pq_file).collect()

        assert result["meta_source_file_idx"].to_list() == [0, 1, 2]


class TestCombineCellFiles:
    def test_row_count_is_sum_of_inputs(self, tmp_path):
        """Combined frame has as many rows as all input files together."""
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        pl.DataFrame({"x": [1, 2]}).write_parquet(f1)
        pl.DataFrame({"x": [3, 4, 5]}).write_parquet(f2)

        result = combine_cell_files([f1, f2]).collect()

        assert result.shape[0] == 5

    def test_source_files_are_tracked(self, tmp_path):
        """Each row records which source file it came from."""
        f1 = tmp_path / "a.parquet"
        f2 = tmp_path / "b.parquet"
        pl.DataFrame({"x": [1]}).write_parquet(f1)
        pl.DataFrame({"x": [2]}).write_parquet(f2)

        result = combine_cell_files([f1, f2]).collect()
        sources = set(result["meta_source_file"].to_list())

        assert sources == {str(f1), str(f2)}


def _make_downsample_df(aa_changes: list[str], cfg) -> pl.DataFrame:
    """Build a post-filter_columns-shaped DataFrame with meta_source_file/
    meta_source_file_idx identity columns, as add_qc_queries's output would
    have (via read_file), for testing add_downsampled_pseudo_variants."""
    n = len(aa_changes)
    df = _make_filtered_df([f"bc{i}" for i in range(n)], aa_changes, cfg=cfg)
    return df.with_columns(
        pl.lit("f1").alias("meta_source_file"),
        pl.Series("meta_source_file_idx", list(range(n))),
    )


class TestAddDownsampledPseudoVariants:
    def test_full_fraction_keeps_all_and_tags(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=1.0,
            seed=0,
        ).collect()

        assert result.shape[0] == 10
        # The tag is baked directly into the label, giving pseudo-variant
        # rows their own distinct group rather than pooling with "A1A".
        assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-1.0").all()

    def test_zero_fraction_raises(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        with pytest.raises(ValueError):
            add_downsampled_pseudo_variants(
                df.lazy(),
                cfg,
                downsample_classes=("Synonymous",),
                downsample_amount=0.0,
                seed=0,
            )

    def test_out_of_range_fraction_raises(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        with pytest.raises(ValueError):
            add_downsampled_pseudo_variants(
                df.lazy(),
                cfg,
                downsample_classes=("Synonymous",),
                downsample_amount=1.5,
                seed=0,
            )

    def test_non_positive_int_raises(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        for amount in (0, -5):
            with pytest.raises(ValueError):
                add_downsampled_pseudo_variants(
                    df.lazy(),
                    cfg,
                    downsample_classes=("Synonymous",),
                    downsample_amount=amount,
                    seed=0,
                )

    def test_partial_fraction_is_deterministic_across_calls(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result1 = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=0.5,
            seed=7,
        ).collect()
        result2 = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=0.5,
            seed=7,
        ).collect()

        assert (
            result1["meta_source_file_idx"].to_list()
            == result2["meta_source_file_idx"].to_list()
        )
        assert result1.shape[0] == 5

    def test_different_seed_can_change_selection(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result_a = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=0.5,
            seed=0,
        ).collect()
        result_b = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=0.5,
            seed=1,
        ).collect()

        # Same count either way, but not guaranteed to be the identical subset.
        assert result_a.shape[0] == result_b.shape[0] == 5

    def test_ineligible_class_excluded(self, cfg):
        df = _make_downsample_df(["WT"] * 10, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=1.0,
            seed=0,
        ).collect()

        assert result.shape[0] == 0

    def test_int_amount_keeps_exact_count(self, cfg):
        df = _make_downsample_df(["A1A"] * 10, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=6,
            seed=0,
        ).collect()

        assert result.shape[0] == 6
        assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-6").all()

    def test_int_amount_skips_group_below_threshold(self, cfg):
        df = _make_downsample_df(["A1A"] * 5, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=10,
            seed=0,
        ).collect()

        assert result.shape[0] == 0

    def test_int_amount_mixed_groups_partial_skip(self, cfg):
        df = _make_downsample_df(["A1A"] * 10 + ["A2A"] * 3, cfg)
        result = add_downsampled_pseudo_variants(
            df.lazy(),
            cfg,
            downsample_classes=("Synonymous",),
            downsample_amount=5,
            seed=0,
        ).collect()

        # Only the 10-row A1A group meets the int amount; the 3-row A2A
        # group is skipped entirely (not clamped to its own size).
        assert result.shape[0] == 5
        assert (result["meta_aa_changes"] == f"A1A:{DOWNSAMPLE_TAG}-5").all()


class TestSelectVariants:
    def test_top_mode_keeps_highest_count_variants(self, cfg):
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

    def test_top_mode_tie_break_is_alphabetical(self, cfg):
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

    def test_non_eligible_classes_always_kept(self, cfg):
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

        # All 5 Synonymous rows survive regardless of n_variants; only one
        # of the two single-row Missense variants is kept.
        assert (result["meta_aa_changes"] == "A1A").sum() == 5
        assert result.shape[0] == 6

    def test_random_mode_is_deterministic_across_calls(self, cfg):
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

    def test_random_mode_different_seed_can_change_selection(self, cfg):
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

        # Same count either way, but not guaranteed to be the identical subset.
        assert result_a.shape[0] == result_b.shape[0] == 2

    def test_invalid_mode_raises(self, cfg):
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


def test_combine_cell_files_assigns_stable_cell_index(tmp_path):
    source = tmp_path / "cells.parquet"
    _write_cells(source, [f"bc{i}" for i in range(6)], ["A1A"] * 6)
    lf = combine_cell_files([source], assign_cell_index=True)
    df = lf.collect()
    assert "meta_cell_index" in df.columns
    assert df["meta_cell_index"].to_list() == list(range(6))
