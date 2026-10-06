"""QC_FILTER: edit-distance and barcode-count QC filtering of cell data.

Reads one or more cell files (CSV or Parquet), renames the barcode / amino-acid-change /
edit-distance columns to their canonical ``meta_*`` names, and applies sequential
edit-distance, barcode-count and variant-barcode-count filters. Writes
``filtered_cells.parquet``, ``barcode_counts.parquet`` and ``variants_per_barcode.parquet``.

If ``n_variants`` is set, variants whose classified label is in ``variant_downsample_classes``
(default ``["Single Missense"]``) are restricted to at most ``n_variants`` distinct variants
before QC thresholding runs -- either the highest-cell-count variants (``"top"`` mode) or a
seeded random sample (``"random"`` mode); ``variant_allow_list_file`` names variants that
bypass that cap. See :func:`select_variants`. Disabled by default.

If ``downsample_amounts`` is set, ``filtered_cells.parquet`` also gets reproducibly
downsampled "pseudo variant" rows (``:downsample-{amount}``-tagged), built from the QC-passing
cells of ``downsample_classes`` (:func:`add_downsampled_pseudo_variants`). Disabled by default
in both pipelines.

Entry point: ``python -m fisseq_common.stages.qcfilter``. Each pipeline's
``conf/modules.config`` sets the input column names (raw starcall CSV names in the data
pipeline, the already-renamed ``meta_*`` names of the embeddings pipeline's
``metadata.parquet``), ``sort_output_by`` and ``assign_cell_index``.
"""

import dataclasses
import logging
import pathlib
from os import PathLike
from typing import Any, Iterable, List, Optional, Tuple, Union

import polars as pl
from omegaconf import MISSING

from fisseq_common.schema import (
    META_BARCODE_COL,
    META_CELL_INDEX_COL,
    META_EDIT_DISTANCE_COL,
    META_VARIANT_TAG_COL,
)
from fisseq_common.variant import classify_variant

from .config import AppConfig, stage_main

DOWNSAMPLE_TAG = "downsample"
DOWNSAMPLE_CLASSES = ("Synonymous", "Single Missense")
VARIANT_DOWNSAMPLE_CLASSES = ("Single Missense",)
VARIANT_DOWNSAMPLE_MODES = ("top", "random")


@dataclasses.dataclass
class QcFilterParams(AppConfig):
    """
    QC_FILTER's configuration; each pipeline's ``conf/modules.config`` sets its
    input column names.

    Extends AppConfig (output_dir, output_root, log_level, random_seed); ``select_variants``'
    ``"random"`` mode and the pseudo-variant downsampling consume ``random_seed``.

    Attributes
    ----------
    cell_files : Any
        Path or list of paths to cell data files (CSV or Parquet).
    bc_threshold : int
        Minimum number of cells required for a barcode to pass QC.
        Defaults to ``10``.
    variant_bc_threshold : int
        Minimum number of unique barcodes required for a variant to pass QC.
        Defaults to ``4``.
    edit_distance_threshold : int
        Maximum edit distance allowed for a cell to pass QC. Defaults to
        ``1``.
    barcode_col_name : str
        Name of the barcode column in the input data. Defaults to
        ``"meta_barcode"``.
    aa_changes_col_name : str
        Name of the amino-acid changes column in the input data. Defaults
        to ``"meta_aa_changes"``.
    edit_distance_col_name : str
        Name of the edit distance column in the input data. Defaults to
        ``"meta_edit_distance"``.
    label_column : str
        Name of the output label column after renaming. Defaults to
        ``"meta_aa_changes"``.
    n_variants : Optional[int]
        If set, restricts variants whose classified label is in
        ``variant_downsample_classes`` to at most this many distinct
        variants (see :func:`select_variants`); every other class passes
        through untouched. Runs before QC thresholding. Defaults to
        ``None`` (disabled).
    variant_downsample_classes : List[str]
        Classes (from :func:`fisseq_common.variant.classify_variant`)
        eligible for the ``n_variants`` restriction. Defaults to
        ``["Single Missense"]``.
    variant_downsample_mode : str
        ``"top"`` keeps the ``n_variants`` variants with the highest cell
        count (ties broken alphabetically); ``"random"`` keeps a seeded
        random sample of ``n_variants`` variants (seeded by
        ``AppConfig.random_seed``). Defaults to ``"top"``.
    variant_allow_list_file : Optional[str]
        Optional path to a Parquet file with a ``label_column`` column of
        variants that bypass the ``n_variants`` cap entirely and aren't
        counted against it (see :func:`select_variants`). Entries not
        present in the data are silently ignored. Meaningless if
        ``n_variants`` is unset -- in that case it is ignored with a
        warning, since the two fields are set independently. Defaults to
        ``None`` (disabled).
    downsample_amounts : Any
        A single float/int, or a list of floats/ints. Each item generates
        reproducibly downsampled "pseudo variant" rows (see
        :func:`add_downsampled_pseudo_variants`) from cells whose
        classified label is in ``downsample_classes`` and that survived QC
        filtering: a float in ``(0, 1]`` keeps that fraction per variant, an
        int keeps that many cells per variant (skipping variants with fewer
        cells than that). Typed ``Any`` (like ``cell_files``) since OmegaConf
        structured configs can't express ``Union[float, int, List[...]]``.
        Defaults to ``None`` (disabled).
    downsample_classes : List[str]
        Classes eligible for ``downsample_amounts`` pseudo-variant
        generation. Defaults to ``["Synonymous", "Single Missense"]``.
    sort_output_by : List[str], optional
        Columns to sort ``filtered_cells`` by (see :func:`run_qc_filter`). Defaults to
        ``None``: the join order.
    assign_cell_index : bool
        Assign ``meta_cell_index`` from the input row order (see :func:`run_qc_filter`).
        Defaults to ``False``.
    """

    cell_files: Any = MISSING
    bc_threshold: int = 10
    variant_bc_threshold: int = 4
    edit_distance_threshold: int = 1
    barcode_col_name: str = "meta_barcode"
    aa_changes_col_name: str = "meta_aa_changes"
    edit_distance_col_name: str = "meta_edit_distance"
    label_column: str = "meta_aa_changes"
    n_variants: Optional[int] = None
    variant_downsample_classes: List[str] = dataclasses.field(
        default_factory=lambda: list(VARIANT_DOWNSAMPLE_CLASSES)
    )
    variant_downsample_mode: str = "top"
    variant_allow_list_file: Optional[str] = None
    downsample_amounts: Any = None
    downsample_classes: List[str] = dataclasses.field(
        default_factory=lambda: list(DOWNSAMPLE_CLASSES)
    )
    sort_output_by: Optional[List[str]] = None
    assign_cell_index: bool = False


# --- read_file / combine_cell_files ---


def read_file(cell_file_path: pathlib.Path) -> pl.LazyFrame:
    """
    Read a single cell file into a lazy frame.

    Supports CSV and Parquet formats (extensions ``.csv``, ``.parquet``,
    ``.parq``, ``.pq``). Adds two metadata columns: ``meta_source_file``
    (file path as a string) and ``meta_source_file_idx`` (row index within
    the source file).

    Parameters
    ----------
    cell_file_path : pathlib.Path
        Path to the cell data file.

    Returns
    -------
    pl.LazyFrame
        Lazy frame of the file contents with metadata columns.

    Raises
    ------
    ValueError
        If ``cell_file_path``'s suffix isn't a recognized CSV/Parquet
        extension.
    """
    logging.info("Scanning file %s", cell_file_path)

    if cell_file_path.suffix == ".csv":
        logging.warning(
            "Lazy evaluation is much slower for CSV files."
            " Consider converting to Parquet for better performance."
        )
        lf = pl.scan_csv(cell_file_path)
    elif cell_file_path.suffix in [".parquet", ".parq", ".pq"]:
        lf = pl.scan_parquet(cell_file_path)
    else:
        raise ValueError(
            f"Unrecognized cell_files suffix {cell_file_path.suffix!r} for "
            f"{cell_file_path} -- expected one of .csv, .parquet, .parq, .pq"
        )

    lf = lf.with_columns(
        pl.lit(str(cell_file_path)).alias("meta_source_file"),
        pl.row_index().alias("meta_source_file_idx"),
    )

    return lf


def combine_cell_files(
    cell_files: Iterable[PathLike], assign_cell_index: bool = False
) -> pl.LazyFrame:
    """
    Read and concatenate multiple cell files into one lazy frame.

    Parameters
    ----------
    cell_files : Iterable[PathLike]
        Iterable of paths to cell data files.
    assign_cell_index : bool
        If ``True``, assign ``META_CELL_INDEX_COL`` over the concatenation in
        ``cell_files`` order, giving every cell a stable identity for the rest of
        the pipeline (the data pipeline's cells have no other key). Defaults to
        ``False``.

    Returns
    -------
    pl.LazyFrame
        Concatenated lazy frame of all input files.
    """
    lf = pl.concat([read_file(pathlib.Path(cell_file)) for cell_file in cell_files])
    if not assign_cell_index:
        return lf
    return lf.with_row_index(name=META_CELL_INDEX_COL).with_columns(
        pl.col(META_CELL_INDEX_COL).cast(pl.Int64)
    )


# --- filter_columns ---


def filter_columns(lf: pl.LazyFrame, cfg: QcFilterParams) -> pl.LazyFrame:
    """
    Rename input columns to canonical meta names and retain only QC-relevant columns.

    Renames ``cfg.barcode_col_name`` -> ``META_BARCODE_COL`` and
    ``cfg.edit_distance_col_name`` -> ``META_EDIT_DISTANCE_COL``.
    ``cfg.aa_changes_col_name`` is split on the first ``":"`` into a base
    label (``cfg.label_column``) and an optional tag (``META_VARIANT_TAG_COL``,
    ``null`` when no tag is present). Then drops all columns except meta
    columns (``meta_`` prefix) and CellProfiler feature columns (starting
    with an uppercase letter and containing ``_``) -- the latter are a
    no-op against metadata.parquet, which never has such columns, but are
    kept for parity with cell_files inputs that do (e.g. a raw upstream
    cell table).

    Parameters
    ----------
    lf : pl.LazyFrame
        Lazy frame containing all raw input columns.
    cfg : QcFilterParams
        Supplies column name mappings.

    Returns
    -------
    pl.LazyFrame
        Lazy frame retaining only the necessary columns with canonical names.
    """
    aa_changes_split = pl.col(cfg.aa_changes_col_name).str.splitn(":", 2)
    lf = lf.with_columns(
        aa_changes_split.struct.field("field_0").alias(cfg.label_column),
        aa_changes_split.struct.field("field_1").alias(META_VARIANT_TAG_COL),
        pl.col(cfg.edit_distance_col_name).alias(META_EDIT_DISTANCE_COL),
        pl.col(cfg.barcode_col_name).alias(META_BARCODE_COL),
    )

    schema_names = lf.collect_schema().names()
    cell_profiler_columns = [
        col for col in schema_names if len(col) > 0 and col[0].isupper() and "_" in col
    ]
    meta_columns = [col for col in schema_names if col.startswith("meta_")]

    return lf.select(pl.col(meta_columns + cell_profiler_columns))


# --- get_barcode_counts / get_barcodes_per_variant / add_qc_queries ---


def get_barcode_counts(lf: pl.LazyFrame, cfg: QcFilterParams) -> pl.LazyFrame:
    """
    Count cells per barcode and flag barcodes meeting the threshold.

    Groups by ``META_BARCODE_COL``, counts occurrences, and adds a
    ``barcode_ok`` column (non-null when count >= ``cfg.bc_threshold``).
    Retains the first ``cfg.label_column`` per barcode group.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level lazy frame containing ``META_BARCODE_COL`` and
        ``cfg.label_column`` (as produced by :func:`filter_columns`).
    cfg : QcFilterParams
        Supplies ``bc_threshold`` and ``label_column``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with one row per barcode, including ``count``,
        ``cfg.label_column``, and ``barcode_ok``.
    """
    return (
        lf.group_by(META_BARCODE_COL)
        .agg(
            [
                pl.len().alias("count"),
                pl.col(cfg.label_column).first(),
            ]
        )
        .with_columns(
            pl.when(pl.col("count") >= cfg.bc_threshold)
            .then(pl.col("count"))
            .otherwise(None)
            .alias("barcode_ok")
        )
    )


def get_barcodes_per_variant(
    cells_lf: pl.LazyFrame, cfg: QcFilterParams
) -> pl.LazyFrame:
    """
    Count distinct barcodes per variant and flag variants meeting threshold.

    Groups by ``cfg.label_column``, counts barcodes, and adds a
    ``variant_barcode_count_ok`` column (non-null when barcode count
    >= ``cfg.variant_bc_threshold``).

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        Cell-level lazy frame containing ``META_BARCODE_COL`` and
        ``cfg.label_column`` (as produced by :func:`filter_columns`).
    cfg : QcFilterParams
        Supplies ``variant_bc_threshold`` and ``label_column``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with one row per variant, including ``barcode_count``
        and ``variant_barcode_count_ok``.
    """
    return (
        cells_lf.group_by(cfg.label_column)
        .agg(
            [
                pl.col(META_BARCODE_COL).n_unique().alias("barcode_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("barcode_count") >= cfg.variant_bc_threshold)
            .then(pl.col("barcode_count"))
            .otherwise(None)
            .alias("variant_barcode_count_ok")
        )
    )


def add_qc_queries(
    lf: pl.LazyFrame, cfg: QcFilterParams
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Apply edit-distance, barcode-level, and variant-level QC filters.

    Filters are applied sequentially:

    1. Retain rows with ``META_EDIT_DISTANCE_COL`` <= ``cfg.edit_distance_threshold``.
    2. Remove barcodes below ``cfg.bc_threshold`` cell count.
    3. Remove variants below ``cfg.variant_bc_threshold`` barcode count.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level lazy frame to filter.
    cfg : QcFilterParams
        Supplies QC thresholds and column names.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]
        ``(filtered_lf, barcode_count_lf, variants_per_barcode_lf)`` where
        the latter two contain the intermediate QC summary frames.
    """
    logging.info("Adding edit distance QC query")
    lf = lf.filter(pl.col(META_EDIT_DISTANCE_COL) <= cfg.edit_distance_threshold)

    logging.info("Adding Barcode Level QC query")
    barcode_count_lf = get_barcode_counts(lf, cfg)
    lf = lf.join(
        barcode_count_lf.filter(pl.col("barcode_ok").is_not_null()).select(
            META_BARCODE_COL
        ),
        on=META_BARCODE_COL,
        how="inner",
    )

    logging.info("Adding Variant Level QC query")
    variants_per_barcode_lf = get_barcodes_per_variant(lf, cfg)
    lf = lf.join(
        variants_per_barcode_lf.filter(
            pl.col("variant_barcode_count_ok").is_not_null()
        ).select(cfg.label_column),
        on=cfg.label_column,
        how="inner",
    )

    return lf, barcode_count_lf, variants_per_barcode_lf


# --- select_variants ---


def select_variants(
    lf: pl.LazyFrame,
    cfg: QcFilterParams,
    variant_downsample_classes: Tuple[str, ...],
    n_variants: int,
    mode: str,
    seed: int,
    variant_allow_list_file: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Restrict rows whose classified ``cfg.label_column`` value is in
    `variant_downsample_classes` to at most `n_variants` distinct variants.
    Rows in any other class are left untouched (not filtered at all).

    `mode` is one of:

    - ``"top"``: keep the `n_variants` variants (within
      `variant_downsample_classes`) with the highest cell count, ties broken
      alphabetically ascending on ``cfg.label_column``. Fully deterministic.
    - ``"random"``: keep a seeded-random sample of `n_variants` distinct
      variants from the eligible pool. Deterministic given `seed`.

    If `variant_allow_list_file` is set, it must point to a Parquet file
    with a ``cfg.label_column`` column. The eligible pool is partitioned
    before the `mode` logic runs: rows whose ``cfg.label_column`` value
    appears in that file pass through unconditionally and are *not* counted
    against `n_variants`; the remaining eligible rows go through the
    existing top/random selection, capped at `n_variants`. Allow-list
    entries absent from the data are silently ignored (a left-semi join
    handles this naturally). If every eligible variant is allow-listed,
    `n_variants` has no effect for this run and a warning is logged.

    Runs upstream of :func:`add_qc_queries`, so ``barcode_counts``/
    ``variants_per_barcode`` reflect the post-selection population.

    Raises
    ------
    ValueError
        If `mode` isn't one of ``VARIANT_DOWNSAMPLE_MODES``.
    """
    if mode not in VARIANT_DOWNSAMPLE_MODES:
        raise ValueError(
            f"variant_downsample_mode must be one of {VARIANT_DOWNSAMPLE_MODES}, "
            f"got {mode!r}"
        )

    logging.info(
        "Restricting classes %s to %d variant(s) (mode=%s, seed=%d)",
        variant_downsample_classes,
        n_variants,
        mode,
        seed,
    )
    classified = lf.with_columns(
        pl.col(cfg.label_column)
        .map_elements(classify_variant, return_dtype=pl.String)
        .alias("_variant_class")
    )
    non_eligible = classified.filter(
        ~pl.col("_variant_class").is_in(variant_downsample_classes)
    )
    eligible = classified.filter(
        pl.col("_variant_class").is_in(variant_downsample_classes)
    )

    if variant_allow_list_file is not None:
        allow_list_lf = (
            pl.read_parquet(variant_allow_list_file)
            .select(cfg.label_column)
            .unique()
            .lazy()
        )
        allow_listed = eligible.join(allow_list_lf, on=cfg.label_column, how="semi")
        selection_pool = eligible.join(allow_list_lf, on=cfg.label_column, how="anti")

        n_eligible_variants = (
            eligible.select(pl.col(cfg.label_column).n_unique()).collect().item()
        )
        n_pool_variants = (
            selection_pool.select(pl.col(cfg.label_column).n_unique()).collect().item()
        )
        if n_eligible_variants > 0 and n_pool_variants == 0:
            logging.warning(
                "All %d eligible variant(s) matched variant_allow_list_file; "
                "n_variants=%d has no effect for this run",
                n_eligible_variants,
                n_variants,
            )
    else:
        allow_listed = None
        selection_pool = eligible

    counts = selection_pool.group_by(cfg.label_column).agg(pl.len().alias("_n_cells"))
    if mode == "top":
        selected = (
            counts.sort(["_n_cells", cfg.label_column], descending=[True, False])
            .head(n_variants)
            .select(cfg.label_column)
        )
    else:
        selected = (
            counts.with_columns(pl.col(cfg.label_column).hash(seed=seed).alias("_rand"))
            .sort("_rand")
            .head(n_variants)
            .select(cfg.label_column)
        )

    selection_kept = selection_pool.join(selected, on=cfg.label_column, how="inner")

    parts = [non_eligible]
    if allow_listed is not None:
        parts.append(allow_listed)
    parts.append(selection_kept)

    return pl.concat(parts, how="vertical_relaxed").drop("_variant_class")


def add_downsampled_pseudo_variants(
    lf: pl.LazyFrame,
    cfg: QcFilterParams,
    downsample_classes: tuple[str, ...],
    downsample_amount: Union[float, int],
    seed: int,
) -> pl.LazyFrame:
    """
    For QC-surviving rows whose classified ``cfg.label_column`` value is in
    `downsample_classes`, reproducibly keep either a target fraction or a
    target count of each variant's rows (per ``cfg.label_column`` group) and
    mark them as a "pseudo variant" by appending ``:{tag}`` directly onto
    ``cfg.label_column`` (e.g. ``"A326P"`` becomes ``"A326P:downsample-0.5"``
    for a float amount of ``0.5``, or ``"A326P:downsample-500"`` for an int
    amount of ``500``). This gives pseudo-variant rows their own distinct
    label value, so downstream label-based grouping (e.g. aggregation)
    treats them as a separate calibration group rather than pooling them
    with the real variant's rows. The tag also becomes the row's
    ``META_VARIANT_TAG_COL``: a pseudo-variant row is its source cell under
    another tag, which is how the stages downstream tell the two rows apart
    (:func:`fisseq_common.stages.config.row_keys`).

    `downsample_amount` is interpreted per ``cfg.label_column`` group:

    - A ``float`` in ``(0, 1]`` keeps ``floor(downsample_amount * group_size)``
      rows from every eligible group (a group may shrink to 0 pseudo rows if
      it's small, but is never skipped outright).
    - An ``int`` >= 1 keeps exactly ``downsample_amount`` rows from every
      eligible group whose size is >= ``downsample_amount`` — groups with
      fewer rows than that are skipped entirely for this amount (no
      clamping, no partial keep).

    To generate pseudo rows for multiple amounts, call this once per amount
    and concatenate the results — see :func:`run_qc_filter`.

    Selection is deterministic given `seed`: each row gets a seeded hash of
    its (`meta_source_file`, `meta_source_file_idx`) identity, rows are
    ranked by that hash within their `cfg.label_column` group, and the
    lowest-ranked rows up to the target are kept.

    Raises
    ------
    ValueError
        If `downsample_amount` is a float outside ``(0, 1]`` or a
        non-positive int.
    """
    if isinstance(downsample_amount, float) and not (0 < downsample_amount <= 1):
        raise ValueError(
            f"downsample_amount float must satisfy 0 < x <= 1, got {downsample_amount}"
        )
    if isinstance(downsample_amount, int) and downsample_amount <= 0:
        raise ValueError(
            f"downsample_amount int must be positive, got {downsample_amount}"
        )

    logging.info(
        "Building downsampled pseudo variants for classes %s (amount=%s, seed=%d)",
        downsample_classes,
        downsample_amount,
        seed,
    )
    eligible = lf.filter(
        pl.col(cfg.label_column)
        .map_elements(classify_variant, return_dtype=pl.String)
        .is_in(downsample_classes)
    )

    ranked = eligible.with_columns(
        pl.struct(["meta_source_file", "meta_source_file_idx"])
        .hash(seed=seed)
        .alias("_rand"),
    ).with_columns(
        pl.col("_rand").rank(method="ordinal").over(cfg.label_column).alias("_rank"),
        pl.len().over(cfg.label_column).alias("_group_size"),
    )

    if isinstance(downsample_amount, float):
        target = (pl.col("_group_size") * downsample_amount).floor()
        group_eligible = pl.lit(True)
    else:
        target = pl.lit(downsample_amount)
        group_eligible = pl.col("_group_size") >= downsample_amount

    tag = f"{DOWNSAMPLE_TAG}-{downsample_amount}"
    pseudo = (
        ranked.filter(group_eligible & (pl.col("_rank") <= target))
        .drop(["_rand", "_rank", "_group_size"])
        .with_columns(
            (pl.col(cfg.label_column) + ":" + tag).alias(cfg.label_column),
            pl.lit(tag).alias(META_VARIANT_TAG_COL),
        )
    )
    return pseudo


def run_qc_filter(
    cfg: QcFilterParams,
    sort_output_by: Optional[List[str]] = None,
    assign_cell_index: Optional[bool] = None,
) -> None:
    """
    Run QC_FILTER with ``cfg`` and write its three outputs to ``cfg.output_dir``.

    Parameters
    ----------
    cfg : QcFilterParams
        The stage config (``output_dir`` must exist).
    sort_output_by : list of str, optional
        Columns to sort ``filtered_cells`` by before writing. ``add_qc_queries``'
        inner joins are not order-preserving under Polars' multithreaded execution,
        so without a sort the same input yields the same rows in a different order
        on every run, and every downstream seeded step (OvWT's wildtype downsample
        and fold assignment, the feature-selection bootstrap splits) diverges despite
        a fixed ``random_seed``. Each pipeline sorts on its cell keys plus
        ``META_VARIANT_TAG_COL`` (:func:`fisseq_common.stages.config.row_keys`), which
        is total even with pseudo-variant rows: each carries its downsample amount's
        tag. ``None`` (the default) uses ``cfg.sort_output_by``.
    assign_cell_index : bool, optional
        Assign ``META_CELL_INDEX_COL`` from the input row order (:func:`combine_cell_files`):
        the data pipeline's raw cells have no other identity. The embeddings pipeline's
        input already carries its own per-tile ``meta_cell_index``, so it leaves this off.
        ``None`` (the default) uses ``cfg.assign_cell_index``.

    Notes
    -----
    Output files, with ``prefix`` = ``{output_root}.`` when ``output_root`` is set:

    - ``{prefix}filtered_cells.parquet``
    - ``{prefix}barcode_counts.parquet``
    - ``{prefix}variants_per_barcode.parquet``
    """
    output_dir = pathlib.Path(cfg.output_dir)
    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""
    if sort_output_by is None:
        sort_output_by = cfg.sort_output_by
    if assign_cell_index is None:
        assign_cell_index = cfg.assign_cell_index

    cell_files = (
        [cfg.cell_files] if isinstance(cfg.cell_files, str) else list(cfg.cell_files)
    )
    combined_lf = filter_columns(
        combine_cell_files(cell_files, assign_cell_index=assign_cell_index),
        cfg,
    )

    if cfg.n_variants is not None:
        combined_lf = select_variants(
            combined_lf,
            cfg,
            variant_downsample_classes=tuple(cfg.variant_downsample_classes),
            n_variants=cfg.n_variants,
            mode=cfg.variant_downsample_mode,
            seed=cfg.random_seed,
            variant_allow_list_file=cfg.variant_allow_list_file,
        )
    else:
        logging.info("n_variants not set; skipping variant-level selection")
        if cfg.variant_allow_list_file is not None:
            logging.warning(
                "variant_allow_list_file is set but n_variants is None; there is no "
                "n_variants cap to bypass, so the allow-list is ignored"
            )

    combined_lf, barcode_count_lf, variants_per_barcode_lf = add_qc_queries(
        combined_lf, cfg
    )

    if cfg.downsample_amounts is not None:
        downsample_amounts = (
            cfg.downsample_amounts
            if isinstance(cfg.downsample_amounts, list)
            else [cfg.downsample_amounts]
        )
        logging.info(
            "Adding downsampled pseudo-variant rows for amounts %s (classes=%s)",
            downsample_amounts,
            cfg.downsample_classes,
        )
        pseudo_lfs = [
            add_downsampled_pseudo_variants(
                combined_lf,
                cfg,
                downsample_classes=tuple(cfg.downsample_classes),
                downsample_amount=amount,
                seed=cfg.random_seed,
            )
            for amount in downsample_amounts
        ]
        if pseudo_lfs:
            combined_lf = pl.concat([combined_lf, *pseudo_lfs], how="vertical_relaxed")
    else:
        logging.info("downsample_amounts not set; skipping pseudo-variant generation")

    if sort_output_by:
        combined_lf = combined_lf.sort(sort_output_by, nulls_last=False)

    logging.info("Writing output files to %s", output_dir)
    # The two report tables come from a group_by, whose row order Polars doesn't fix: sort
    # them on their key so a rerun writes identical files.
    for name, lf in [
        ("filtered_cells", combined_lf),
        ("barcode_counts", barcode_count_lf.sort(META_BARCODE_COL)),
        ("variants_per_barcode", variants_per_barcode_lf.sort(cfg.label_column)),
    ]:
        logging.info("Writing %s", name)
        lf.sink_parquet(output_dir / f"{prefix}{name}.parquet")

    logging.info("Done")


main = stage_main("qc_filter_main", QcFilterParams, run_qc_filter)

if __name__ == "__main__":
    main()
