"""The filter (normalization) stage: QC-passed keys plus a control-fitted Normalizer.

No stage copies another stage's data wholesale. The filter stage publishes only

- ``filtered_keys.parquet``: the QC-passed cells' join keys and every ``meta_*`` column,
  including ``meta_is_control``, and no feature columns, and
- ``normalizer.parquet``: per-feature means and standard deviations fitted on the control
  rows (:class:`~fisseq_common.normalizer.Normalizer`).

Every stage that needs the normalized cell table rebuilds it on demand with
:func:`load_filtered_cells` (:func:`load_cells` from a :class:`~.config.CellsInput`): join the
keys to the pipeline's cell feature table and apply the normalizer. The ``meta_*`` columns
always come from QC_FILTER's side (``filtered_keys``), the features from the cell table, so a QC
pseudo-variant row -- a copy of a cell under its own label and tag -- gets its source cell's
features.

Which rows are controls is the ``control`` parameter: a SQL predicate, by default wildtype
(:data:`WT_CONTROL`, both pipelines), or ``"synonymous"`` for untagged synonymous variants
(:func:`variant_classification`). The join keys are the pipeline's cell identity
(:data:`~.config.DATA_JOIN_KEYS`, :data:`~.config.EMBEDDINGS_JOIN_KEYS`); rows are sorted on
:func:`~.config.row_keys`, which adds the variant tag.

Entry point: ``python -m fisseq_common.stages.filter`` (Nextflow ``NORMALIZE``).
"""

import dataclasses
import logging
import pathlib
from typing import List, Optional, Sequence

import polars as pl
from omegaconf import MISSING

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import CONTROL_COLUMN_NAME, META_BATCH_COL, META_SELECTOR
from fisseq_common.variant import classify_variant

from .config import DATA_JOIN_KEYS, AppConfig, CellsInput, row_keys, stage_main

#: ``control`` value selecting untagged synonymous variants as controls.
SYNONYMOUS_CONTROL = "synonymous"

#: The default ``control``: wildtype cells.
WT_CONTROL = "meta_aa_changes = 'WT'"


@dataclasses.dataclass
class FilterParams(AppConfig):
    """
    The filter stage's configuration.

    Attributes
    ----------
    cells_file : str
        The cell feature table (see :class:`~.config.CellsInput`). Required.
    qc_passed_file : str
        QC_FILTER's ``filtered_cells.parquet``. The data pipeline passes it as ``cells_file``
        too: its QC output carries the features. Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    control : str
        Which rows the normalizer is fitted on; see :func:`mark_controls`. Defaults to
        :data:`WT_CONTROL`.
    join_keys : list of str
        The columns identifying a cell in both files. Defaults to
        :data:`~.config.DATA_JOIN_KEYS`.
    batch_name : str, optional
        If set, written as ``meta_batch`` (the data pipeline's QC output has no batch column).
    """

    cells_file: str = MISSING
    qc_passed_file: str = MISSING
    label_column: str = "meta_aa_changes"
    control: str = WT_CONTROL
    join_keys: List[str] = dataclasses.field(
        default_factory=lambda: list(DATA_JOIN_KEYS)
    )
    batch_name: Optional[str] = None


def variant_classification(lf: pl.LazyFrame, label_column: str) -> pl.LazyFrame:
    """
    Mark control (synonymous, untagged) rows via a boolean ``CONTROL_COLUMN_NAME``.

    A row is control when its ``label_column`` value classifies as ``"Synonymous"``
    (:func:`fisseq_common.variant.classify_variant`) *and* carries no ``":<tag>"``
    metadata suffix (e.g. a downsampled pseudo-variant tag) -- tagged rows are never
    treated as control, avoiding double-counting in the Normalizer fit.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame containing ``label_column``.
    label_column : str
        Name of the variant label column to classify.

    Returns
    -------
    pl.LazyFrame
        ``lf`` with an added boolean ``CONTROL_COLUMN_NAME`` column.
    """
    return lf.with_columns(
        (
            pl.col(label_column).map_elements(
                lambda v: classify_variant(v) == "Synonymous", return_dtype=pl.Boolean
            )
            & ~pl.col(label_column).str.contains(":")
        ).alias(CONTROL_COLUMN_NAME)
    )


def mark_controls(lf: pl.LazyFrame, control: str, label_column: str) -> pl.LazyFrame:
    """
    Add the boolean ``CONTROL_COLUMN_NAME`` column selecting ``control`` rows.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level frame containing ``label_column``.
    control : str
        ``"synonymous"`` (:data:`SYNONYMOUS_CONTROL`) for untagged synonymous variants,
        otherwise a SQL predicate evaluated against ``lf`` (e.g.
        ``"meta_aa_changes = 'WT'"``).
    label_column : str
        Name of the variant label column (used by the synonymous rule).
    """
    if control == SYNONYMOUS_CONTROL:
        return variant_classification(lf, label_column)
    return lf.with_columns(pl.sql_expr(control).alias(CONTROL_COLUMN_NAME))


def filter_and_fit_normalizer(
    cells_lf: pl.LazyFrame,
    qc_passed_lf: pl.LazyFrame,
    label_column: str,
    join_keys: Sequence[str],
    control: str = WT_CONTROL,
    sort_by: Optional[Sequence[str]] = None,
) -> "tuple[pl.LazyFrame, Normalizer]":
    """
    Determine the QC-passed join keys and fit the control z-score -- no feature data copied.

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        The pipeline's cell feature table: ``join_keys`` and features (its other ``meta_*``
        columns are ignored).
    qc_passed_lf : pl.LazyFrame
        QC_FILTER's ``filtered_cells.parquet``, the source of every ``meta_*`` column; must
        contain ``join_keys``. May be ``cells_lf`` itself when the QC output carries the
        features.
    label_column : str
        Name of the variant label column.
    join_keys : sequence of str
        Columns identifying a cell in both frames.
    control : str
        Which rows are controls; see :func:`mark_controls`. Defaults to wildtype.
    sort_by : sequence of str, optional
        Columns to sort ``filtered_keys`` by, for a reproducible row order (the join
        doesn't preserve one). ``None`` keeps the join order.

    Returns
    -------
    tuple[pl.LazyFrame, Normalizer]
        ``(filtered_keys, normalizer)``: ``filtered_keys`` holds every ``meta_*`` column of
        the QC-passed cells, including ``CONTROL_COLUMN_NAME``, and no features;
        ``normalizer`` is fit only on the control rows.
    """
    filtered = _join_features(qc_passed_lf.select(META_SELECTOR), cells_lf, join_keys)
    filtered = mark_controls(filtered, control, label_column)
    normalizer = Normalizer.from_lazyframe(filtered, fit_only_on_control=True)
    filtered_keys = filtered.select(META_SELECTOR)
    if sort_by:
        filtered_keys = filtered_keys.sort(list(sort_by), nulls_last=False)
    return filtered_keys, normalizer


def _join_features(
    meta_lf: pl.LazyFrame, cells_lf: pl.LazyFrame, join_keys: Sequence[str]
) -> pl.LazyFrame:
    """``meta_lf`` with ``cells_lf``'s other columns joined on by ``join_keys`` (inner): the
    columns both frames have come from ``meta_lf``."""
    join_keys = list(join_keys)
    meta_columns = set(meta_lf.collect_schema().names())
    cell_columns = [
        c
        for c in cells_lf.collect_schema().names()
        if c in join_keys or c not in meta_columns
    ]
    return meta_lf.join(
        cells_lf.select(cell_columns), on=join_keys, how="inner", nulls_equal=True
    )


def load_filtered_cells(
    cells_lf: pl.LazyFrame,
    filtered_keys_lf: pl.LazyFrame,
    normalizer: Normalizer,
    join_keys: Sequence[str],
    sort_by: Optional[Sequence[str]] = None,
) -> pl.LazyFrame:
    """
    Reconstruct the QC-passed, control-normalized cell table on demand.

    Joins ``cells_lf``'s features onto ``filtered_keys_lf`` by ``join_keys``: every column the
    keys have (the ``meta_*`` columns, ``CONTROL_COLUMN_NAME`` included) comes from the keys,
    the rest from ``cells_lf``, so the join never ``_right``-suffixes. Then applies the
    normalizer to the feature columns.

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        The pipeline's cell feature table.
    filtered_keys_lf : pl.LazyFrame
        The filter stage's ``filtered_keys.parquet``.
    normalizer : Normalizer
        The filter stage's fitted normalizer.
    join_keys : sequence of str
        Columns identifying a cell in both frames.
    sort_by : sequence of str, optional
        Columns to sort the result by. Polars joins are not order-preserving under
        multithreading; a pipeline whose downstream seeded steps depend on row order
        sorts on its cell identity here. ``None`` keeps the join order.

    Returns
    -------
    pl.LazyFrame
        The QC-passed subset of ``cells_lf``, features z-scored against the controls, with
        ``CONTROL_COLUMN_NAME`` added.
    """
    filtered = _join_features(filtered_keys_lf, cells_lf, join_keys)
    if sort_by:
        filtered = filtered.sort(list(sort_by), nulls_last=False)
    return normalizer.apply(filtered)


def write_filter_outputs(
    filtered_keys_lf: pl.LazyFrame, normalizer: Normalizer, cfg: AppConfig
) -> None:
    """
    Write ``{prefix}filtered_keys.parquet`` and ``{prefix}normalizer.parquet`` to
    ``cfg.output_dir``, with ``prefix`` = ``{output_root}.`` when ``output_root`` is set.
    """
    output_dir = pathlib.Path(cfg.output_dir)
    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""

    keys_path = output_dir / f"{prefix}filtered_keys.parquet"
    logging.info("Writing %s", keys_path)
    filtered_keys_lf.sink_parquet(keys_path)

    normalizer_path = output_dir / f"{prefix}normalizer.parquet"
    logging.info("Writing %s", normalizer_path)
    normalizer.save(normalizer_path)

    logging.info("Done")


def load_cells(cfg: CellsInput) -> pl.LazyFrame:
    """The normalized cell table ``cfg`` names (:func:`load_filtered_cells`), sorted on
    :func:`~.config.row_keys`."""
    logging.info(
        "Rebuilding normalized cells from %s, %s and %s",
        cfg.cells_file,
        cfg.filtered_keys_file,
        cfg.normalizer_file,
    )
    return load_filtered_cells(
        pl.scan_parquet(cfg.cells_file),
        pl.scan_parquet(cfg.filtered_keys_file),
        Normalizer.load(cfg.normalizer_file),
        cfg.join_keys,
        sort_by=row_keys(cfg.join_keys),
    )


def run_filter(cfg: FilterParams) -> None:
    """Run the filter stage with ``cfg`` (``output_dir`` must exist): write
    ``filtered_keys.parquet`` and ``normalizer.parquet`` (:func:`write_filter_outputs`)."""
    logging.info("Reading cells from %s", cfg.cells_file)
    cells_lf = pl.scan_parquet(cfg.cells_file)
    logging.info("Reading QC-passed cells from %s", cfg.qc_passed_file)
    qc_passed_lf = pl.scan_parquet(cfg.qc_passed_file)
    if cfg.batch_name is not None:
        qc_passed_lf = qc_passed_lf.with_columns(
            pl.lit(cfg.batch_name).alias(META_BATCH_COL)
        )
    logging.info("Fitting normalizer on rows matching %r", cfg.control)
    filtered_keys_lf, normalizer = filter_and_fit_normalizer(
        cells_lf,
        qc_passed_lf,
        cfg.label_column,
        cfg.join_keys,
        control=cfg.control,
        sort_by=row_keys(cfg.join_keys),
    )
    write_filter_outputs(filtered_keys_lf, normalizer, cfg)


main = stage_main("filter_main", FilterParams, run_filter)

if __name__ == "__main__":
    main()
