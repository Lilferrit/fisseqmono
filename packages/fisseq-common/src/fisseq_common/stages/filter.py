"""The filter (normalization) stage: QC-passed keys plus a control-fitted Normalizer.

No stage copies another stage's data wholesale. The filter stage publishes only

- ``filtered_keys.parquet``: the QC-passed cells' join keys and every ``meta_*`` column,
  including ``meta_is_control``, and no feature columns, and
- ``normalizer.parquet``: per-feature means and standard deviations fitted on the control
  rows (:class:`~fisseq_common.normalizer.Normalizer`).

Every stage that needs the normalized cell table rebuilds it on demand with
:func:`load_filtered_cells`: join the pipeline's cell feature table back to the keys and apply
the normalizer.

Which rows are controls is the ``control`` parameter: ``"synonymous"`` marks untagged
synonymous variants (:func:`variant_classification`; the embeddings pipeline), anything else is
a SQL predicate (the data pipeline's ``"meta_aa_changes = 'WT'"``). The join keys are the
pipeline's cell identity: ``(meta_batch, meta_well, meta_tile, meta_cell_index)`` for the
embeddings pipeline, ``(meta_cell_index, meta_variant_tag)`` for the data pipeline, where
pseudo-variant rows share their source cell's index.
"""

import dataclasses
import logging
import pathlib
from typing import Optional, Sequence

import polars as pl

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import CONTROL_COLUMN_NAME, META_SELECTOR
from fisseq_common.variant import classify_variant

from .config import AppConfig

#: ``control`` value selecting untagged synonymous variants as controls.
SYNONYMOUS_CONTROL = "synonymous"


@dataclasses.dataclass
class FilterParams(AppConfig):
    """
    Settings shared by both pipelines' filter stages.

    Attributes
    ----------
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    """

    label_column: str = "meta_aa_changes"


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
    control: str = SYNONYMOUS_CONTROL,
    sort_by: Optional[Sequence[str]] = None,
) -> "tuple[pl.LazyFrame, Normalizer]":
    """
    Determine the QC-passed join keys and fit the control z-score -- no feature data copied.

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        The pipeline's cell feature table: ``join_keys``, ``meta_*`` columns and features.
    qc_passed_lf : pl.LazyFrame
        QC_FILTER's ``filtered_cells.parquet``; must contain ``join_keys``. May be
        ``cells_lf`` itself when the QC output carries the features.
    label_column : str
        Name of the variant label column.
    join_keys : sequence of str
        Columns identifying a cell in both frames.
    control : str
        Which rows are controls; see :func:`mark_controls`. Defaults to synonymous.
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
    join_keys = list(join_keys)
    filtered = cells_lf.join(
        qc_passed_lf.select(join_keys), on=join_keys, how="inner", nulls_equal=True
    )
    filtered = mark_controls(filtered, control, label_column)
    normalizer = Normalizer.from_lazyframe(filtered, fit_only_on_control=True)
    filtered_keys = filtered.select(META_SELECTOR)
    if sort_by:
        filtered_keys = filtered_keys.sort(list(sort_by), nulls_last=False)
    return filtered_keys, normalizer


def load_filtered_cells(
    cells_lf: pl.LazyFrame,
    filtered_keys_lf: pl.LazyFrame,
    normalizer: Normalizer,
    join_keys: Sequence[str],
    sort_by: Optional[Sequence[str]] = None,
) -> pl.LazyFrame:
    """
    Reconstruct the QC-passed, control-normalized cell table on demand.

    Joins ``cells_lf`` to ``filtered_keys_lf`` on ``join_keys``, bringing over
    ``CONTROL_COLUMN_NAME`` plus any column only the keys have (columns present on both
    sides come from ``cells_lf``, so the join never ``_right``-suffixes), then applies the
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
    join_keys = list(join_keys)
    cell_columns = set(cells_lf.collect_schema().names())
    key_columns = join_keys + [
        c
        for c in filtered_keys_lf.collect_schema().names()
        if c not in cell_columns and c not in join_keys
    ]
    if CONTROL_COLUMN_NAME not in key_columns:
        key_columns.append(CONTROL_COLUMN_NAME)
    filtered = cells_lf.join(
        filtered_keys_lf.select(key_columns),
        on=join_keys,
        how="inner",
        nulls_equal=True,
    )
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
