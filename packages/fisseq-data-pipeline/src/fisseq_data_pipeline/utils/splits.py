"""Restricting a cell-level LazyFrame to one pseudo-replicate half.

GENERATE_SPLIT writes each half as the cells' keys (``meta_cell_index``,
``meta_variant_tag``; :mod:`fisseq_common.utils.splits`). Older split files held positional
row indices in a ``TMP_IDX_COL`` column instead; :func:`filter_by_index_file` still reads
those, but they are deprecated, since they are only right when the reader sees the cells in
exactly the writer's row order.
"""

import logging
import os
from typing import Optional, Union

import polars as pl

from fisseq_common.utils.splits import filter_by_split_file

from ..cells import JOIN_KEYS

TMP_IDX_COL = "tmp_cell_idx"


def add_row_index(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add a ``TMP_IDX_COL`` integer row-index column to a LazyFrame, in scan order.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame.

    Returns
    -------
    pl.LazyFrame
        The input frame with an additional ``TMP_IDX_COL`` integer column.
    """
    return lf.with_columns(pl.row_index(TMP_IDX_COL))


def get_replicate_lf(
    lf: pl.LazyFrame, rep_idx: Union[list[int], pl.Series]
) -> pl.LazyFrame:
    """
    Filter a LazyFrame to rows belonging to one pseudo-replicate.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame that must already contain a ``TMP_IDX_COL``
        integer column (added by :func:`add_row_index`).
    rep_idx : list[int] or pl.Series
        Row indices that belong to this replicate half.

    Returns
    -------
    pl.LazyFrame
        Subset of ``lf`` containing only the rows whose ``TMP_IDX_COL`` value
        is in ``rep_idx``.

    Notes
    -----
    ``rep_idx`` is kept as a ``pl.Series`` rather than being boxed into a
    Python ``set`` first. ``BaseAggregator.aggregate`` re-applies this
    predicate once per feature chunk, and on a real half-split ``rep_idx``
    holds hundreds of thousands of indices -- materializing those as Python
    ints on every chunk is pure overhead.

    ``.implode()`` is required, not cosmetic: passing a Series of the same
    dtype straight to ``is_in`` is ambiguous between element-wise and
    collection membership, and Polars deprecated it (pola-rs/polars#22149).
    Imploding to a single List value states "membership in this one
    collection" explicitly.
    """
    if not isinstance(rep_idx, pl.Series):
        rep_idx = pl.Series(TMP_IDX_COL, rep_idx)
    return lf.filter(pl.col(TMP_IDX_COL).is_in(rep_idx.implode()))


def filter_by_index_file(
    lf: pl.LazyFrame, index_file: Optional[os.PathLike]
) -> pl.LazyFrame:
    """
    Restrict ``lf`` to the cells of a split file, or keep every row for ``None``.

    A split file of cell keys (GENERATE_SPLIT's output) is applied with a semi-join on
    ``(meta_cell_index, meta_variant_tag)``. A deprecated positional file (a single
    ``TMP_IDX_COL`` column) selects rows by their position in ``lf``.
    """
    if index_file is None:
        return lf
    columns = pl.read_parquet_schema(index_file)
    if TMP_IDX_COL not in columns:
        return filter_by_split_file(lf, index_file, JOIN_KEYS)
    logging.warning(
        "%s holds positional row indices (%s); split files of cell keys replace them",
        index_file,
        TMP_IDX_COL,
    )
    lf = add_row_index(lf)
    idx = pl.read_parquet(index_file).get_column(TMP_IDX_COL)
    return get_replicate_lf(lf, idx).drop(TMP_IDX_COL)
