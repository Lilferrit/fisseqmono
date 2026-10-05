"""Pseudo-replicate split files: write one, and filter a cell-level frame by it.

A split half is a parquet of composite cell keys (the filter stage's join keys), applied with a
semi-join. Keys rather than positional row indices, because the stages that write and read a
split each rebuild the cell-level table through a join, whose row order Polars does not
guarantee to be the same in two separate processes.
"""

import logging
import pathlib
from typing import Optional, Union

import polars as pl


def write_split(keys_df: pl.DataFrame, path: Union[str, pathlib.Path]) -> None:
    """
    Write one pseudo-replicate half as a parquet of composite cell keys.

    Parameters
    ----------
    keys_df : pl.DataFrame
        Exactly the join keys
        columns, one row per cell in this half. No feature columns -- a
        split file names cells, it never copies their data.
    path : str or pathlib.Path
        Destination parquet path.
    """
    keys_df.write_parquet(path)


def filter_by_split_file(
    lf: pl.LazyFrame,
    split_file: Optional[Union[str, pathlib.Path]],
    join_keys: list[str],
) -> pl.LazyFrame:
    """
    Restrict a cell-level LazyFrame to the cells named by a split file.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level frame carrying every column in ``join_keys``.
    split_file : str, pathlib.Path, or None
        Path to a :func:`write_split` output. ``None`` is a no-op, so the
        same call site serves AGGREGATE_HALF (a half) and
        AGGREGATE_PASSTHROUGH (every cell).
    join_keys : list of str
        The pipeline's cell identity (the filter stage's join keys). Null key values
        match each other (the data pipeline's untagged cells have a null
        ``meta_variant_tag``).

    Returns
    -------
    pl.LazyFrame
        A semi-join, so no columns are added and no rows are duplicated
        even if the split file somehow repeated a key.
    """
    if split_file is None:
        return lf
    logging.info("Restricting to the cells named by %s", split_file)
    split_lf = pl.scan_parquet(split_file).select(join_keys)
    return lf.join(split_lf, on=join_keys, how="semi", nulls_equal=True)
