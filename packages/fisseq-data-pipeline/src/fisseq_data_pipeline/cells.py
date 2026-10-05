"""Loading the QC-passed, normalized cell table in the stages downstream of NORMALIZE.

NORMALIZE publishes only the QC-passed cells' keys (``filtered_keys.parquet``) and the fitted
normalizer (``normalizer.parquet``); see :mod:`fisseq_common.stages.filter`. Every stage that
needs normalized cells takes three files -- QC_FILTER's ``filtered_cells.parquet``
(``cells_file``), NORMALIZE's two outputs -- and rebuilds the table with :func:`load_cells`.

A cell is identified by ``(meta_cell_index, meta_variant_tag)``: QC_FILTER assigns
``meta_cell_index`` from the raw input row order, and a pseudo-variant row shares its source
cell's index under a different tag. The rebuilt table is sorted on those keys, which is
QC_FILTER's own output order, so every seeded step downstream sees the same row order.

The older ``input_file`` (a pre-normalized cell table, or a glob of them) is still accepted
but deprecated.
"""

import dataclasses
import logging
import pathlib
import warnings
from typing import Optional

import polars as pl

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import META_CELL_INDEX_COL, META_VARIANT_TAG_COL
from fisseq_common.stages.filter import load_filtered_cells
from fisseq_common.utils.batches import load_batches

#: How the data pipeline identifies a cell between QC_FILTER and its consumers.
JOIN_KEYS = [META_CELL_INDEX_COL, META_VARIANT_TAG_COL]


@dataclasses.dataclass
class CellsInput:
    """
    Config fields of a stage that reads the normalized cell table.

    Attributes
    ----------
    cells_file : str, optional
        QC_FILTER's ``filtered_cells.parquet``.
    filtered_keys_file : str, optional
        NORMALIZE's ``filtered_keys.parquet``.
    normalizer_file : str, optional
        NORMALIZE's ``normalizer.parquet``.
    input_file : str, optional
        Deprecated: a pre-normalized cell table (or glob of them), ``meta_batch`` taken from
        each file's stem. Set either this or the three files above.
    """

    cells_file: Optional[str] = None
    filtered_keys_file: Optional[str] = None
    normalizer_file: Optional[str] = None
    input_file: Optional[str] = None


def load_cells(cfg: CellsInput) -> "tuple[pl.LazyFrame, str]":
    """
    The normalized cell table described by ``cfg``, and an output filename stem.

    Returns
    -------
    tuple[pl.LazyFrame, str]
        ``(cells, output_stem)``. With the three files, ``output_stem`` is
        ``cells_file``'s stem; with ``input_file``, :func:`load_batches`'.

    Raises
    ------
    ValueError
        Unless exactly one of the two input forms is fully given.
    """
    files = (cfg.cells_file, cfg.filtered_keys_file, cfg.normalizer_file)
    if all(f is not None for f in files) and cfg.input_file is None:
        logging.info("Rebuilding normalized cells from %s, %s and %s", *files)
        cells = load_filtered_cells(
            pl.scan_parquet(cfg.cells_file),
            pl.scan_parquet(cfg.filtered_keys_file),
            Normalizer.load(cfg.normalizer_file),
            JOIN_KEYS,
            sort_by=JOIN_KEYS,
        )
        return cells, pathlib.Path(cfg.cells_file).stem
    if cfg.input_file is not None and all(f is None for f in files):
        warnings.warn(
            "input_file (a pre-normalized cell table) is deprecated; pass cells_file, "
            "filtered_keys_file and normalizer_file instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logging.warning(
            "input_file is deprecated; pass cells_file, filtered_keys_file and "
            "normalizer_file instead"
        )
        logging.info("Loading normalized cells from %s", cfg.input_file)
        return load_batches(cfg.input_file)
    raise ValueError(
        "Set either cells_file, filtered_keys_file and normalizer_file (NORMALIZE's "
        "outputs), or the deprecated input_file -- not both, and not a partial set"
    )
