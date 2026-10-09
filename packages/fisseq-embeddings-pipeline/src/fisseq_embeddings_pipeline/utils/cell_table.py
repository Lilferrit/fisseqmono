"""The per-cell ``meta_*`` columns BUILD_CELL_IMAGES writes, and their projections.

BUILD_CELL_IMAGES writes the same per-cell metadata twice, from the same
starcall CSVs (``build_cell_images_table.tile_cell_meta``): as the leading
columns of ``cell_table.parquet``, and as each WebDataset sample's
``meta.json``. Those columns are :data:`CELL_META_SCHEMA`: the cell's key
within its experiment (well, tile, cell index), the three genotype fields
QC_FILTER thresholds on, and the variant class. Neither carries
``meta_batch``: the shards are cached in starcall's tree, and the batch
name is a run-level one.

BUILD_CELL_METADATA (the QC_FILTER input), BUILD_CP_FEATURES and
EMBED_CELLS each add ``meta_batch`` and keep :data:`CELL_METADATA_SCHEMA`'s
seven columns -- ``filter.py``'s ``JOIN_KEYS`` plus the three QC fields.
The variant class stays out of them: every downstream stage derives it
from ``meta_aa_changes`` itself.

Not to be confused with ``utils/metadata.py``, which is the vendored
*per-variant* aggregation helper (``get_aggregate_meta_data``).
"""

import polars as pl

from fisseq_common.schema import (
    META_BARCODE_COL,
    META_BATCH_COL,
    META_EDIT_DISTANCE_COL,
    META_VARIANT_CLASS,
)

META_WELL_COL: str = "meta_well"
META_TILE_COL: str = "meta_tile"
META_CELL_INDEX_COL: str = "meta_cell_index"
META_AA_CHANGES_COL: str = "meta_aa_changes"

#: What ``cell_table.parquet`` leads with and every shard sample's
#: ``meta.json`` holds, in this order.
CELL_META_SCHEMA: dict[str, pl.DataType] = {
    META_WELL_COL: pl.String,
    META_TILE_COL: pl.String,
    META_CELL_INDEX_COL: pl.Int64,
    META_BARCODE_COL: pl.String,
    META_AA_CHANGES_COL: pl.String,
    META_EDIT_DISTANCE_COL: pl.Int64,
    META_VARIANT_CLASS: pl.String,
}

#: The per-batch projection BUILD_CELL_METADATA, BUILD_CP_FEATURES and
#: EMBED_CELLS write: ``meta_batch`` plus :data:`CELL_META_SCHEMA` without
#: the variant class. Also the schema of an empty frame.
CELL_METADATA_SCHEMA: dict[str, pl.DataType] = {
    META_BATCH_COL: pl.String,
    **{k: v for k, v in CELL_META_SCHEMA.items() if k != META_VARIANT_CLASS},
}


def cell_metadata_exprs(batch_stem: str) -> list[pl.Expr]:
    """
    The :data:`CELL_METADATA_SCHEMA` columns out of ``cell_table.parquet``
    (or a frame of ``meta.json`` rows), in order.

    Parameters
    ----------
    batch_stem : str
        This experiment's identifier, written into every row as
        ``meta_batch`` -- a literal, since neither source carries it.

    Returns
    -------
    list[pl.Expr]
        Seven expressions for a ``select()``, optionally followed by
        further ones (BUILD_CP_FEATURES appends its CellProfiler columns).
    """
    return [pl.lit(batch_stem).alias(META_BATCH_COL)] + [
        pl.col(name).cast(dtype)
        for name, dtype in CELL_METADATA_SCHEMA.items()
        if name != META_BATCH_COL
    ]
