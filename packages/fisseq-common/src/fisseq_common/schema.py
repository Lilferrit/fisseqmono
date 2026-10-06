"""Column names and Polars selectors shared by the pipelines and fisseqborn.

Every metadata column starts with ``meta_``; every other column of a cell- or variant-level
table is a feature. ``FEATURE_SELECTOR`` selects those features by exclusion, so it covers
both CellProfiler feature names and embedding dimensions. ``EMBEDDING_SELECTOR`` selects only
the embeddings pipeline's zero-padded ``emb_NNNN`` dimensions.
"""

from typing import Any

import numpy as np
import polars as pl
from polars import selectors as cs

FEATURE_SELECTOR: pl.Expr = cs.exclude("^meta_.*$")
META_SELECTOR: pl.Expr = cs.matches("^meta_.*$")
EMBEDDING_SELECTOR: pl.Expr = cs.matches(r"^emb_\d+$")
# Below this a standard deviation counts as zero (the feature is constant).
EPS: np.floating[Any] = np.finfo(np.float32).eps
CONTROL_COLUMN_NAME: str = "meta_is_control"
CONTROL_COLUMN: pl.Expr = pl.col(CONTROL_COLUMN_NAME)
META_BARCODE_COL: str = "meta_barcode"
META_BATCH_COL: str = "meta_batch"
META_EDIT_DISTANCE_COL: str = "meta_edit_distance"
# Stable per-cell identity. The data pipeline's QC_FILTER assigns it from the raw input row
# order (qcfilter.combine_cell_files) and sorts its output on it. Polars inner joins are not
# order-preserving under multithreading, so without an explicit key the published cell tables
# come out in a different row order on every run, which silently breaks every downstream
# seeded step (OvWT's wildtype downsample and fold assignment, the feature-selection bootstrap
# splits) even at a fixed random_seed.
#
# The embeddings pipeline has a column of the same name with a different meaning: the cell's
# index within its tile (fisseq_embeddings_pipeline.utils.cell_table), unique only together
# with meta_batch, meta_well and meta_tile. Its QC_FILTER and filter stage sort on those four
# columns for the same reason.
META_CELL_INDEX_COL: str = "meta_cell_index"
META_VARIANT_TAG_COL: str = "meta_variant_tag"
META_VARIANT_CLASS: str = "meta_variant_class"
IMPACT_SCORE_COL: str = "meta_impact_score"
PC_COL_PREFIX: str = "meta_pc_"
UMAP_COL_PREFIX: str = "meta_umap_"
COMPONENT_IDX_COL: str = "meta_component_idx"
VARIANCE_EXPLAINED_COL: str = "meta_variance_explained"
CUMULATIVE_VARIANCE_EXPLAINED_COL: str = "meta_cumulative_variance_explained"

#: Every aggregation method's name (the keys of fisseq_common.stages.aggregate's registry). An
#: aggregate column is ``<feature>_<method>``, except a median-only embeddings run's bare
#: ``<feature>`` columns.
AGGREGATOR_NAMES: tuple[str, ...] = (
    "mean",
    "median",
    "MAD",
    "std",
    "KS",
    "signedKS",
    "QQ",
    "AUROC",
    "KSnegLogP",
    "AUROCnegLogP",
)
