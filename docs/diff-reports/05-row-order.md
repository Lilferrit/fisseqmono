# Stable row order: META_CELL_INDEX_COL ported to the embeddings pipeline

**Commit:** the data pipeline's stable-row-order fix (`META_CELL_INDEX_COL`, data pipeline
`aggregatefeaturetype`/QC_FILTER) applied to the embeddings pipeline, plus the other unordered
outputs found while checking it.

**Why.** Polars joins and `group_by` don't preserve a row order under multithreaded execution.
Every seeded step downstream of a join (OvWT's wildtype downsample and fold assignment, the
bootstrap splits, a seeded UMAP) then sees the same cells in a different order on each run,
and gives different results despite a fixed `random_seed`. The data pipeline fixed this by
giving every cell a stable index and sorting on it; the embeddings pipeline never got the fix.

## Code changes

1. **Embeddings pipeline, QC_FILTER and the filter stage:** `filtered_cells.parquet`,
   `filtered_keys.parquet` and every rebuilt cell table are sorted on the cell keys
   `(meta_batch, meta_well, meta_tile, meta_cell_index)`. The embeddings pipeline already had
   a `meta_cell_index` (the cell's index within its tile), so nothing new is assigned:
   `run_qc_filter`'s new `assign_cell_index` parameter (data pipeline: on) is what assigns
   `META_CELL_INDEX_COL` from the input row order; before, sorting on that column implied it.
2. **Both pipelines, QC_FILTER:** `barcode_counts.parquet` and `variants_per_barcode.parquet`
   (each a `group_by`) are sorted on their key (barcode; variant label).
3. **Data pipeline, FINALIZE_FEATURE_SELECT:** the per-variant table is sorted on the label
   before PCA/UMAP and when written (`output.parquet`).

## Output change

Row order only; every value is unchanged. Before regenerating, every scenario matched its old
references under the comparison, which ignores row order (`tests/_compare.py`).

| File | Change |
|---|---|
| `qc_filter/<batch>/{barcode_counts,variants_per_barcode}.parquet` (both pipelines) | rows sorted by barcode / variant label |
| data: `feature_select_batchwise/<batch>/output.parquet` | rows sorted by variant label |
| data: `feature_select_batchwise/<batch>/aggregates/*.parquet` | rows sorted by variant label (the shared aggregation stage has sorted since it was shared; these references were captured before) |
| embeddings: `qc_filter/<batch>/filtered_cells.parquet`, `filter_*/<batch>/filtered_keys.parquet` | byte-identical: on the small fixtures the join already produced key order |

**Determinism check.** Each pipeline scenario (`data`, `embeddings`, `embeddings_two`) run twice
from scratch now gives identical files, row order included (186, 47 and 94 published
parquets). Before the change, the data pipeline's `output.parquet` came out in a different
row order on each run. The embeddings fixtures happened to be identical even before the
change: their tables are too small for Polars to parallelize the joins, so the fix shows on
real data, not here.

References regenerated for `data`, `embeddings` and `embeddings_two`; their `global/`
baselines are unchanged and `test_global_parity` still passes.
