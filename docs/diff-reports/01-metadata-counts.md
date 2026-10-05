# Shared `get_aggregate_meta_data`: sorted `*_counts` lists

**Commit:** moving the utilities into `fisseq_common` (phase 3).

**Code difference.** The data pipeline now uses the embeddings pipeline's
`get_aggregate_meta_data` (`fisseq_common.utils.metadata`), which sorts the
`meta_barcode_counts` / `meta_batch_counts` list columns by value. The data pipeline's copy
returned them in `value_counts()` order, which Polars does not define, so it differed between
runs of the same input.

**Output change (data pipeline).**

| File | Column | Change |
|---|---|---|
| `feature_select_batchwise/<batch>/output.parquet` | `meta_barcode_counts` | element order within each list; now sorted by barcode. Same elements and counts. |

No other file or column changed (checked with row order normalized and lists compared
element by element). The embeddings pipeline's outputs are unchanged; it already sorted.
