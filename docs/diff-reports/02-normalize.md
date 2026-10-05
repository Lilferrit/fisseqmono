# NORMALIZE switches to the shared filter stage

**Commit:** sharing the normalization stage (`fisseq_common.stages.filter`).

**Code difference.** The data pipeline's NORMALIZE wrote a full normalized copy of every cell
plus the fitted normalizer. It now uses the embeddings pipeline's design (decision D3): it
publishes only the QC-passed cells' keys and the normalizer, and every consumer (OVWT_BATCHWISE,
AGGREGATE_FEATURE_TYPE, GENERATE_SPLIT, AGGREGATE_HALF, FINALIZE_FEATURE_SELECT) rebuilds the
normalized table from QC_FILTER's `filtered_cells.parquet` and those two files, joining on
`(meta_cell_index, meta_variant_tag)` and sorting on the same keys. The control rows stay
wildtype (`control_sample_query`, decision D1).

**Output change (data pipeline).**

| Before | After |
|---|---|
| `normalization/cells/<batch>.parquet` (all columns, normalized) | removed |
| `normalization/normalizers/<batch>.normalizer.parquet` | `normalization/<batch>/normalizer.parquet`, identical contents |
| — | `normalization/<batch>/filtered_keys.parquet`: the old normalized table's `meta_*` columns (same rows, same order) plus `meta_batch` |

Every other output file of the data pipeline's reference scenario is unchanged (OvWT results
and cell scores, all aggregates, splits, half aggregates, correlations, blocklists,
`output.parquet`): the rebuilt table has the same rows, values and row order as the old
normalized file, so every seeded step downstream draws the same samples.

The stage CLIs keep accepting the old `input_file` (a pre-normalized table), with a deprecation
warning; `save_normalizer` is accepted and ignored.
