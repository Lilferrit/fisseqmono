# Embeddings pipeline on the data pipeline's stages

**Commit:** every stage both pipelines run moves whole into fisseq-common (entry point
`python -m fisseq_common.stages.<stage>`, Nextflow module under
`packages/fisseq-common/nextflow/`). Downstream of EMBED_CELLS, the embeddings pipeline now runs
the data pipeline's stages with the data pipeline's settings.

**Why.** Except for what is specific to cellDINO (cell images, sharding, embedding), the
pipelines should pool and normalize variants the same way. That covers the same QC, the same
controls, the same per-type aggregation and bootstrap feature selection, the same final table,
and the same parameters and publish layout.

## Code changes

1. **Controls.** The embeddings pipeline's normalizer is fit on the wildtype cells, as in the data
   pipeline. Before, it was fit on untagged synonymous variants. The synonymous variants are now
   ordinary aggregate rows. WT is the reference of KS/AUROC/QQ and is left out of the aggregates.
2. **Aggregation.** The single multi-method `aggregate.parquet` (AGGREGATE_EMBEDDINGS) is
   replaced by one file per method, `aggregates/<method>.parquet` (AGGREGATE_FEATURE_TYPE).
   Columns are always `<feature>_<method>`, with no bare median columns, and are z-scored
   against the experiment's synonymous variants. `feature_select_downsample_wt` and the per-half
   seed `random_seed + rep*2 + half` apply as in the data pipeline.
3. **Final table.** FILTER_AGGREGATE's `filtered_aggregate.parquet` and
   `aggregate_with_passthrough.parquet` are replaced by FINALIZE_FEATURE_SELECT's
   `output.parquet`. It has the blocklist applied, a z-score to the synonymous variants,
   `meta_is_control`, `meta_impact_score`, per-variant metadata counts and the passthrough
   aggregates.
4. **Parameters.** The embeddings pipeline now uses the data pipeline's names and defaults:
   - `feature_select_types` (default `mean, median, MAD, std, KS, QQ, AUROC`; was
     `aggregate_methods`, `median, KS, AUROC`)
   - `feature_select_passthrough_types`
   - `feature_select_bootstrap_reps`, `feature_select_min_correlation`,
     `feature_select_downsample_wt`
   - `feature_select_types_cp_features`
   - the `run_ovwt` / `run_feature_selection` gates
   - the `qc_*` variant and pseudo-variant downsampling parameters, which are off by default

   A run that still sets an old name warns and ignores it.
5. **Layout.** The embeddings pipeline publishes the same per-experiment layout as the data
   pipeline:
   - `normalization/` (was `filter_embeddings/`)
   - `splits/bootstrap_N/`, `half_aggregates/bootstrap_N/<method>/halfK_agg.parquet` and
     `correlations/<method>/bootstrap_N.parquet` (were the `repN/` paths)
   - `output.parquet`

   The CellProfiler track uses `normalization_cp_features/` and `aggregates/median.parquet`, and
   stays aggregation plus OvWT only.
6. **Both pipelines: QC pseudo-variants.** A pseudo-variant row now carries its own
   `meta_variant_tag` (`downsample-<amount>`), and every stage sorts and splits on the cell keys
   plus that tag. Before, a pseudo row had the same `(meta_cell_index, meta_variant_tag)` as its
   source cell, so the data pipeline's NORMALIZE join duplicated both rows whenever
   `qc_downsample_amounts` was set. The default is unset, so no reference changes from this.
7. **Both pipelines: metadata source.** The stages take `meta_*` columns from QC_FILTER's side and
   the features from the cell table. FINALIZE computes its metadata counts from
   `filtered_keys.parquet`.
8. **Retired:** `tests/test_global_parity.py` and the `embeddings_global` scenario, together with
   the `global/` references of the embeddings scenarios. They compared fisseqborn with the
   deleted global stages' outputs, which were computed from per-experiment outputs the pipeline
   no longer writes. `tests/test_layout_load.py` now runs `fisseqborn.write_global` on the new
   embeddings reference run instead.

## Output change

**Data pipeline:** none. Every reference file matched before and after this commit.

**Embeddings pipeline, `embeddings_two` scenario** (its fixture is unchanged):

| Output | Change |
|---|---|
| `cell_images/`, `cell_metadata/`, `embeddings/`, `cp_features/`, `qc_filter/` | identical |
| `ovwt_batchwise{,_cp_features}/<batch>/results.parquet` | identical (OvWT always scores against WT) |
| `ovwt_batchwise{,_cp_features}/<batch>/cell_scores.parquet` | changed: the features are normalized to WT, not to the synonymous variants |
| `filter_embeddings/` → `normalization/` | `meta_is_control` marks WT (6 cells per batch), not A1A/A2A |
| `aggregate.parquet` (2 rows: M1K, WT; 1158 columns) → `aggregates/<method>.parquet` × 7 | 3 rows (A1A, A2A, M1K), 384 features each, z-scored to A1A/A2A |
| `blocklist.parquet` | 2688 features (7 methods) instead of 1152 (3); 828 reproducible in batch1 (was 413) |
| `filtered_aggregate.parquet` / `aggregate_with_passthrough.parquet` → `output.parquet` | 3 × 1220 |
| `passthrough_aggregates/KSnegLogP.parquet` | changed: computed against WT over A1A, A2A, M1K |
| CP track `aggregate.parquet` → `aggregates/median.parquet` | `_median`-suffixed, z-scored to the synonymous variants |

**Embeddings pipeline, `embeddings` scenario.** The fixture now has two synonymous variants
(2 barcodes × 3 cells each of WT, A1A, A2A and M1K; it was 4 WT barcodes, A1A and M1K), because
the z-score to the synonymous variants needs at least two. Every file changed, upstream ones
included, because the synthetic input changed.

On these fixtures `meta_impact_score` is 0.5 for every variant. With exactly two synonymous
variants z-scored to ±x, their median is the zero vector, and the cosine distance from it is
undefined.
