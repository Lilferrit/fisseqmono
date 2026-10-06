# No pycytominer or PCA/UMAP in FINALIZE_FEATURE_SELECT

**Commit:** the data pipeline's final feature-selection stage drops its pycytominer filters
(variance threshold, pycytominer blocklist, correlation threshold) and its optional PCA/UMAP.

**Why.** Per-experiment feature selection is the bootstrap reproducibility blocklist alone, as in
the embeddings pipeline. pycytominer's filters were written for named CellProfiler features, and
dimensionality reduction belongs to the cross-experiment step (`fisseqborn-global`, which still
runs its full-rank PCA).

## Code changes

1. `fisseq_data_pipeline.featureselect`: no `pycytominer_operations`, `run_pca`,
   `pca_n_components`, `run_umap` or `umap_*` fields. The steps are: join the per-type
   aggregates, apply the combined blocklist, z-score to the synonymous variants, impact score,
   per-variant metadata, passthrough aggregates.
2. `fisseq_common.stages.pycytominer` and `fisseq_common.stages.dimreduction` are deleted, and
   `pycytominer` leaves fisseq-common's `stages` extra. The data pipeline no longer depends on
   pycytominer, umap-learn, numba, llvmlite or pandas.
3. `params.yaml` loses `run_pca`, `pca_n_components`, `run_umap` and `umap_*`. A run that still
   sets one warns and ignores it. FINALIZE no longer has a `pca_components.parquet` output.

## Output change

Only `feature_select_batchwise/<batch>/output.parquet` changes. Every other published file is
identical.

| File | Change |
|---|---|
| `batch1/output.parquet` | `Cells_AreaShape_Perimeter_mean` and `Nuclei_AreaShape_Area_mean` are now kept (pycytominer dropped them); `meta_impact_score` moves by at most 0.023 |
| `batch2/output.parquet` | `Cells_AreaShape_Area_median` is now kept; `meta_impact_score` moves by at most 0.097 |

Every column present before has the same values, except `meta_impact_score`. That score is a
cosine distance over all kept features, so it changes when the feature set does.
