# fisseq-common

Code shared by `fisseq-data-pipeline`, `fisseq-embeddings-pipeline` and `fisseqborn`.

```bash
pip install "fisseq-common @ git+https://github.com/Lilferrit/fisseqmono@v2.0.0#subdirectory=packages/fisseq-common"
pip install "fisseq-common[stages] @ git+https://github.com/Lilferrit/fisseqmono@v2.0.0#subdirectory=packages/fisseq-common"
```

The base install needs only polars, pyarrow and numpy:

| Module | Contents |
|---|---|
| `fisseq_common.schema` | `meta_*` column names, `FEATURE_SELECTOR` / `EMBEDDING_SELECTOR` / `META_SELECTOR`, `META_CELL_INDEX_COL`, `AGGREGATOR_NAMES` |
| `fisseq_common.variant` | `classify_variant` and its polars version, `variant_type_expr` / `control_expr` |
| `fisseq_common.normalizer` | `Normalizer`: control-fitted z-scoring |
| `fisseq_common.layout` | where each pipeline publishes its per-experiment outputs (`DataPipelineLayout`, `EmbeddingsPipelineLayout`, `detect`) |
| `fisseq_common.global_aggregation` | cross-experiment aggregation: the blocklist vote, median pooling, OvWT z-score-then-median, the reduced PCA view |
| `fisseq_common.utils` | `batches`, `log`, `metadata`, `splits`, `vectors` |

The `stages` extra (scikit-learn, xgboost, hydra, scipy, pycytominer) adds
`fisseq_common.stages`, the per-experiment stages both pipelines run. Each pipeline's
`python -m` module is a thin Hydra entry point over one of them:

| Stage | Shared module | Parameters each pipeline sets |
|---|---|---|
| QC_FILTER | `stages.qcfilter` | input column names; pseudo-variant downsampling (data: on) |
| NORMALIZE / FILTER_EMBEDDINGS / FILTER_CP_FEATURES | `stages.filter` | `control` (data: WT query; embeddings: synonymous); `join_keys`; `sort_by` |
| OVWT_BATCHWISE | `stages.ovwt`, `stages.xgbparams` | `feature_selector` |
| GENERATE_SPLIT | `stages.generatesplit` | `join_keys` |
| AGGREGATE_FEATURE_TYPE / AGGREGATE_HALF / AGGREGATE_EMBEDDINGS / ... | `stages.aggregate` | `feature_selector`; `downsample_controls`; `normalize_to_synonymous`; `bare_median` |
| CORRELATE_FEATURES, BLOCKLIST, COMBINE_BLOCKLISTS | `stages.correlatefeatures`, `stages.blocklist`, `stages.combineblocklists` | — |
| FILTER_AGGREGATE / FINALIZE_FEATURE_SELECT | `stages.filter_aggregate`, `stages.pycytominer` | `pycytominer_operations` (data: variance, blocklist, correlation; embeddings: none) |
| PCA / UMAP | `stages.dimreduction` | `random_state` |

The Nextflow processes for these stages also have one copy each, in the repository's
`nextflow/modules/local/<stage>/main.nf`; each pipeline's `conf/modules.config` sets its entry
point, arguments and publish paths.
