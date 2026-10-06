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

The `stages` extra (scikit-learn, xgboost, hydra, scipy) adds `fisseq_common.stages`: every
stage both pipelines run, whole. Each module holds the algorithm, its Hydra structured config and
its entry point, `python -m fisseq_common.stages.<stage>`; the pipelines have no wrapper modules
of their own. What differs between the pipelines is a config field, set in each pipeline's
`conf/modules.config` ([Shared stages](stages.md) has the details):

| Entry module | Nextflow process(es) | Data pipeline sets | Embeddings pipeline sets |
|---|---|---|---|
| `stages.qcfilter` | QC_FILTER | raw column names (`upBarcode`, `aaChanges`, `editDistance`); `sort_output_by`; `assign_cell_index=true` | `sort_output_by` |
| `stages.filter` | NORMALIZE (+ NORMALIZE_CP_FEATURES) | `batch_name` | `join_keys` |
| `stages.ovwt` (+ `stages.xgbparams`) | OVWT_BATCHWISE (+ OVWT_BATCHWISE_CP_FEATURES) | — | `join_keys`; `feature_selector` |
| `stages.aggregate` | AGGREGATE_FEATURE_TYPE_BATCHWISE, AGGREGATE_FEATURE_TYPE_PASSTHROUGH, AGGREGATE_HALF_BATCHWISE (+ AGGREGATE_FEATURE_TYPE_CP_FEATURES) | `downsample_wt`; `normalize_to_synonymous`; `ext.seed` | the same, plus `join_keys`; `feature_selector` |
| `stages.generatesplit` | GENERATE_SPLIT_BATCHWISE | — | `join_keys` |
| `stages.correlatefeatures` | CORRELATE_FEATURES_BATCHWISE | — | — |
| `stages.blocklist` | BLOCKLIST_BATCHWISE | `minimum_correlation` | `minimum_correlation` |
| `stages.combineblocklists` | COMBINE_BLOCKLISTS_BATCHWISE | — | — |
| `stages.finalize` | FINALIZE_FEATURE_SELECT_BATCHWISE | — | — |

`stages.config` holds the config base classes, `CellsInput` (the three files a stage rebuilds
the normalized cells from, plus `join_keys` and `feature_selector`), `row_keys` and
`stage_main`, which makes each module's entry point. Both pipelines fit the normalizer on the
wildtype cells.

## Nextflow processes

The Nextflow processes for these stages have one copy each, in
`packages/fisseq-common/nextflow/modules/local/<stage>/main.nf` (fisseq-common has modules
only, no workflow). Each module hardcodes its entry point and passes what both pipelines pass
alike; each pipeline's `conf/modules.config` sets only `ext.args`, `ext.seed` and `publishDir`.
See [Shared Nextflow modules](nextflow.md).
