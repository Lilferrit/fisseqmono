# Shared Nextflow modules

The Nextflow process of every [shared stage](stages.md) has one copy, in fisseq-common:

```text
packages/fisseq-common/nextflow/modules/local/
  functions.nf                     threadEnv(), hydraList()
  qc_filter/main.nf                QC_FILTER
  filter/main.nf                   FILTER (each pipeline includes it as NORMALIZE)
  ovwt_batchwise/main.nf           OVWT_BATCHWISE
  aggregate/main.nf                AGGREGATE
  generate_split/main.nf           GENERATE_SPLIT
  correlate_features/main.nf       CORRELATE_FEATURES
  blocklist/main.nf                BLOCKLIST
  combine_blocklists/main.nf       COMBINE_BLOCKLISTS
  finalize_feature_select/main.nf  FINALIZE_FEATURE_SELECT
```

fisseq-common has modules only, no workflow. Each pipeline's workflow includes them by a
relative path, under its own process names:

```groovy
include { QC_FILTER                                       } from '../../fisseq-common/nextflow/modules/local/qc_filter/main'
include { FILTER    as NORMALIZE                          } from '../../fisseq-common/nextflow/modules/local/filter/main'
include { AGGREGATE as AGGREGATE_FEATURE_TYPE_BATCHWISE   } from '../../fisseq-common/nextflow/modules/local/aggregate/main'
include { AGGREGATE as AGGREGATE_FEATURE_TYPE_PASSTHROUGH } from '../../fisseq-common/nextflow/modules/local/aggregate/main'
include { AGGREGATE as AGGREGATE_HALF_BATCHWISE           } from '../../fisseq-common/nextflow/modules/local/aggregate/main'
```

Both pipelines use the same names: `QC_FILTER`, `NORMALIZE`, `OVWT_BATCHWISE`,
`AGGREGATE_FEATURE_TYPE_BATCHWISE`, `AGGREGATE_FEATURE_TYPE_PASSTHROUGH`,
`GENERATE_SPLIT_BATCHWISE`, `AGGREGATE_HALF_BATCHWISE`, `CORRELATE_FEATURES_BATCHWISE`,
`BLOCKLIST_BATCHWISE`, `COMBINE_BLOCKLISTS_BATCHWISE`, `FINALIZE_FEATURE_SELECT_BATCHWISE`. The
embeddings pipeline adds `NORMALIZE_CP_FEATURES`, `AGGREGATE_FEATURE_TYPE_CP_FEATURES` and
`OVWT_BATCHWISE_CP_FEATURES` for its CellProfiler track.

## What a module carries, and what a pipeline sets

A module hardcodes its entry point (`python -m fisseq_common.stages.<stage>`) and passes
everything both pipelines pass alike: the staged input files, `output_dir=.`, the shared params
(`filter_label_column`, the QC thresholds, the `ovwt_*` params,
`aggregate_feature_chunk_size`, `random_seed`) and, for the per-method modules, the method name
as `output_name`. Every module carries `errorStrategy 'ignore'` and pins the threaded libraries
to the task's cpus (`threadEnv`).

Each pipeline's `conf/modules.config` sets, per process name, only:

- `ext.args`: the pipeline-specific config fields. The data pipeline sets QC_FILTER's raw column
  names, `sort_output_by` and `assign_cell_index`, and NORMALIZE's `batch_name`. The embeddings
  pipeline sets QC_FILTER's `sort_output_by`, and `join_keys` and `feature_selector` wherever the
  normalized cells are read. Both set the aggregate processes' `downsample_wt` and
  `normalize_to_synonymous`, and BLOCKLIST's `minimum_correlation`. See each stage's "What each
  pipeline sets" in [Shared stages](stages.md).
- `ext.seed`: the seed of `AGGREGATE_HALF_BATCHWISE`,
  `random_seed + rep * 2 + half`. Other processes use `params.random_seed`.
- `publishDir`: where the outputs go. The paths must match `fisseq_common.layout`; each
  pipeline's `tests/unit/test_publish_layout.py` checks.

The parameters these read have the same names and defaults in both pipelines' `params.yaml`.
