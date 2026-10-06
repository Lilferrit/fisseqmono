# Feature Selection

The BATCHWISE bootstrap pseudo-replicate feature-selection pipeline (run once
per batch) is implemented as five Hydra entry points, one per module, each a
Nextflow process (see [Nextflow Workflow](../nextflow.md)). Cells are split
into stratified 50/50 pseudo-replicate halves across
`params.feature_select_bootstrap_reps` replicates; each half is
aggregated per feature type (via [`python -m fisseq_data_pipeline.aggregatefeaturetype`](aggregate.md)),
correlated against its partner half, and a per-feature blocklist is derived from
the median correlation across all bootstrap replicates. The final stage joins the
per-feature-type aggregates, applies the blocklist, and z-scores the result
against the synonymous variants. The blocklist is the only feature selection.

Everything here is per batch. Combining batches across experiments is done
downstream by [fisseqborn](https://github.com/FowlerLab/fisseqborn) from the
published outputs (see
[Architecture](../architecture.md#cross-experiment-aggregation)).

The full per-feature-type aggregates for `params.feature_select_types` are
published to `feature_select_batchwise/<batch>/aggregates/` z-scored against
the batch's synonymous variants (`normalize_to_synonymous=true`; see
[Aggregate: Synonymous normalization](aggregate.md#synonymous-normalization)).
The bootstrap half-aggregates stay raw.

All configs extend the [common config fields](qcfilter.md#common-config-fields).

## Two lists of aggregate types

`params.feature_select_types` names the aggregators that *decide* which
features survive: each one is bootstrapped, correlated across pseudo-replicate
halves, and blocklisted on median `r`.

`params.feature_select_passthrough_types` names aggregators that are computed
and joined onto the final per-variant table but take no part in any of that.
They skip stages 1b→4 of the bootstrap chain entirely (no splits, no
correlation, no blocklist), and are joined *after* the synonymous-baseline
normalization — so they are also outside the impact score. They are published separately and raw
(not z-scored), under `feature_select_batchwise/<batch>/passthrough_aggregates/`
rather than `aggregates/`, so a glob over `aggregates/` never mixes normalized
and raw-scale values.

This exists for the p-value aggregators (`KSnegLogP`, `AUROCnegLogP`). They are
wanted in the output, but they are not reproducibility statistics: a median-`r`
threshold means nothing for a p-value, and they must stay on their own scale.
Skipping the bootstrap also saves `2 × bootstrap_reps`
aggregation tasks per type.

The two lists must be disjoint; `workflows/fisseq.nf` rejects an overlap before
any task is submitted.

!!! note
    The consequence is that `output.parquet` can carry non-`meta_` columns that
    were never blocklisted or normalized. Selecting feature
    columns from that file by the usual `^meta_` convention no longer yields
    "the selected features".

## 1. `python -m fisseq_data_pipeline.generatesplit` (`GENERATE_SPLIT`)

Generates one stratified 50/50 pseudo-replicate split.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `filtered_keys_file` | **required** | NORMALIZE's `filtered_keys.parquet`. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `bootstrap_idx` | `0` | Bootstrap replicate number. The split is seeded with `random_seed + bootstrap_idx`; Nextflow passes `params.random_seed` and the replicate number, so each replicate is distinct and reproducible. |
| `input_file` | `null` | Deprecated: a normalized cell table, instead of `filtered_keys_file`; writes the old positional split files. |

**Output**: `half1.parquet`, `half2.parquet`: each half's cells as `(meta_cell_index, meta_variant_tag)` keys. The stage is shared with the embeddings pipeline (`fisseq_common.stages.generatesplit`).

```bash
uv run python -m fisseq_data_pipeline.generatesplit \
    output_dir=./out \
    filtered_keys_file=out/normalization/batch1/filtered_keys.parquet \
    bootstrap_idx=3 \
    random_seed=0
```

## 2. `python -m fisseq_data_pipeline.correlatefeatures` (`CORRELATE_FEATURES`)

Computes per-feature Pearson correlation between two aggregate halves for the same
feature type.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `half1_file` | **required** | First half's per-feature-type aggregate parquet. |
| `half2_file` | **required** | Second half's per-feature-type aggregate parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `output_name` | `"correlations"` | Basename of the output file. |

**Output**: `correlations.parquet` (columns: `feature`, `r`, `r_squared`). An undefined
correlation (a constant column) is stored as null, so BLOCKLIST's median skips it instead of
propagating NaN.

```bash
uv run python -m fisseq_data_pipeline.correlatefeatures \
    output_dir=./out \
    half1_file=out/half1.mean.parquet \
    half2_file=out/half2.mean.parquet
```

## 3. `python -m fisseq_data_pipeline.blocklist` (`BLOCKLIST`)

The one intentional cross-bootstrap synchronization point: gathers every bootstrap
replicate's correlation table for one feature type and computes each feature's
median `r` across replicates.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `correlation_files` | **required** | Glob pattern matching all bootstrap-replicate correlation parquet files for one feature type. |
| `minimum_correlation` | `0.5` | Minimum median Pearson `r` required for a feature to pass. |
| `output_name` | `"blocklist"` | Basename of the output file. |

**Output**: `blocklist.parquet` (columns: `feature`, `median_r`, `feature_ok`), sorted by
`feature`. A feature whose correlation was undefined in every replicate has a null `median_r`
and `feature_ok = false`: it is blocked.

CORRELATE_FEATURES, BLOCKLIST and COMBINE_BLOCKLISTS are shared with the embeddings pipeline
(`fisseq_common.stages`).

```bash
uv run python -m fisseq_data_pipeline.blocklist \
    output_dir=./out \
    'correlation_files=out/correlations/mean/*.parquet' \
    minimum_correlation=0.5
```

## 4. `python -m fisseq_data_pipeline.combineblocklists` (`COMBINE_BLOCKLISTS`)

Concatenates every feature type's blocklist into one combined blocklist (a plain
concat is correct — stat-suffixed feature names never collide across feature
types).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `blocklist_files` | **required** | Glob pattern matching all per-feature-type blocklist parquet files. |

**Output**: `blocklist.parquet`.

```bash
uv run python -m fisseq_data_pipeline.combineblocklists \
    output_dir=./out \
    'blocklist_files=out/blocklists/*.parquet'
```

## 5. `python -m fisseq_data_pipeline.featureselect` (`FINALIZE_FEATURE_SELECT`)

The final stage: joins every feature type's full aggregate (from
[`python -m fisseq_data_pipeline.aggregatefeaturetype`](aggregate.md)) on `label_column`, drops blocked
feature columns. The remaining table is then z-scored
against the synonymous variants (a `Normalizer` fit on
`variant_classification()`'s synonymous rows). In the pipeline its inputs are
already synonymous-z-scored by `AGGREGATE_FEATURE_TYPE`, so this second pass is
effectively a no-op; it keeps the stage correct when run standalone on raw
aggregates. Like the upstream normalization, it needs at least two synonymous
variants.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cells_file` | **required** | QC_FILTER's `filtered_cells.parquet` — used only to derive per-variant metadata. |
| `filtered_keys_file` | **required** | NORMALIZE's `filtered_keys.parquet`. |
| `normalizer_file` | **required** | NORMALIZE's `normalizer.parquet`. |
| `input_file` | `null` | Deprecated: a pre-normalized cell table (or glob), instead of the three files above. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `feature_type_files` | **required** | Glob pattern matching per-feature-type full aggregate parquet files. |
| `block_list_file` | **required** | Combined blocklist parquet, with `feature` and `feature_ok` columns. Features with `feature_ok` false or null are dropped; features the blocklist doesn't mention are kept. |
| `compute_impact_score` | `true` | Compute per-variant impact score (cosine distance vs. synonymous baseline) after feature selection. |
| `passthrough_feature_type_files` | `null` | Glob matching per-feature-type aggregates to join onto the output *without* feature selection or normalization (see [Two lists of aggregate types](#two-lists-of-aggregate-types)). Unlike `feature_type_files`, a glob matching nothing warns rather than raising — an empty passthrough list is the default. |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.parquet` or `{output_dir}/{stem}.parquet`.

```bash
uv run python -m fisseq_data_pipeline.featureselect \
    output_dir=./out \
    cells_file=out/qc_filter/batch1/filtered_cells.parquet \
    filtered_keys_file=out/normalization/batch1/filtered_keys.parquet \
    normalizer_file=out/normalization/batch1/normalizer.parquet \
    'feature_type_files=out/aggregates/*.parquet' \
    block_list_file=out/blocklist.parquet
```

See [API Reference: features](../api/features.md) for full function
documentation, including `compute_feature_correlations`.
