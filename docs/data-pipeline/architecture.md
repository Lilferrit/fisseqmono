# Architecture

The FISSEQ Data Pipeline is a Nextflow + Python workflow for processing
single-cell CellProfiler morphological profiling data from FISSEQ
(Fluorescence In-Situ Sequencing) experiments. Each cell carries a genetic
variant label; the pipeline measures how each variant's cell population differs
from wildtype (WT) controls using morphological features.

## Pipeline DAG

```
params.yaml (experiments: [...])
      │
      ▼
   INPUT            (per experiment)  ──►  input/<batch_stem>.parquet
      │
      ▼
   QC_FILTER        (per experiment)   ← edit distance, barcode count,
      │                                   variant barcode count
      ▼
   NORMALIZE        (per experiment)   ← z-score fit on WT control cells
      │
      ├──► OVWT_BATCHWISE  (per experiment; gated by run_ovwt)
      │            k-fold CV one-vs-wildtype scoring
      │
      └──► Feature selection, batchwise (gated by run_feature_selection):
             AGGREGATE_FEATURE_TYPE_BATCHWISE   (per feature type;     ─┐
                                                 synonymous z-score)    │
             AGGREGATE_FEATURE_TYPE_PASSTHROUGH (per passthrough type; │
                                                 raw)                   │
             GENERATE_SPLIT_BATCHWISE           (per bootstrap)         │
               └─► AGGREGATE_HALF_BATCHWISE     (per bootstrap × type × │
                                                 half)                  │
                     └─► CORRELATE_FEATURES_BATCHWISE (per bootstrap ×  │
                                                       type)            │
                           └─► BLOCKLIST_BATCHWISE (gathers all         │
                                                    bootstraps — the    │
                                                    one sync point)     │
                                 └─► COMBINE_BLOCKLISTS_BATCHWISE ─────┘
                                       └─► FINALIZE_FEATURE_SELECT_BATCHWISE
```

There is a single pipeline mode. `main.nf` includes one workflow,
`workflows/fisseq.nf`, and runs it. Every stage runs per experiment; nothing in
the pipeline combines experiments (see
[Cross-experiment aggregation](#cross-experiment-aggregation)).

Only INPUT is this pipeline's own. Every stage after it is a
[shared stage](../common/stages.md), `python -m fisseq_common.stages.<stage>`, run from the
[shared Nextflow modules](../common/nextflow.md) in
`packages/fisseq-common/nextflow/modules/local/`. The embeddings pipeline runs the same
stages, with the same process names, parameters and publish layout, downstream of its own
cell tables. What this pipeline sets for them (`conf/modules.config`) is listed under each
stage's "What each pipeline sets".

## Experiments

Every experiment is declared as a map in the `experiments:` list in
`params.yaml`. There is no `<pipeline_dir>/configs/` directory, and no
per-experiment override of arbitrary pipeline parameters — an entry may set only
`batch_stem`, `input_paths`, and the three INPUT-stage fields.
Anything else is rejected with an error naming the key. See
[Configuration](configuration.md).

## Cross-experiment aggregation

The pipeline stops at per-experiment outputs. Combining experiments — merging
blocklists across experiments, z-scoring each experiment's per-variant profiles
and taking the per-variant median across experiments, and re-centering OvWT
AUROCs against synonymous variants before a cross-experiment median — is the job
of the downstream [fisseqborn](../fisseqborn/index.md) package (`fisseqborn-global`).
It reads each experiment's published
`feature_select_batchwise/<batch_stem>/{aggregates,passthrough_aggregates,blocklists}/<feature_type>.parquet`
and `ovwt_batchwise/<batch_stem>/results.parquet`.

## Reproducibility

`params.random_seed` is the only seed in the pipeline. Every stochastic step
reads it: QC pseudo-variant downsampling, the feature-selection bootstrap splits
and their wildtype subsampling, OvWT's fold shuffle / inner calibration split /
XGBoost `seed`.

Stages that must differ from one another derive a fixed offset rather than
owning a seed — `GENERATE_SPLIT_BATCHWISE` uses `random_seed + bootstrap_idx`, and
`AGGREGATE_HALF_BATCHWISE` uses `random_seed + bootstrap_idx * 2 + half_num`, so bootstrap
replicates still draw independent subsamples. `tests/unit/test_config.py` sweeps
every config class to keep a second seed from reappearing.

Reproducibility also depends on **row order** being stable, which is easy to
lose: polars inner joins are not order-preserving under multithreaded execution.
`QC_FILTER` therefore assigns `meta_cell_index` over the raw input order and
sorts on `(meta_cell_index, meta_variant_tag)` before publishing; every later stage sorts its
rebuilt cell table the same way (see
[Cell identity](../common/stages.md#cell-identity-and-the-normalized-cell-table)). Without that, the same input yields the same rows
in a different order on every run, and every seeded step downstream silently
diverges despite a fixed seed.

## Two different control baselines

These are easy to confuse:

| Stage | Control population |
| ----- | ------------------ |
| `NORMALIZE` (cell level) | **Wildtype** cells (`meta_aa_changes = 'WT'`) |
| `AGGREGATE_FEATURE_TYPE_BATCHWISE` (aggregate level, `normalize_to_synonymous`) | **Synonymous** variants |
| `FINALIZE_FEATURE_SELECT_BATCHWISE` (aggregate level) | **Synonymous** variants |

`fisseq_common.stages.filter.variant_classification` flags synonymous, untagged labels as
`meta_is_control = True`; NORMALIZE uses the WT SQL predicate (its `control` field) instead.
Both synonymous-baseline normalizations fit a `ddof=1` std, so each experiment needs
at least two synonymous variants — with one, every feature comes out null.

`OVWT_BATCHWISE` consumes the wildtype-normalized cells. The synonymous re-centering happens
at the score level, on the AUROCs, downstream in fisseqborn. The embeddings pipeline uses the
same controls.

## Variant tags

`aaChanges` values may carry a `:<tag>` suffix (e.g. `V123A:downsampled-half`).
The tag is stripped exactly once, in `fisseq_common.stages.qcfilter.filter_columns`:
`meta_aa_changes` is always the tag-stripped base label and `meta_variant_tag`
holds the tag (`null` when absent). Every stage after `QC_FILTER` therefore sees
clean, pooled variant labels. Do not re-strip or re-parse tags downstream — use
`meta_variant_tag` directly.

Two independent things can put a tag there: upstream raw data may arrive
pre-tagged, or QC_FILTER's optional pseudo-variant downsampling
(`qc_downsample_amounts`) may generate one (`downsample-{amount}`) for the rows
it creates. A pseudo-variant row is a copy of its source cell with the same
`meta_cell_index` but its own `meta_variant_tag`, so a cell is identified by
`(meta_cell_index, meta_variant_tag)`: that is what every stage joins, sorts and splits on.

## Output layout

```
<pipeline_dir>/
  input/<batch_stem>.parquet
  qc_filter/<batch_stem>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/<batch_stem>/
    filtered_keys.parquet    # QC-passed cells' meta_* columns + meta_is_control
    normalizer.parquet       # wildtype-fitted per-feature mean/std
  ovwt_batchwise/<batch_stem>/
    results.parquet          # per variant: auroc_pooled, auroc_median_barcode,
                             # auroc_folds, auroc_median_fold,
                             # meta_n_barcodes, meta_n_cells
    cell_scores.parquet      # per cell per variant scored against: meta_* +
                             # score + meta_variant_scored_against
    models.pkl               # dict[variant -> list[(Booster, calibrator|None)]]
  feature_select_batchwise/<batch_stem>/
    aggregates/<feature_type>.parquet              # z-scored to synonymous variants
    passthrough_aggregates/<feature_type>.parquet  # raw scale (passthrough types)
    splits/bootstrap_<n>/half{1,2}.parquet
    half_aggregates/bootstrap_<n>/<feature_type>/half{1,2}_agg.parquet
    correlations/<feature_type>/bootstrap_<n>.parquet
    blocklists/<feature_type>.parquet
    blocklist.parquet
    output.parquet
```

The layout is `fisseq_common.layout.DataPipelineLayout`; `tests/unit/test_publish_layout.py`
checks `conf/modules.config`'s `publishDir` settings against it.

## Column naming

| Pattern | Meaning |
| ------- | ------- |
| `meta_*` | Metadata (barcode, batch, labels, QC flags, scores, cell index) |
| `UPPERCASE_WITH_UNDERSCORE` | CellProfiler morphological feature columns |
| `tmp_*` | Ephemeral intermediates, dropped before output |

The `meta_` prefix is load-bearing: `FEATURE_SELECTOR` is defined as
`exclude("^meta_.*$")`, so any new non-`meta_` column is treated as a feature.

## Components

- `src/fisseq_data_pipeline/` — Python package: `input` (INPUT), the standalone
  `aggregate` entry point and `config`. The other stages are
  `fisseq_common.stages.<stage>` (fisseq-common's `stages` extra).
- `modules/local/input.nf` — this pipeline's own process (INPUT)
- `packages/fisseq-common/nextflow/modules/local/<stage>/main.nf` — the shared processes,
  included by `workflows/fisseq.nf` by relative path
  (`../../fisseq-common/nextflow/modules/local/<stage>/main`)
- `conf/modules.config` — this pipeline's `ext.args`, `ext.seed` and `publishDir` for each
  shared process
- `workflows/fisseq.nf` — the DAG
- `main.nf` — entry point
- `params.yaml` — every parameter default
- `nextflow.config` — executor/profile/container settings only
- `Dockerfile` — the single image every process runs in
