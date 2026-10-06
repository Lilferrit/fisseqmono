# Shared stages

Every stage both pipelines run downstream of their own cell tables lives whole in
`fisseq_common.stages` (the `stages` extra): the algorithm, its Hydra structured config and its
entry point, `python -m fisseq_common.stages.<stage>`. Both pipelines run the same entry points
from the same [Nextflow modules](nextflow.md). What differs between them is a config field, set
by each pipeline's `conf/modules.config`.

| Entry point | Nextflow process(es) | Config |
|---|---|---|
| [`qcfilter`](#qcfilter) | `QC_FILTER` | `QcFilterParams` |
| [`filter`](#filter) | `NORMALIZE`, `NORMALIZE_CP_FEATURES` | `FilterParams` |
| [`ovwt`](#ovwt) | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` | `OvwtConfig` |
| [`aggregate`](#aggregate) | `AGGREGATE_FEATURE_TYPE_BATCHWISE`, `AGGREGATE_FEATURE_TYPE_PASSTHROUGH`, `AGGREGATE_HALF_BATCHWISE`, `AGGREGATE_FEATURE_TYPE_CP_FEATURES` | `AggregateConfig` |
| [`generatesplit`](#generatesplit) | `GENERATE_SPLIT_BATCHWISE` | `GenerateSplitParams` |
| [`correlatefeatures`](#correlatefeatures) | `CORRELATE_FEATURES_BATCHWISE` | `CorrelateFeaturesParams` |
| [`blocklist`](#blocklist) | `BLOCKLIST_BATCHWISE` | `BlocklistParams` |
| [`combineblocklists`](#combineblocklists) | `COMBINE_BLOCKLISTS_BATCHWISE` | `CombineBlocklistsParams` |
| [`finalize`](#finalize) | `FINALIZE_FEATURE_SELECT_BATCHWISE` | `FinalizeConfig` |

The `*_CP_FEATURES` processes are the embeddings pipeline's CellProfiler-feature track.
The full function documentation is in the [API reference](api.md#stages-stages-extra).

## Common config fields

Every stage config extends `AppConfig` (`fisseq_common.stages.config`):

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for the output files; created if absent. |
| `output_root` | `null` | If set, every output file is prefixed `{output_root}.` instead. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | The one seed every stochastic stage reads. A stage that must differ from its siblings derives a fixed offset from it (GENERATE_SPLIT: `random_seed + bootstrap_idx`); no config has a seed of its own. |

### Cell identity and the normalized cell table

The filter stage publishes no normalized copy of the cells. Every stage downstream of it
rebuilds the normalized table on demand (`fisseq_common.stages.filter.load_cells`) from three
files, the `CellsInput` fields:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cells_file` | **required** | The pipeline's cell feature table: the data pipeline's `qc_filter/<batch>/filtered_cells.parquet`; the embeddings pipeline's `embeddings/<batch>/embeddings.parquet` or `cp_features/<batch>/cp_features.parquet`. |
| `filtered_keys_file` | **required** | The filter stage's `filtered_keys.parquet`. |
| `normalizer_file` | **required** | The filter stage's `normalizer.parquet`. |
| `join_keys` | `[meta_cell_index, meta_variant_tag]` | The columns that identify a cell in both files. |
| `feature_selector` | `"features"` | Which columns are features: `"features"` (every non-`meta_` column) or `"embeddings"` (`emb_NNNN` only). |

The keys are joined to the cell table on `join_keys`. Every `meta_*` column comes from the
keys (QC_FILTER's side), the features from the cell table, and the normalizer is applied to the
features. The rows are sorted on `row_keys(join_keys)`: `join_keys` plus `meta_variant_tag`.

Each pipeline has its own cell identity:

- **Data pipeline** (`DATA_JOIN_KEYS`, the default): `(meta_cell_index, meta_variant_tag)`.
  QC_FILTER assigns `meta_cell_index` from the raw input row order.
- **Embeddings pipeline** (`EMBEDDINGS_JOIN_KEYS`):
  `(meta_batch, meta_well, meta_tile, meta_cell_index)`. `meta_cell_index` is the cell's index
  within its tile.

A QC pseudo-variant row (see [qcfilter](#qcfilter)) is a copy of a cell under its own
`meta_variant_tag`. Because the `meta_*` columns come from the QC side, it gets its source
cell's features and keeps its own label and tag. Sorting and splitting on `row_keys` keeps it
distinct from its source cell.

## qcfilter

`python -m fisseq_common.stages.qcfilter` (Nextflow `QC_FILTER`) reads one or more cell files
(CSV or Parquet), renames the barcode, amino-acid-change and edit-distance columns to their
`meta_*` names, and applies three filters in order:

1. **Edit distance**: drops cells with an edit distance above `edit_distance_threshold`.
2. **Barcode cell count**: drops barcodes with fewer than `bc_threshold` cells.
3. **Variant barcode count**: drops variants with fewer than `variant_bc_threshold` distinct
   barcodes.

If `n_variants` is set, the variants whose class is in `variant_downsample_classes` are first
restricted to at most `n_variants` distinct variants: the highest-cell-count ones
(`variant_downsample_mode: "top"`) or a seeded random sample (`"random"`). Every other class
passes through. Variants listed in `variant_allow_list_file` bypass the cap and don't count
against it.

If `downsample_amounts` is set (a float or int, or a list of them), `filtered_cells.parquet`
also gets "pseudo-variant" rows, drawn reproducibly from the QC-passing cells of
`downsample_classes`. A float in `(0, 1]` keeps that fraction of each variant's cells; an int
keeps that many and skips variants with fewer. Each amount is its own group, labelled
`<variant>:downsample-<amount>` with `meta_variant_tag = "downsample-<amount>"`. The pseudo rows
carry their own tag, so they never share a `row_keys` identity with their source cell.
`barcode_counts.parquet` and `variants_per_barcode.parquet` are computed before this step and
never include pseudo-variant rows.

### Config fields (`QcFilterParams`)

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_files` | **required** | Path or list of paths to the cell files. |
| `bc_threshold` | `10` | Minimum cells per barcode. |
| `variant_bc_threshold` | `4` | Minimum distinct barcodes per variant. |
| `edit_distance_threshold` | `1` | Maximum edit distance. |
| `barcode_col_name` | `"meta_barcode"` | Input barcode column. |
| `aa_changes_col_name` | `"meta_aa_changes"` | Input amino-acid-change column. |
| `edit_distance_col_name` | `"meta_edit_distance"` | Input edit-distance column. |
| `label_column` | `"meta_aa_changes"` | Output variant label column. |
| `n_variants` | `null` | Cap on the distinct variants of `variant_downsample_classes`; `null` disables it. |
| `variant_downsample_classes` | `["Single Missense"]` | Classes the `n_variants` cap applies to. |
| `variant_downsample_mode` | `"top"` | `"top"` or `"random"` (seeded by `random_seed`). |
| `variant_allow_list_file` | `null` | Parquet with a `label_column` column of variants that bypass the cap. Ignored, with a warning, when `n_variants` is unset. |
| `downsample_amounts` | `null` | Pseudo-variant amounts; `null` disables them. |
| `downsample_classes` | `["Synonymous", "Single Missense"]` | Classes pseudo-variants are drawn from. |
| `sort_output_by` | `null` | Columns to sort `filtered_cells` by. QC's joins don't preserve row order, and every seeded downstream step depends on it. `null` keeps the join order. |
| `assign_cell_index` | `false` | Assign `meta_cell_index` from the input row order. |

### Outputs

- `filtered_cells.parquet`: the cells passing all three filters (plus any pseudo-variant rows).
- `barcode_counts.parquet`: per-barcode cell counts and pass/fail flags.
- `variants_per_barcode.parquet`: per-variant barcode counts and pass/fail flags.

### What each pipeline sets

| | Data pipeline | Embeddings pipeline |
|---|---|---|
| Input | INPUT's `input/<batch>.parquet` | BUILD_CELL_METADATA's `metadata.parquet` |
| Column names | `barcode_col_name=upBarcode`, `aa_changes_col_name=aaChanges`, `edit_distance_col_name=editDistance` | the defaults (already `meta_*`) |
| `sort_output_by` | `[meta_cell_index, meta_variant_tag]` | `[meta_batch, meta_well, meta_tile, meta_cell_index, meta_variant_tag]` |
| `assign_cell_index` | `true` | `false` (the input has a per-tile index) |

### Example

```bash
uv run python -m fisseq_common.stages.qcfilter \
    output_dir=./out \
    'cell_files=[data/plate1.parquet,data/plate2.parquet]' \
    barcode_col_name=upBarcode aa_changes_col_name=aaChanges edit_distance_col_name=editDistance \
    "'sort_output_by=[meta_cell_index,meta_variant_tag]'" assign_cell_index=true
```

## filter

`python -m fisseq_common.stages.filter` (Nextflow `NORMALIZE`; the embeddings pipeline also
runs it as `NORMALIZE_CP_FEATURES`) joins QC_FILTER's QC-passed cells to the pipeline's cell
feature table on `join_keys`, marks the control rows `meta_is_control`, and fits a per-feature
z-score (`fisseq_common.normalizer.Normalizer`) on them. Zero-variance features get a null
standard deviation.

It writes no normalized copy of the cells: only the keys and the normalizer. Every downstream
stage rebuilds the normalized table from these (see
[Cell identity](#cell-identity-and-the-normalized-cell-table)).

The controls are the `control` field: a SQL predicate, by default wildtype
(`meta_aa_changes = 'WT'`), in both pipelines. `"synonymous"` selects untagged synonymous
variants instead; no pipeline uses it.

### Config fields (`FilterParams`)

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cells_file` | **required** | The cell feature table. |
| `qc_passed_file` | **required** | QC_FILTER's `filtered_cells.parquet`, the source of every `meta_*` column. The data pipeline passes it as `cells_file` too: its QC output carries the features. |
| `label_column` | `"meta_aa_changes"` | Variant label column. |
| `control` | `"meta_aa_changes = 'WT'"` | Which rows the normalizer is fit on. |
| `join_keys` | `[meta_cell_index, meta_variant_tag]` | The columns identifying a cell in both files. |
| `batch_name` | `null` | If set, written as `meta_batch`. |

### Outputs

- `filtered_keys.parquet`: every `meta_*` column of the QC-passed cells, plus
  `meta_is_control`; no features. Sorted on `row_keys(join_keys)`.
- `normalizer.parquet`: the control-fitted normalizer.

### What each pipeline sets

| | Data pipeline | Embeddings pipeline |
|---|---|---|
| `cells_file` | `filtered_cells.parquet` (same file as `qc_passed_file`) | `embeddings.parquet` / `cp_features.parquet` |
| `join_keys` | default | `[meta_batch, meta_well, meta_tile, meta_cell_index]` |
| `batch_name` | the batch stem (its QC output has no batch column) | unset |

### Example

```bash
uv run python -m fisseq_common.stages.filter \
    output_dir=./out \
    cells_file=out/qc_filter/batch1/filtered_cells.parquet \
    qc_passed_file=out/qc_filter/batch1/filtered_cells.parquet \
    batch_name=batch1
```

## ovwt

`python -m fisseq_common.stages.ovwt` (Nextflow `OVWT_BATCHWISE`; the embeddings pipeline also
runs it as `OVWT_BATCHWISE_CP_FEATURES`) scores how distinguishable each variant is from
wildtype, one experiment at a time. It rebuilds the normalized cells from the
[`CellsInput`](#cell-identity-and-the-normalized-cell-table) fields.

For each non-wildtype variant it cross-validates XGBoost binary classifiers over the variant's
cells plus the (optionally downsampled) wildtype pool. The folds partition the data, so every
cell gets exactly one *out-of-fold* score, from a model that never saw it. Wildtype is the
positive class: the models predict P(wildtype).

### Two cross-validation modes

`cv_mode` selects how the folds are cut. Both give every cell one out-of-fold score and the
same output columns; what changes is the question they answer.

- **`"kfold"`** (default): `n_folds` folds, stratified jointly on `(meta_barcode, is_wt)`. A
  stratum with fewer than 10 members collapses into a shared `rare|wt` / `rare|variant`
  bucket; the wildtype/variant half of the key is never merged. Every fold's model has seen
  every barcode, so the AUROCs measure separability *within* the barcodes trained on.
- **`"barcode_holdout"`**: whole barcodes are held out of training, one barcode (or one group
  of barcodes) per fold, so no model scores a barcode it was trained on. Wildtype cells are
  still split across the folds. `n_folds` caps the fold count: `null` gives one fold per
  barcode; `k` below the barcode count packs the barcodes into `k` cell-count-balanced groups
  (greedy, longest first; deterministic); `k` at or above the barcode count is the same as
  `null`. A variant with one barcode can't be scored and is skipped with a warning. Use this
  mode to ask whether a variant's signal *generalizes to an unseen barcode*.

### Scores per variant

| Column | Meaning |
| ------ | ------- |
| `auroc_pooled` | AUROC over all of the variant's out-of-fold scores at once. |
| `auroc_median_barcode` | Each of the variant's barcodes scored separately against the full wildtype set, then medianed. Shows whether the signal is broad-based or driven by one or two barcodes. |
| `auroc_folds` | Per-fold test AUROCs, each from that fold's own model; `null` where a fold's test slice holds one class. |
| `auroc_median_fold` | Median of the non-null `auroc_folds`. |

`auroc_pooled` and `auroc_median_barcode` put scores from different fold models (each with its
own calibrator) into one ROC curve. Those scores don't share a scale, which can inflate the
AUROC. A per-fold AUROC only ranks one model's scores.

### Wildtype downsampling

`downsample_wt: true` shrinks the wildtype pool to the size of the largest remaining variant
group, barcode-proportionally: a wildtype barcode holding fraction `p` of the pool keeps about
`p × target` of its cells. `auroc_median_barcode` measures every variant barcode against this
same wildtype set, so a uniform draw could skew it.

### Resilience

A variant whose folds raise (most often a stratum too small for the inner
train/calibration split) is skipped with a warning instead of aborting the run. If no variant
survives, the outputs are still written, empty but typed. Each variant logs a `[i/N]` header,
one line per fold and per barcode, and a summary.

The cells come in z-scored against wildtype. The synonymous re-centering of the AUROCs happens
across experiments, in [fisseqborn](../fisseqborn/index.md); don't add a second normalization
here.

### Config fields (`OvwtConfig` = `CellsInput` + `OvwtParams`)

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cells_file`, `filtered_keys_file`, `normalizer_file`, `join_keys`, `feature_selector` | | See [`CellsInput`](#cell-identity-and-the-normalized-cell-table). |
| `label_column` | `"meta_aa_changes"` | Variant label column. |
| `wt_label` | `"WT"` | Label of the wildtype cells. |
| `cv_mode` | `"kfold"` | `"kfold"` or `"barcode_holdout"`. |
| `n_folds` | `5` | Folds per variant (`"kfold"`), or a cap on the barcode groups (`"barcode_holdout"`). `null` is valid only under `"barcode_holdout"`; otherwise at least 2. |
| `calibrate` | `true` | Fit a per-fold sigmoid (Platt) calibrator on a slice held out of that fold's training data. |
| `min_cells` | `250` | Drop variants with fewer cells before scoring; wildtype is always kept. `null` disables it. |
| `downsample_wt` | `true` | Barcode-proportional wildtype downsampling. |
| `xgboost.*` | see `XGBoostConfig` | `num_boost_round`, `early_stopping_rounds`, `weigh_samples` and the booster `params` (`fisseq_common.stages.xgbparams`). |

`random_seed` drives the fold shuffle, the inner split, the wildtype downsample and XGBoost's
own `seed`.

### Outputs

| File | Contents |
| ---- | -------- |
| `results.parquet` | One row per scored variant: `label_column`, `auroc_pooled`, `auroc_median_barcode`, `auroc_folds`, `auroc_median_fold`, `meta_n_barcodes`, `meta_n_cells`. |
| `cell_scores.parquet` | One row per cell per variant it was scored against: every `meta_*` column, `score` (out-of-fold) and `meta_variant_scored_against`. Wildtype cells appear once per variant. |
| `models.pkl` | `dict[variant, list[(Booster, calibrator or None)]]`, one tuple per fold. |

### What each pipeline sets

The Nextflow module passes the `ovwt_*` params (`ovwt_wt_label`, `ovwt_cv_mode`,
`ovwt_n_folds`, `ovwt_calibrate`, `ovwt_min_cells`, `ovwt_downsample_wt`), the same in both
pipelines. The embeddings pipeline sets `join_keys` and `feature_selector=embeddings`
(`OVWT_BATCHWISE`) or `feature_selector=features` (`OVWT_BATCHWISE_CP_FEATURES`).

### Example

```bash
uv run python -m fisseq_common.stages.ovwt \
    output_dir=./out \
    cells_file=out/qc_filter/batch1/filtered_cells.parquet \
    filtered_keys_file=out/normalization/batch1/filtered_keys.parquet \
    normalizer_file=out/normalization/batch1/normalizer.parquet \
    cv_mode=barcode_holdout n_folds=null
```

## aggregate

`python -m fisseq_common.stages.aggregate` runs one aggregation method over an experiment's
normalized cells, or over one GENERATE_SPLIT half, and writes the lean
`[label_column] + <stat columns>` table, one row per non-control variant, sorted by label.
Every aggregator excludes the control rows (`meta_is_control`) before grouping: the controls
are the reference distribution, not something scored against it.

The pipelines run it under three process names (four in the embeddings pipeline):

| Process | Cells | `normalize_to_synonymous` | Published to |
|---|---|---|---|
| `AGGREGATE_FEATURE_TYPE_BATCHWISE` | every cell | `true` | `feature_select_batchwise/<batch>/aggregates/<method>.parquet` |
| `AGGREGATE_FEATURE_TYPE_PASSTHROUGH` | every cell | `false` | `feature_select_batchwise/<batch>/passthrough_aggregates/<method>.parquet` |
| `AGGREGATE_HALF_BATCHWISE` | one bootstrap half | `false` | `feature_select_batchwise/<batch>/half_aggregates/bootstrap_<N>/<method>/half<K>_agg.parquet` |
| `AGGREGATE_FEATURE_TYPE_CP_FEATURES` (embeddings) | every cell | `true` | `feature_select_batchwise_cp_features/<batch>/aggregates/<method>.parquet` |

### Aggregators

| `aggregator` | Statistic |
| ----- | ----------- |
| `mean`, `median`, `MAD`, `std` | Per-variant location and spread. |
| `KS` | Kolmogorov-Smirnov statistic against the control distribution. |
| `signedKS` | `KS`, signed by which empirical CDF is larger at the maximizing point: positive when the variant's CDF is larger there (it skews lower). |
| `QQ` | Q-Q Pearson correlation against the control distribution. |
| `AUROC` | AUROC against the control distribution. Directional: `0.5` is identical, `1.0` consistently higher, `0.0` consistently lower. |
| `KSnegLogP`, `AUROCnegLogP` | `-log10(p)` of the KS and AUROC statistics (asymptotic approximations, no multiple-testing correction). |

Each output column is `<feature>_<method>`. There is no combined option: the pipelines run one
task per method.

### Synonymous normalization

With `normalize_to_synonymous=true` the per-variant table is z-scored before it is written: a
`Normalizer` is fit (mean, `ddof=1` std) on the untagged synonymous variants' rows and applied
to every row. This makes aggregates comparable across experiments. It needs at least two
synonymous variants; with fewer, every column comes out null. The bootstrap halves and the
passthrough methods (p-values keep their own scale) stay raw.

### Wildtype downsampling and seeds

`downsample_wt` downsamples the control (wildtype) rows first, seeded with `random_seed`: a
float in `(0, 1)` keeps that fraction, an int that many. `AGGREGATE_HALF_BATCHWISE` is seeded
with `random_seed + rep * 2 + half` (`ext.seed`), so every half of every replicate draws its own
wildtype subsample. The full aggregates use `random_seed`.

### Feature chunking

`feature_chunk_size` feature columns are aggregated per Polars query, and the chunks joined
back on the label. The values are identical at every chunk size; peak memory scales with
`chunk_size × n_variant_labels` (and the cross-joined control pool, for the reference-based
aggregators). Aggregating ~1731 features in one query OOM-killed every `KS`/`AUROC` task of the
111925 run, and `errorStrategy 'ignore'` hid it. If those tasks come back killed, halve
`params.aggregate_feature_chunk_size`; a run with only `mean`/`median`/`std`/`MAD` can raise it.
`null` disables chunking.

### Config fields (`AggregateConfig` = `CellsInput` + `AppConfig`)

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cells_file`, `filtered_keys_file`, `normalizer_file`, `join_keys`, `feature_selector` | | See [`CellsInput`](#cell-identity-and-the-normalized-cell-table). |
| `label_column` | `"meta_aa_changes"` | Variant label column. |
| `aggregator` | **required** | One of the aggregators above. |
| `split_file` | `null` | A GENERATE_SPLIT half, selected with a semi-join on `row_keys(join_keys)`; `null` aggregates every cell. |
| `downsample_wt` | `null` | Control downsampling (see above). |
| `feature_chunk_size` | `32` | See [Feature chunking](#feature-chunking). |
| `normalize_to_synonymous` | `false` | See [Synonymous normalization](#synonymous-normalization). |
| `output_name` | `"aggregate"` | The output is `{output_name}.parquet`; the Nextflow module sets the method name. |

### What each pipeline sets

Both: `downsample_wt` from `params.feature_select_downsample_wt`, `normalize_to_synonymous` per
process (table above), `ext.seed` on `AGGREGATE_HALF_BATCHWISE`. The embeddings pipeline also
sets its `join_keys` and `feature_selector` (`embeddings`; `features` on the CP track).

### Example

```bash
uv run python -m fisseq_common.stages.aggregate \
    output_dir=./out \
    cells_file=out/qc_filter/batch1/filtered_cells.parquet \
    filtered_keys_file=out/normalization/batch1/filtered_keys.parquet \
    normalizer_file=out/normalization/batch1/normalizer.parquet \
    aggregator=KS output_name=KS \
    split_file=out/feature_select_batchwise/batch1/splits/bootstrap_1/half1.parquet \
    random_seed=3
```

## generatesplit

`python -m fisseq_common.stages.generatesplit` (Nextflow `GENERATE_SPLIT_BATCHWISE`) splits an
experiment's QC-passed cells (the filter stage's `filtered_keys.parquet`, nothing else) into two
halves, stratified by variant label, for one bootstrap replicate. The split is seeded with
`random_seed + bootstrap_idx`, so every replicate is distinct and reproducible.

### Config fields (`GenerateSplitParams`)

| Field | Default | Description |
| ----- | ------- | ----------- |
| `filtered_keys_file` | **required** | The filter stage's `filtered_keys.parquet`. |
| `label_column` | `"meta_aa_changes"` | The column the split is stratified on. |
| `bootstrap_idx` | `1` | Replicate number. |
| `join_keys` | `[meta_cell_index, meta_variant_tag]` | The pipeline's cell identity. |

### Outputs

`half1.parquet`, `half2.parquet`: each half's cells as `row_keys(join_keys)` columns. The
embeddings pipeline sets its `join_keys`.

```bash
uv run python -m fisseq_common.stages.generatesplit \
    output_dir=./out \
    filtered_keys_file=out/normalization/batch1/filtered_keys.parquet \
    bootstrap_idx=3
```

## correlatefeatures

`python -m fisseq_common.stages.correlatefeatures` (Nextflow `CORRELATE_FEATURES_BATCHWISE`)
correlates every aggregate column across the variants of one replicate's two half aggregates,
for one method.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `half1_file` | **required** | The first half's aggregate. |
| `half2_file` | **required** | The second half's aggregate, same method. |
| `label_column` | `"meta_aa_changes"` | Variant label column. |
| `output_name` | `"correlations"` | Output basename; the Nextflow module sets the method name. |

Output: one row per feature, `feature`, `r`, `r_squared`. An undefined correlation (a constant
column) is null, so BLOCKLIST's median skips it. Published to
`feature_select_batchwise/<batch>/correlations/<method>/bootstrap_<N>.parquet`. Neither pipeline
sets anything.

```bash
uv run python -m fisseq_common.stages.correlatefeatures \
    output_dir=./out half1_file=half1/KS.parquet half2_file=half2/KS.parquet output_name=KS
```

## blocklist

`python -m fisseq_common.stages.blocklist` (Nextflow `BLOCKLIST_BATCHWISE`) gathers every
replicate's correlations for one method, the one synchronization point across replicates. A
feature is reproducible (`feature_ok`) when its median `r` across replicates is at least
`minimum_correlation`; a null median (undefined in every replicate) is blocked.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `correlation_files` | **required** | Glob matching one method's correlation files. |
| `minimum_correlation` | `0.5` | Minimum median `r`. Both pipelines set `params.feature_select_min_correlation`. |
| `output_name` | `"blocklist"` | Output basename; the Nextflow module sets the method name. |

Output: `feature`, `median_r`, `feature_ok`, sorted by `feature`. Published to
`feature_select_batchwise/<batch>/blocklists/<method>.parquet`.

```bash
uv run python -m fisseq_common.stages.blocklist \
    output_dir=./out 'correlation_files=correlations/KS/*.parquet' minimum_correlation=0.5
```

## combineblocklists

`python -m fisseq_common.stages.combineblocklists` (Nextflow `COMBINE_BLOCKLISTS_BATCHWISE`)
concatenates one experiment's per-method blocklists into `blocklist.parquet`, sorted by
`feature`. The stat suffixes keep each method's feature names disjoint. One field,
`blocklist_files` (a glob). Published to `feature_select_batchwise/<batch>/blocklist.parquet`.

```bash
uv run python -m fisseq_common.stages.combineblocklists \
    output_dir=./out 'blocklist_files=blocklists/*.parquet'
```

## finalize

`python -m fisseq_common.stages.finalize` (Nextflow `FINALIZE_FEATURE_SELECT_BATCHWISE`) builds
one experiment's per-variant table:

1. Join every selected method's aggregates (`aggregates/<method>.parquet`) on the label.
2. Drop the columns the combined blocklist marks as not reproducible. A null verdict counts as
   blocked; columns the blocklist doesn't mention are kept. This is the only feature selection.
3. Mark the untagged synonymous variants `meta_is_control` (kept in the output) and z-score
   every feature against them. The pipelines' aggregates are already z-scored, so this is
   effectively a no-op; it keeps the stage correct on raw aggregates. Needs at least two
   synonymous variants.
4. Optionally add `meta_impact_score`: the cosine distance from the synonymous variants'
   median.
5. Join the per-variant metadata counts, computed from the filter stage's
   `filtered_keys.parquet`.
6. Join the passthrough aggregates, raw and last, so steps 3 and 4 never see them.

The output, `output.parquet`, has one row per variant, sorted by label. It can carry non-`meta_`
columns (the passthrough methods) that were never blocklisted or normalized.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `feature_type_files` | **required** | Glob matching the selected methods' aggregates; matching nothing is an error. |
| `block_list_file` | **required** | COMBINE_BLOCKLISTS' `blocklist.parquet`. |
| `filtered_keys_file` | **required** | The filter stage's `filtered_keys.parquet`. |
| `passthrough_feature_type_files` | `null` | Glob matching the passthrough aggregates; `null` or no match joins none. |
| `label_column` | `"meta_aa_changes"` | Variant label column. |
| `compute_impact_score` | `true` | Add `meta_impact_score`. |
| `output_name` | `"output"` | Output basename. |

Published to `feature_select_batchwise/<batch>/output.parquet`. Neither pipeline sets anything.
The embeddings pipeline's CellProfiler track has no bootstrap feature selection, so no
`output.parquet`.

```bash
uv run python -m fisseq_common.stages.finalize \
    output_dir=./out \
    'feature_type_files=out/feature_select_batchwise/batch1/aggregates/*.parquet' \
    block_list_file=out/feature_select_batchwise/batch1/blocklist.parquet \
    filtered_keys_file=out/normalization/batch1/filtered_keys.parquet
```

## The two method lists

`params.feature_select_types` names the methods that *decide* which features survive: each is
aggregated on every cell and on both halves of every bootstrap replicate, correlated and
blocklisted. `params.feature_select_passthrough_types` names methods that are only aggregated
on every cell (raw) and joined onto `output.parquet` last: no halves, no correlation, no
blocklist, no synonymous z-score, no impact score. It exists for the p-value aggregators
(`KSnegLogP`, `AUROCnegLogP`). The two lists must be disjoint; both pipelines reject an
overlap before any stage runs.
