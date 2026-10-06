# Configuration

Every pipeline parameter lives in **`params.yaml`** at the package root
(`packages/fisseq-data-pipeline/`). It must be
passed explicitly:

```bash
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
```

`nextflow.config` carries executor, profile and container settings **only** —
never parameter defaults.

## Precedence

1. A bare CLI flag (`--ovwt_min_cells 500`) — highest.
2. `-params-file params.yaml`.
3. The Python Hydra dataclass default for that stage (see [Shared stages](../common/stages.md)).

Nextflow accepts only one `-params-file` at a time, so to run an alternate
parameter set, copy `params.yaml` and edit the copy.

!!! warning "Do not add a `params { }` block to `nextflow.config`"
    Not even an empty one. Combined with `-params-file`, Nextflow's
    `ConfigBuilder` before 26.04.6 misattributes every `params.yaml` key into the
    `process{}` scope and fails with `Unknown config attribute 'params'`.

## Declaring experiments

`experiments:` is a list of maps, one per experiment. This replaces the former
mandatory `<pipeline_dir>/configs/*.yaml` directory.

```yaml
experiments:
  - batch_stem: plate1
    input_paths:
      - /data/plate1/cellprofiler_features.csv
      - /data/plate1/barcode_calls.parquet
  - batch_stem: plate2
    input_paths: [/data/plate2/cellprofiler_features.csv]
    csv_schema_scan_rows: null             # per-experiment override
  - batch_stem: plate3
    input_paths: [/data/plate3/features.parquet]
```

An entry may set **only** these keys:

| Key | Required | Meaning |
| --- | -------- | ------- |
| `batch_stem` | yes | Unique experiment id. Names every output subdirectory. Must be unique across the list. |
| `input_paths` | yes | Raw CSV/parquet source files that INPUT merges. No pipeline-wide default exists — a list of raw files is inherently per-experiment. |
| `feature_allowlist_file` | no | INPUT-stage; falls back to the pipeline-wide default. |
| `feature_blocklist_file` | no | Likewise. |
| `csv_schema_scan_rows` | no | Likewise. |

Any other key is **rejected with an error naming it**, rather than silently
ignored. If you want to change a QC threshold or an OvWT hyperparameter, set it
at the top level of `params.yaml` — it applies pipeline-wide. This is a
deliberate simplification: per-experiment override of arbitrary parameters was
removed for the release.

The workflow also fails fast on an empty `experiments:` list, a non-map entry, a
missing or blank `batch_stem`, a missing or empty `input_paths`, and duplicate
`batch_stem` values.

## Cross-experiment aggregation

Every stage runs per experiment, and nothing in the pipeline combines
experiments. Cross-experiment aggregation — merging blocklists, per-variant
medians across experiments, AUROC re-centering against synonymous variants — is
done downstream by [fisseqborn](../fisseqborn/index.md), which
reads the published per-experiment outputs (see
[Architecture](architecture.md#cross-experiment-aggregation)).

## The one random seed

```yaml
random_seed: 0
```

This is the only seed in the pipeline. Every stochastic step derives from it:
QC pseudo-variant downsampling, the feature-selection bootstrap splits and their
wildtype subsampling, OvWT's fold shuffle / inner calibration split / XGBoost
`seed`. Changing it moves all of them coherently.

Stages that must differ from one another derive a fixed offset rather than
owning a seed of their own (`GENERATE_SPLIT_BATCHWISE` uses `random_seed + bootstrap_idx`;
`AGGREGATE_HALF_BATCHWISE` uses `random_seed + bootstrap_idx * 2 + half_num`), so
bootstrap replicates still draw independent subsamples.

There is deliberately no stage-local `random_state` anywhere, and
`tests/unit/test_config.py` fails if one reappears.

## Run gates

All pipeline-wide.

| Parameter | Default | Effect when `false` |
| --------- | ------- | ------------------- |
| `run_ovwt` | `true` | Skips `OVWT_BATCHWISE`. |
| `run_feature_selection` | `true` | Skips the whole batchwise feature-selection chain. |

## Parameter reference

### Required, no default

| Parameter | Meaning |
| --------- | ------- |
| `pipeline_dir` | Root output directory. Also settable as `--pipeline_dir`. |
| `container_image` | Image every process runs in. Pin to `:<short-sha>` for a reproducible run rather than floating on `:latest`. |
| `experiments` | See above. |

### Shared

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `random_seed` | `0` | The one seed. |
| `filter_label_column` | `"meta_aa_changes"` | Variant label column, threaded to every stage that reads one. |

### INPUT

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `feature_allowlist_file` | `null` | Restrict to these feature columns. |
| `feature_blocklist_file` | `null` | Drop these feature columns. |
| `csv_schema_scan_rows` | `100` | Rows scanned to infer CSV dtypes; `null` scans every row. No effect on parquet sources. |

### QC_FILTER

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `barcode_count_threshold` | `10` | Minimum cells per barcode. |
| `variant_barcode_count_threshold` | `4` | Minimum barcodes per variant. |
| `edit_distance_threshold` | `1` | Maximum barcode edit distance. |
| `qc_n_variants` | `null` | Cap the number of distinct variants in `qc_variant_downsample_classes`. |
| `qc_variant_downsample_classes` | `["Single Missense"]` | Classes eligible for that cap. |
| `qc_variant_downsample_mode` | `"top"` | `"top"` (highest cell count) or `"random"` (seeded by `random_seed`). |
| `qc_downsample_amounts` | `null` | Float in (0,1] or int, or a list of them: pseudo-variant downsampling per label group. Each amount gets its own `:downsample-{amount}` tag, and its rows their own `meta_variant_tag`. |
| `qc_downsample_classes` | `["Synonymous", "Single Missense"]` | Classes eligible for pseudo-variant generation. |

### OVWT_BATCHWISE

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `ovwt_wt_label` | `"WT"` | Label identifying wildtype cells. |
| `ovwt_cv_mode` | `"kfold"` | Fold scheme: `"kfold"` or `"barcode_holdout"`. |
| `ovwt_n_folds` | `5` | Cross-validation folds per variant. Under `"kfold"` the fold count outright; under `"barcode_holdout"` a cap — the variant's barcodes are packed into at most this many cell-count-balanced groups, one held out per fold. `null` = one fold per barcode, and is an error under `"kfold"`. |
| `ovwt_calibrate` | `true` | Per-fold sigmoid (Platt) calibration. |
| `ovwt_min_cells` | `250` | Minimum cells for a variant to be scored; wildtype always kept. `null` disables. |
| `ovwt_downsample_wt` | `true` | Barcode-proportional wildtype downsampling to the largest remaining variant group. |

See [Shared stages: ovwt](../common/stages.md#ovwt) for what these actually do.

### Feature selection

| Parameter | Default | Meaning |
| --------- | ------- | ------- |
| `feature_select_types` | `["mean","median","MAD","std","KS","QQ","AUROC"]` | Aggregators to compute and correlate. Their published aggregates (`feature_select_batchwise/<batch>/aggregates/`) are z-scored against the experiment's synonymous variants. |
| `feature_select_passthrough_types` | `[]` | Aggregators computed and joined onto the final per-variant table but excluded from every selection step — no bootstrap, no blocklist, no normalization; published raw to `passthrough_aggregates/`. Intended for the p-value statistics (`KSnegLogP`, `AUROCnegLogP`). Must not overlap `feature_select_types`. |
| `feature_select_bootstrap_reps` | `10` | Bootstrap replicates per feature type. |
| `feature_select_downsample_wt` | `null` | Optional wildtype downsampling in the `AGGREGATE_*` processes: a float in (0,1) keeps that fraction, an int that many. |
| `feature_select_min_correlation` | `0.5` | Median-`r` threshold for a feature to pass. |
| `aggregate_feature_chunk_size` | `32` | Feature columns the `AGGREGATE_*` processes evaluate per Polars query. A memory dial: peak memory scales with `chunk_size × n_variant_labels` (× the control pool, for the reference-based aggregators), while runtime is essentially flat in it. Halve it if a task is OOM-killed (exit 137); raise it for runs using only `mean`/`median`/`std`/`MAD`. `null` disables chunking entirely (every feature in one query) — the pre-chunking shape, for small inputs only. See [Feature chunking](../common/stages.md#feature-chunking). |

!!! note "Each experiment needs at least two synonymous variants"
    The `feature_select_types` aggregates and `FINALIZE_FEATURE_SELECT_BATCHWISE`'s
    output are z-scored against the experiment's synonymous variants (mean and
    `ddof=1` std). A single synonymous variant leaves the std undefined and
    nulls every feature for that experiment; zero-variance features are null
    too.

### Removed parameters

`run_pca`, `pca_n_components`, `run_umap` and `umap_*` are gone: the pipeline does no
dimensionality reduction (cross-experiment PCA is `fisseqborn-global`'s), and no pycytominer
step. A run that still sets one logs a warning and ignores it.

## Passing list values on the CLI

List-valued parameters cannot be expressed as a bare CLI flag:
`--feature_select_types mean,median` arrives as the single string
`"mean,median"`, not a two-element list. Set list parameters in `params.yaml`
(or a copy of it) instead.
