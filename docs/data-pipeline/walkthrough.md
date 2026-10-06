# Walkthrough: running the pipeline end to end

This walks through a full run of the default `fisseq` workflow, from raw
CellProfiler output to final feature-selected results.

## 1. Install

```bash
git clone https://github.com/Lilferrit/fisseqmono.git
cd fisseqmono
uv sync
cd packages/fisseq-data-pipeline
```

See [Installation](installation.md) for details, including cluster/HPC setup.

## 2. Declare your experiments

Experiments are declared as a list under `experiments:` in `params.yaml`. Each
entry names its raw CellProfiler feature matrix (or matrices) via `input_paths`;
there is no mode where the pipeline scans a directory of pre-staged Parquet
files. See [Configuration](configuration.md#declaring-experiments) for the full
schema and [CLI Reference: Input](cli/input.md) for what INPUT does with them.

```yaml
# params.yaml
experiments:
  - batch_stem: batch1
    input_paths: [/path/to/batch1_raw.parquet]
  - batch_stem: batch2
    input_paths: [/path/to/batch2_raw.parquet]
```

`<pipeline_dir>` is just the output root — it needs no pre-existing contents.

## 3. Run the pipeline

```bash
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
```

This chains every stage described in [Architecture](architecture.md):

1. `INPUT` — merges each experiment's `input_paths` into one
   `input/<batch_stem>.parquet` (always runs, once per experiment).
2. `QC_FILTER` — edit-distance, barcode-count and variant-barcode-count
   filtering, per experiment. Also assigns `meta_cell_index`, the stable
   per-cell identity everything downstream depends on for reproducibility.
3. `NORMALIZE` — z-score normalization fit on WT control cells, per experiment.
   It publishes only the QC-passed keys and the normalizer; every later stage rebuilds the
   normalized cells from them.
4. `OVWT_BATCHWISE` — k-fold cross-validated one-vs-wildtype scoring, per
   experiment (`params.run_ovwt`). Each variant gets a pooled AUROC and a
   median-of-per-barcode AUROC, and every cell gets one out-of-fold score.
5. Bootstrap feature selection (`params.run_feature_selection`) — see
   [Nextflow Workflow](nextflow.md#processes) for the stage breakdown. The
   per-feature-type aggregates are z-scored against the experiment's synonymous
   variants, so each experiment needs at least two of them.

Every stage runs per experiment. Combining experiments (cross-experiment
medians, AUROC re-centering against synonymous variants) is done downstream by
[fisseqborn](../fisseqborn/index.md) — see
[Architecture: Cross-experiment aggregation](architecture.md#cross-experiment-aggregation).

Override any [parameter](configuration.md#parameter-reference) on the command
line:

```bash
nextflow run . \
    --pipeline_dir /path/to/experiment \
    -params-file params.yaml \
    --barcode_count_threshold 15
```

To run on a cluster, supply your own config:

```bash
nextflow run . -c your.config -profile sge \
    --pipeline_dir /path/to/experiment -params-file params.yaml
```

If a run is interrupted, resume from the last completed task:

```bash
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -resume
```

## 4. Inspect the results

All outputs land under `<pipeline_dir>` — see
[Architecture: Output layout](architecture.md#output-layout) for the full tree.
The results most analyses care about:

- `<pipeline_dir>/feature_select_batchwise/<batch_stem>/output.parquet` — final
  per-variant, feature-selected profiles (plus any raw passthrough columns).
- `<pipeline_dir>/feature_select_batchwise/<batch_stem>/{aggregates,passthrough_aggregates,blocklists}/<feature_type>.parquet`
  — the per-feature-type inputs fisseqborn reads for cross-experiment
  aggregation (`aggregates/` synonymous-z-scored, `passthrough_aggregates/` raw).
- `<pipeline_dir>/ovwt_batchwise/<batch_stem>/results.parquet` — per-variant
  `auroc_pooled`, `auroc_median_barcode`, per-fold `auroc_folds` and
  `auroc_median_fold`. Synonymous correction and the cross-experiment median of
  these happen downstream in fisseqborn.

## 5. Running individual steps

INPUT runs `python -m fisseq_data_pipeline.input` ([CLI Reference](cli/input.md)); every
other process runs a shared stage, `python -m fisseq_common.stages.<stage>`. To debug or rerun
one stage manually, invoke it directly — see [Shared stages](../common/stages.md) for each
stage's config fields:

```bash
uv run python -m fisseq_common.stages.qcfilter \
    output_dir=./out/qc \
    'cell_files=[out/input/plate1.parquet]' \
    barcode_col_name=upBarcode aa_changes_col_name=aaChanges edit_distance_col_name=editDistance \
    "'sort_output_by=[meta_cell_index,meta_variant_tag]'" assign_cell_index=true

uv run python -m fisseq_common.stages.filter \
    output_dir=./out/norm \
    cells_file=out/qc/filtered_cells.parquet \
    qc_passed_file=out/qc/filtered_cells.parquet \
    batch_name=plate1
```

The arguments this pipeline passes are its `conf/modules.config` `ext.args` plus what the
[shared module](../common/nextflow.md) passes; a task's `.command.sh` under `work/` shows the
exact command.
