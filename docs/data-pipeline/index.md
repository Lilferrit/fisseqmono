# FISSEQ Data Pipeline

A Nextflow + Python workflow for processing single-cell CellProfiler morphological
profiling data from FISSEQ (Fluorescence In-Situ Sequencing) experiments. Each cell
carries a genetic variant label; the pipeline measures how each variant's cell
population differs from wildtype (WT) controls using morphological features.

## Where to start

- **[Quickstart](quickstart.md)** — the fastest path from a fresh checkout to
  a first pipeline run, including setting up a cluster config.
- **[Architecture](architecture.md)** — the full pipeline DAG, what each stage
  produces/consumes, and the output layout.
- **[Installation](installation.md)** — environment setup, including cluster/HPC
  configuration.
- **[Nextflow Workflow](nextflow.md)** — the Nextflow processes, how they're wired
  together, and how to run the pipeline (profiles).
- **[Configuration](configuration.md)** — every parameter, the `experiments:`
  schema, and the single `random_seed`.
- **[Shared stages](../common/stages.md)** — every stage after INPUT (QC filter,
  normalize, OvWT, aggregation, feature selection) is a fisseq-common entry point,
  `python -m fisseq_common.stages.<stage>`, shared with the embeddings pipeline: config
  fields, outputs and a runnable example for each.
- **CLI Reference** — this package's own entry points: [INPUT](cli/input.md) and the
  [standalone aggregate](cli/aggregate.md).
- **API Reference** — function/class-level documentation for this package's modules; the
  shared stages' is in [fisseq-common's](../common/api.md).
- **[Walkthrough](walkthrough.md)** — a complete end-to-end run, from raw
  CellProfiler output to final feature-selected results.

## Pipeline at a glance

```text
params.yaml (experiments: [...])  ──►  INPUT  ──►  input/<batch_stem>.parquet
     │
     ▼
QC_FILTER   (per experiment)
     │
     ▼
NORMALIZE   (per experiment)          z-score fit on WT control cells
     │
     ├──► OVWT_BATCHWISE               (per experiment — params.run_ovwt)
     │
     └──► Feature selection            (per experiment — params.run_feature_selection)
```

Every output is per experiment; cross-experiment aggregation is done downstream
by [fisseqborn](../fisseqborn/index.md).

See [Architecture](architecture.md) for the full diagram and stage-by-stage detail.
