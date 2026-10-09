# CellProfiler Feature Dataset (`BUILD_CP_FEATURES`)

`python -m fisseq_embeddings_pipeline.cp_features` (Nextflow process `BUILD_CP_FEATURES`) selects the CellProfiler feature columns
of `BUILD_CELL_IMAGES`' `cell_table.parquet` (every column without a
`meta_` prefix, already under CellProfiler's own names) into one
per-experiment `cp_features.parquet` -- the CellProfiler-feature analog of
`EMBED_CELLS`' `embeddings.parquet`. Its `meta_*` columns are the same seven
`BUILD_CELL_METADATA` writes (`utils.cell_table.cell_metadata_exprs`),
`meta_batch` included. A table with no CellProfiler columns (an experiment
whose `cp_features` was off when `BUILD_CELL_IMAGES` ran) logs a warning.

This stage no longer discovers tiles, reads any CSV, or touches
`starcall-workflow`'s tree at all -- `BUILD_CELL_IMAGES` is the only stage
that does, and it already joined each tile's CellProfiler CSV into
`cell_table.parquet` (by row position -- CellProfiler's own `ObjectNumber`
numbering has no shared index with the segmentation table) before this
stage ever runs. See
[Architecture](../architecture.md#cell-images-build_cell_images-output-from-starcall-workflow).

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_images_dir` | **required** | `BUILD_CELL_IMAGES`' per-experiment output directory (holds `cell_table.parquet`, already carrying this experiment's CellProfiler columns). Injected automatically when run through the pipeline (the process stages `cell_table.parquet` into its work directory and passes `cell_images_dir=.`); set explicitly only when invoking this module's CLI directly against a `BUILD_CELL_IMAGES` output you already have. |
| `batch_stem` | **required** | This experiment's identifier, written into every row as `meta_batch`. |

`phenotyping_dir`/`wells`/`grid_size`/`segmentation_type`/`use_corrected`/
`cellprofiler_cycle`/`cellprofiler_pipeline`/`csv_schema_scan_rows`, and the
genotype column names (`barcode_col_name`/`aa_changes_col_name`/
`edit_distance_col_name`), are no longer fields on this stage -- they're `BUILD_CELL_IMAGES`-only now
(starcall-workflow-discovery concerns, including which CellProfiler CSV to
force and fold in, and which reads-table columns to rename to `meta_*`).

## Output file

Written to `output_dir`:

- `cp_features.parquet` -- one row per cell: `meta_batch`, `meta_well`,
  `meta_tile`, `meta_cell_index`, `meta_barcode`, `meta_aa_changes`,
  `meta_edit_distance`, plus every CellProfiler feature column under its
  own CellProfiler name, in the table's order.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.cp_features \
    output_dir=./out \
    cell_images_dir=/pipeline/cell_images/experiment1 \
    batch_stem=experiment1 \
    random_seed=0
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage). |

See [API Reference: cp_features](../api/cp_features.md) for full function documentation.
