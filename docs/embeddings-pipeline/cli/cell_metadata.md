# Cell Metadata (`BUILD_CELL_METADATA`)

`python -m fisseq_embeddings_pipeline.cell_metadata` (Nextflow process `BUILD_CELL_METADATA`) projects `BUILD_CELL_IMAGES`' `cell_table.parquet`
down to the seven `meta_*` columns `QC_FILTER` reads, as one
per-experiment `metadata.parquet`. No images, no feature columns, no
`starcall-workflow` tree access.

This stage exists to make `QC_FILTER` the point where the cellDINO and
CellProfiler tracks fan out, off a cheap table projection rather than the
image-reading cellDINO track. Before it, QC's input was the (since
removed) `BUILD_DATASET` stage's own `metadata.parquet`, written inside
its WebDataset shard-writing loop -- which made the expensive dataset
build a hard dependency of the CellProfiler track too, since
`NORMALIZE_CP_FEATURES` consumes the same QC output. See
[Nextflow Workflow: Track independence](../nextflow.md#track-independence).

`cell_table.parquet` already carries canonical `meta_*` names
(`BUILD_CELL_IMAGES` renames the reads tables' genotype columns), so the
projection only adds `meta_batch` -- a run-level name neither the table nor
the shards carry -- and leaves out `meta_variant_class`, which `QC_FILTER`
and every stage after it derive from `meta_aa_changes` themselves.
`EMBED_CELLS` builds the same seven columns from each shard sample's
`meta.json`, which `BUILD_CELL_IMAGES` writes from the same per-tile CSVs by
the same function (`build_cell_images_table.tile_cell_meta`), so a cell's
`meta_*` values are identical in QC's input and in `embeddings.parquet`.
The projection (`utils.cell_table.cell_metadata_exprs`) is shared with
`BUILD_CP_FEATURES` and `EMBED_CELLS`, so the three can't drift.

`QC_FILTER` ([shared](../../common/stages.md#qcfilter)) doesn't read
`cell_table.parquet` directly: the table has no `meta_batch`, the first of
the pipeline's `join_keys`, and for a `cp_features: true` experiment its
CellProfiler columns look like the feature columns `filter_columns` keeps,
so they'd be carried into `filtered_cells.parquet`.

Structurally this is the analog of `fisseq-data-pipeline`'s own `INPUT`
stage: the cheap read-and-normalize step producing the one per-batch
parquet `QC_FILTER` consumes.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `cell_table` | **required** | Path to `BUILD_CELL_IMAGES`' `cell_table.parquet`. The file itself, not its directory -- the process takes `BUILD_CELL_IMAGES`' `cell_table.parquet` output as a staged `path` input, so no host path needs binding. |
| `batch_stem` | **required** | This experiment's identifier, written into every row as `meta_batch`. |

The reads tables' genotype column names (`barcode_col_name`/
`aa_changes_col_name`/`edit_distance_col_name`) are `BUILD_CELL_IMAGES`
settings now: it renames those columns to `meta_*` in both
`cell_table.parquet` and the shards, so this stage reads fixed names.

## Output file

Written to `output_dir`:

- `metadata.parquet` -- one row per cell, exactly seven columns:
  `meta_batch`, `meta_well`, `meta_tile`, `meta_cell_index`,
  `meta_barcode`, `meta_aa_changes`, `meta_edit_distance`
  (`CELL_METADATA_SCHEMA`). An empty table gives an empty frame with that
  schema.

Note this covers *every* row of `cell_table.parquet`, where the removed
`BUILD_DATASET` stage's own `metadata.parquet` only ever held cells that
made it into a shard. `filtered_cells.parquet` can therefore cover strictly more
cells than it used to; every downstream consumer inner-joins it back on
the pipeline's `join_keys` (`meta_batch`, `meta_well`, `meta_tile`, `meta_cell_index`), so
the extra rows drop out where they don't
apply.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.cell_metadata \
    output_dir=./out \
    cell_table=/pipeline/cell_images/experiment1/cell_table.parquet \
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

See [API Reference: cell_metadata](../api/cell_metadata.md) for full function documentation.
