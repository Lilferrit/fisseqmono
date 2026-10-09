# Cell Images, Phase 3: Build Table (`BUILD_CELL_IMAGES`)

`python -m fisseq_embeddings_pipeline.build_cell_images_table` is the
third of `BUILD_CELL_IMAGES`' three phases (Nextflow process
`BUILD_CELL_IMAGES`), run after that process's nested starcall `snakemake`
invocation (phase 2 -- the one step needing the `ops` conda env baked into
the root `Dockerfile`) has materialized every tile's
segmentation/reads/CellProfiler CSVs and every well's WebDataset shards.

It reads `manifest` (written in phase 2 by `snakemake/Snakefile`'s
`fisseq_tiles_manifest` rule -- see
[`build_cell_images_prepare`](build_cell_images_prepare.md#tiles-and-grid-size-the-fisseq_tiles_manifest-rule)) and
builds the experiment's cell table in the data pipeline's shape: one
`output` (`cell_table.parquet`) covering the whole experiment -- the ONE
complete, self-sufficient cell table `BUILD_CELL_METADATA`/`BUILD_CP_FEATURES`
need; neither reads `starcall-workflow`'s tree directly. Per tile,
`tile_cell_meta` joins the segmentation-side `{segtype}.csv` to
`sequencing_dir`'s `{segtype}_reads{params}.csv` (by index value) and
renames the reads table's genotype columns to `meta_barcode`/
`meta_aa_changes`/`meta_edit_distance`; `build_tile_table` then appends,
if `cp_features`, the tile's CellProfiler CSV (by row position, under
CellProfiler's own column names). The genotype column names come from
`snakemake_config` (the `fisseq_*_col` keys phase 1 writes from the
experiment's `barcode_col_name`/`aa_changes_col_name`/
`edit_distance_col_name`), and every shard sample's `meta.json` is built
by the same `tile_cell_meta` on the same CSVs (see
[Cell Shards](tile_shard.md#output)), so the table and the shards can't
disagree on a cell's metadata. It also reads
`shards_manifest` (written by the same rule) and writes `shards_output`
(`shards.parquet`): one row per shard `EMBED_CELLS` reads. The segmentation CSV is read
with `read_segmentation_table`, which [Cell Shards](tile_shard.md) shares,
so the row order here and the mask label a shard was cut by can't drift. See
[Architecture](../architecture.md#cell-images-build_cell_images-output-from-starcall-workflow)
for the full data-contract rationale.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `manifest` | `"tiles_manifest.csv"` | Tile manifest CSV (under `output_dir`), written by the `fisseq_tiles_manifest` rule. |
| `shards_manifest` | `"shards_manifest.csv"` | Shard manifest CSV (under `output_dir`), written by the same rule. |
| `output` | `"cell_table.parquet"` | Output parquet filename (under `output_dir`). |
| `shards_output` | `"shards.parquet"` | Shard table filename (under `output_dir`). |
| `snakemake_config` | `"snakemake_config.yaml"` | The nested snakemake's `--configfile` (under `output_dir`), written by phase 1; its `fisseq_barcode_col`/`fisseq_aa_changes_col`/`fisseq_edit_distance_col` keys name the reads tables' genotype columns. |

## Output files

Written to `output_dir`:

- `cell_table.parquet` -- one row per cell across every tile in the
  manifest, in each tile's segmentation-CSV row order. First the
  `CELL_META_SCHEMA` columns (`utils/cell_table.py`): `meta_well`,
  `meta_tile`, `meta_cell_index` (the segmentation CSV's own index),
  `meta_barcode`, `meta_aa_changes`, `meta_edit_distance` and
  `meta_variant_class` (`fisseq_common.variant`'s class of
  `meta_aa_changes`, null where that is null). Then, only for
  `cp_features: true` experiments, every CellProfiler column under its own
  name (e.g. `Cells_AreaShape_Area`). Nothing else: no `bbox_*`,
  `crop_index`, raw genotype or other aux-table columns, and no
  `meta_batch` (`BUILD_CELL_METADATA`/`BUILD_CP_FEATURES` add it).
  Concatenated `how="diagonal_relaxed"` across tiles. A reads table whose
  index set differs from the segmentation table's, or that lacks one of the
  three genotype columns, raises, as does a CellProfiler CSV whose row count
  differs.
- `shards.parquet` -- `well`, `shard_tar`: one row per shard, in order,
  the path copied from the shard manifest (a real path under
  `phenotyping_dir`; nothing is copied or linked).

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.build_cell_images_table \
    output_dir=./out \
    manifest=tiles_manifest.csv \
    shards_manifest=shards_manifest.csv \
    output=cell_table.parquet \
    shards_output=shards.parquet \
    snakemake_config=snakemake_config.yaml
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage; not consumed by this stage's own logic. |

See [API Reference: build_cell_images_table](../api/build_cell_images_table.md)
for full function documentation.
