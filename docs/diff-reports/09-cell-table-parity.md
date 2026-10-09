# Cell table in the data pipeline's shape; self-describing WebDataset samples

**Commit:** the embeddings pipeline's `cell_images/<batch>/cell_table.parquet` becomes a per-cell
table shaped like the data pipeline's input: `meta_*` columns, then the CellProfiler features
under their own names. Each WebDataset sample's `meta.json` carries the same `meta_*` values.

**Why.** Parity with the data pipeline. There is a separate CellProfiler table, and a
WebDataset whose samples carry their own metadata: the variant and its class, the fields QC
filters on, and a key that joins them to the table. Before this change, `cell_table.parquet` was a
raw join of starcall's per-tile CSVs under starcall's column names, with the CellProfiler columns
prefixed `cp_`. `meta.json` held only the cell's location, and `EMBED_CELLS` joined
`metadata.parquet` back on to get the rest.

## Code changes

1. `build_cell_images_table.tile_cell_meta` builds each tile's `meta_*` rows
   (`utils.cell_table.CELL_META_SCHEMA`):
   - `meta_well`, `meta_tile`, `meta_cell_index`: the key, unique within an experiment;
   - `meta_barcode`, `meta_aa_changes`, `meta_edit_distance`;
   - `meta_variant_class`: `fisseq_common.variant`'s class, null where the variant is null.

   `cell_table.parquet` and `tile_shard.py`'s `meta.json` both come from this function. Neither
   carries `meta_batch`: the shards are cached in starcall's tree, and the batch name is a
   run-level one.
2. `make_cell_shard` takes the tile's reads CSV as an input. `barcode_col_name`,
   `aa_changes_col_name` and `edit_distance_col_name` move from BUILD_CELL_METADATA and
   BUILD_CP_FEATURES to BUILD_CELL_IMAGES. The build-table phase reads them back from the nested
   snakemake's config, so the table and the shards rename the same columns.
3. BUILD_CELL_METADATA and BUILD_CP_FEATURES add `meta_batch` and keep the same seven `meta_*`
   columns as before. `meta_variant_class` is left out, since QC and every later stage derive
   the class themselves. BUILD_CP_FEATURES no longer strips a `cp_` prefix.
4. EMBED_CELLS takes its `meta_*` columns from `meta.json`, plus `meta_batch` from the new
   `batch_stem` field. `metadata_path` and `attach_metadata` are removed. It no longer waits for
   BUILD_CELL_METADATA.

## Output change

| File | Change |
|---|---|
| `cell_images/<batch>/cell_table.parquet` | was `tile_cell_index`, `bbox_x1/y1/x2/y2`, `orig_index`, `mask8`, `crop_index`, `editDistance`, `upBarcode`, `aaChanges`, `cp_Cells_AreaShape_Area`, `well`, `tile`; now `meta_well`, `meta_tile`, `meta_cell_index`, `meta_barcode`, `meta_aa_changes`, `meta_edit_distance`, `meta_variant_class`, `Cells_AreaShape_Area` |

The rows are the same, in the same order. The genotype values equal `metadata.parquet`'s, and
`Cells_AreaShape_Area` equals the old `cp_Cells_AreaShape_Area`. On the fixture, the variant
classes are 12 Synonymous, 6 Single Missense and 6 WT. Every other published file is
byte-identical in the `embeddings` and `embeddings_two` scenarios, including `metadata.parquet`,
`cp_features.parquet` and `embeddings.parquet`.
