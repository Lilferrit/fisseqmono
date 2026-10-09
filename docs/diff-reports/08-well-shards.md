# Per-well WebDataset shards: `tiles.parquet` becomes `shards.parquet`

**Commit:** the embeddings pipeline writes one set of gzipped WebDataset shards per well
(`well_{n}_shard_{k}.tar.gz`, `shard_size` cells each) in place of one uncompressed shard per
tile.

**Why.** A well cut into a 20 x 20 grid gave 400 small shards, each named by its tile, and every
tile's shard was kept next to the starcall outputs. The per-tile shards are now `temp()`
intermediates. A new `make_well_shards` rule packs a well's tile shards into shards of a chosen
size, and those are what the pipeline keeps and EMBED_CELLS reads.

## Code changes

1. `snakemake/Snakefile`: `make_cell_shard`'s output is `temp()`. The new `make_well_shards`
   rule (`fisseq_embeddings_pipeline.well_shards`) writes
   `{phenotyping_dir}/{well}_grid{N}/{segmentation_type}_{raw|corrected}_shards_{window}_{shard_size|all}/`.
   The `fisseq_shards` pass asks for those directories. The manifest rule also writes
   `shards_manifest.csv`, one row per shard.
2. New param `shard_size` (global default `null` = one shard per well; can be set per
   experiment), routed to BUILD_CELL_IMAGES like `window`.
3. BUILD_CELL_IMAGES publishes `cell_images/<batch>/shards.parquet` (`well`, `shard_tar`) in place
   of `tiles.parquet` (`well`, `tile`, `shard_tar`). `EmbedCellsConfig.tiles_path` is renamed
   `shards_path`, and `EmbeddingsPipelineLayout.tiles` is renamed `shards`.

## Output change

| File | Change |
|---|---|
| `cell_images/<batch>/tiles.parquet` | removed |
| `cell_images/<batch>/shards.parquet` | new: `well`, `shard_tar` (one row per shard; a fixture well has one shard, `.../cells_raw_shards_<window>_all/well_1_shard_000000.tar.gz`) |

Every other published file is byte-identical in the `embeddings` and `embeddings_two` scenarios,
including `embeddings.parquet`. The samples are copied into the well shards unchanged, in the
same order.
