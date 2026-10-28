# Published shards: `cell_images/<batch>/` is the whole dataset

**Commit:** BUILD_CELL_IMAGES publishes the well shards themselves under
`cell_images/<batch>/shards/`, and `shards.parquet` names them relative to its own directory.

**Why.** The shards used to stay only in starcall's tree, and `shards.parquet` held their absolute
paths there, so the dataset in `pipeline_dir` couldn't be read, moved or shared without
`phenotyping_dir`. They are now hard links to the shards snakemake keeps in starcall's tree (its
cache), so publishing them takes no space twice on the same filesystem.

## Code changes

1. `build_cell_images_table.link_shards` hard-links each shard into `shards_dir` (`shards`,
   new config field), copying instead when a hard link can't be made (another filesystem), and
   `build_shards_table` records `shard_tar` as `shards/<name>`.
2. BUILD_CELL_IMAGES emits and publishes `shards/*.tar.gz` (publishDir pattern
   `{cell_table.parquet,shards.parquet,shards/*}`) in place of its `env("phenotyping_dir")`
   output. EMBED_CELLS stages them under `shards/` next to `shards.parquet` instead of binding
   `phenotyping_dir`, and `embed.shard_urls` resolves a relative `shard_tar` against
   `shards.parquet`'s directory. `build_cell_images_prepare` no longer writes
   `resolved_dirs.env`, which only existed for that bind path.
3. `EmbeddingsPipelineLayout.shard_dir(batch)`: `cell_images/<batch>/shards`.

## Output change

| File | Change |
|---|---|
| `cell_images/<batch>/shards.parquet` | `shard_tar` is `shards/well_1_shard_000000.tar.gz` (was `<RUN>/pipeline/phenotyping/well1_grid1/cells_raw_shards_<window>_all/well_1_shard_000000.tar.gz`) |
| `cell_images/<batch>/shards/well_*_shard_*.tar.gz` | new (not a parquet, so not in `tests/reference/`) |

Every other published file is byte-identical in the `embeddings` and `embeddings_two` scenarios,
including `embeddings.parquet`: EMBED_CELLS reads the same shard files, by another path.
