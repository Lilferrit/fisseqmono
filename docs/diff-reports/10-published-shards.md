# Published shards: `cell_images/<batch>/` is the whole dataset

**Commit:** BUILD_CELL_IMAGES publishes the well shards themselves under
`cell_images/<batch>/shards/`, and EMBED_CELLS reads every shard in that directory.
`shards.parquet` is gone.

**Why.** The shards used to stay only in starcall's tree, and `shards.parquet` held their absolute
paths there, so the dataset in `pipeline_dir` couldn't be read, moved or shared without
`phenotyping_dir`. They are now hard links to the shards snakemake keeps in starcall's tree (its
cache), so publishing them takes no space twice on the same filesystem. With the shards next to
the cell table, `shards.parquet` would only list the directory beside it (its `well` column is in
each file name), so it's dropped.

## Code changes

1. `build_cell_images_table.link_shards` hard-links each shard into `shards_dir` (`shards`,
   new config field), copying instead when a hard link can't be made (another filesystem).
   `build_shards_table` and the `shards_output` field are removed.
2. BUILD_CELL_IMAGES emits and publishes `cell_table.parquet` and `shards/*.tar.gz`
   (publishDir pattern `{cell_table.parquet,shards/*}`) in place of `shards.parquet` and its
   `env("phenotyping_dir")` output. EMBED_CELLS stages the shards under `shards/` instead of
   binding `phenotyping_dir`. `EmbedCellsConfig.shards_path` is replaced by `shards_dir`:
   `embed.shard_urls` reads every `*.tar.gz` in it, in name order, and raises if there are
   none. `build_cell_images_prepare` no longer writes `resolved_dirs.env`, which only existed
   for that bind path.
3. `EmbeddingsPipelineLayout.shard_dir(batch)`: `cell_images/<batch>/shards`.
   `EmbeddingsPipelineLayout.shards(batch)` is removed.

## Output change

| File | Change |
|---|---|
| `cell_images/<batch>/shards.parquet` | removed |
| `cell_images/<batch>/shards/well_*_shard_*.tar.gz` | new (not a parquet, so not in `tests/reference/`) |

Every other published file is byte-identical in the `embeddings` and `embeddings_two` scenarios,
including `embeddings.parquet`: EMBED_CELLS reads the same shard files, by another path. Shards
are read in file-name order rather than manifest order, which only changes `embeddings.parquet`'s
row order for an experiment with 10 or more wells (`well_10` sorts before `well_2`); every stage
downstream joins on keys.
