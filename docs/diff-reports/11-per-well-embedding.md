# Per-well embedding: one WebDataset and one EMBED_CELLS task per well

**Commit:** EMBED_CELLS runs once per well, and a new CONCAT_EMBEDDINGS stage stacks an
experiment's wells into its `embeddings.parquet`. Each well's shards are published as their own
WebDataset, `cell_images/<batch>/shards/<well>/`. Shards are sized by their size on disk
(`shard_max_size`, default `1GB`) instead of a cell count (`shard_size`, now ignored with a
warning). The `embeddings_two` reference now also pins the ComBat outputs.

**Why.** A GPU task embedding a whole experiment that gets killed (walltime, preemption)
loses the whole experiment. Per well, it loses one well, and `-resume` reruns just that one. A
cell count made shard sizes depend on crop size and channel count; a size on disk doesn't. 1 GB
is WebDataset's own `ShardWriter` default (ImageNet WebDatasets use ~100 MB-1 GB). ComBat was
already wired into the embeddings pipeline (#27), but no reference covered it.

## Code changes

1. `well_shards.write_well_shards(..., shard_max_bytes, ...)`: a shard closes after the sample
   that takes its compressed size on disk to `shard_max_bytes` (samples still never straddle
   shards). The snakemake shard directory is named `..._<window>_<bytes>B` (`..._all` for
   `null`); the `B` keeps it apart from the old cell-count directories.
2. `config.experiments`: `shard_max_size` (global and per experiment; `parse_size` reads
   `1GB`/`512MiB`/bytes) reaches BUILD_CELL_IMAGES as `shard_max_bytes`. `shard_size` is in
   `REMOVED_PARAMS`/`REMOVED_EXPERIMENT_FIELDS`.
3. `build_cell_images_table.link_shards` links into `shards/<well>/`; BUILD_CELL_IMAGES emits
   `shards/*/*.tar.gz` and publishes `shards/**`. `EmbeddingsPipelineLayout.well_shard_dir`.
4. `workflows/embeddings.nf`: BUILD_CELL_IMAGES' shards are grouped by well into one EMBED_CELLS
   task each (`EmbedCellsConfig.output`, `embeddings_<well>.parquet`, not published), then
   `groupKey(batch_stem, n_wells)` + `groupTuple` hand an experiment's tables to
   CONCAT_EMBEDDINGS (`concat_embeddings.py`) once all of its wells are done. An experiment with
   a failed well gets no `embeddings.parquet`.

## Output change

| File | Change |
|---|---|
| `cell_images/<batch>/shards/well_*_shard_*.tar.gz` | moved to `cell_images/<batch>/shards/<well>/` (not a parquet, so not in `tests/reference/`) |
| `combat/{features,ovwt_scores}.parquet`, `combat_cp_features/{features,ovwt_scores}.parquet` | newly captured in `tests/reference/embeddings_two/` (unchanged by this commit) |

Every published parquet is identical in the `embeddings` and `embeddings_two` scenarios,
including `embeddings.parquet`: the same cells, embedded by the same model, stacked by well. Its
row order now follows the wells' natural order (`well2` before `well10`) rather than the
shards' file names; every stage downstream joins on keys, and `tests/_compare.py` sorts rows.
