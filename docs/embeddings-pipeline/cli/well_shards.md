# Well Shards (`make_well_shards`)

`python -m fisseq_embeddings_pipeline.well_shards` packs **one well's**
tile shards (cut by [`make_cell_shard`](tile_shard.md)) into the gzipped
**WebDataset** shards `EMBED_CELLS` streams from:
`well_{n}_shard_{k:06}.tar.gz`, `shard_size` cells each.

Like `make_cell_shard`, it is not a Nextflow process. It's the body of
`rule make_well_shards` in `snakemake/Snakefile`, run inside
`BUILD_CELL_IMAGES`' nested snakemake once per well. `BUILD_CELL_IMAGES`
asks for every well's shard directory, first on their own (the
`fisseq_shards` pass), then as inputs of the manifest rule. See
[Architecture](../architecture.md) decision 17.

## The rule

```text
input:   {phenotyping_dir}{well}_grid{N}/tile{x}x{y}y/{segmentation_type}_{raw|corrected}_shard_{window}.tar  (every tile of the grid)
output:  {phenotyping_dir}{well}_grid{N}/{segmentation_type}_{raw|corrected}_shards_{window}_{shard_size|all}/  (directory)
log:     the same path, .log
```

How many shards a well has depends on its cell count, so the output is a
directory. `shard_size` is part of its name (`all` when it's `null`), like
`raw`/`corrected` and `window`: changing any of them asks for a different
directory instead of reusing stale shards. The tile shards are `temp()`,
so snakemake deletes them once the well is packed. A rerun with a new
`shard_size` cuts every tile again.

The tile shards reach the module through a file (`tile_tars_file`), one
path per line, because a 20 x 20 grid has 400 of them.

## Packing

Samples are copied tar member by tar member, never decoded: tile by tile
in grid order, then in each tile's own order. A sample is the run of
consecutive members sharing a key (the name up to its first `.`,
WebDataset's rule). Each shard holds `shard_size` samples except the last,
which holds the rest; a sample never straddles two shards. With
`shard_size` unset the whole well is one shard. A well with no cells still
gets one valid, empty shard.

`n` is the well's number (`well3` -> `well_3_shard_000000.tar.gz`); a
well not named `well<n>` keeps its whole name. The six-digit shard number
keeps a well's shards in order when sorted by name, and fits WebDataset's
brace patterns (`well_3_shard_{000000..000041}.tar.gz`).

Shards are gzipped at level 6 (gzip's own default; `tarfile`'s 9 is much
slower for little gain on image data). WebDataset opens `.tar.gz` shards
as they are.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `tile_tars_file` | **required** | Text file listing the well's tile shards, one path per line, in the order their cells are written. |
| `well` | **required** | The well's name, which names its shards. |
| `shard_size` | `null` | Cells per shard. `null` writes the whole well into one shard. From the experiment's `shard_size` (falling back to `params.yaml`'s global default; see [Configuration](../configuration.md)). |
| `shard_dir` | **required** | Directory to write the shards into. |

## Output

`shard_dir/well_{n}_shard_{k:06}.tar.gz`, `k` from 0. Each sample is
exactly what `make_cell_shard` wrote: keyed
`"{well}_{tile}_{tile_cell_index}"`, carrying `crop.npy`, `mask.npy` and
`meta.json`. `BUILD_CELL_IMAGES`' `shards.parquet` lists every shard
(`well`, `shard_tar`). The shards are read in place and never copied into
`pipeline_dir`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.well_shards \
    output_dir=/tmp/log \
    tile_tars_file=/tmp/log/tile_tars.txt \
    well=well1 shard_size=2000 \
    shard_dir=phenotyping/well1_grid4/cells_raw_shards_224_2000
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage. |

See [API Reference: well_shards](../api/well_shards.md) for full function documentation.
