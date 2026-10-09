# Cell Shards (`make_cell_shard`)

`python -m fisseq_embeddings_pipeline.tile_shard` crops every cell of
**one tile** out of the tile's stitched phenotype image and writes them into that tile's **WebDataset** shard (a `.tar`, one sample per
cell, unfiltered). `make_well_shards` then packs each well's tile shards
into the gzipped shards `EMBED_CELLS` streams from -- see
[Well Shards](well_shards.md).

It is not a Nextflow process. It's the crop step of `rule make_cell_shard`,
which this repo adds on top of starcall's own Snakefile
(`snakemake/Snakefile`), so it runs inside `BUILD_CELL_IMAGES`' nested
snakemake, once per tile, as phase 2 of that stage. `BUILD_CELL_IMAGES`
requests every well's shards, and so, through `make_well_shards`, every
tile's. As a result, cropping fans out one job per tile under a
`starcall_profile`, and snakemake's own mtime check caches each well's
shards, so a well already packed isn't cut again. See
[Architecture](../architecture.md) decisions 17 and 24.

Building the shards (and running `EMBED_CELLS` over them) is deliberately
decoupled from `QC_FILTER`. QC thresholds get tuned and re-run often. Keeping
embedding as a separate, unconditional branch means changing a QC threshold
never re-triggers the expensive GPU embedding pass. You pay for embedding
every cell once, up front.

## The rule

```text
input:   each phenotype cycle's input images (starcall's find_input_tiles;
           corrected_tiles.tif for corrected)
         {stitching_dir}{well}/cycle{c}/composite.json, for each phenotype cycle
         {stitching_dir}{well}_grid{N}/grid_composite.json
         the segmentation-grid masks (starcall's get_grid_filenames)
         {segmentation_dir}{well}_grid{S}/grid_composite.json
         {phenotyping_dir}{well}_grid{N}/tile{x}x{y}y/{segmentation_type}.csv
         {sequencing_dir}{well}_grid{N}/tile{x}x{y}y/{segmentation_type}_reads{sequencing_reads_params}.csv
params:  the reads table's genotype column names (fisseq_barcode_col,
         fisseq_aa_changes_col, fisseq_edit_distance_col)
output:  {phenotyping_dir}{well}_grid{N}/tile{x}x{y}y/{segmentation_type}_{raw|corrected}_shard_{window}.tar  (temp)
log:     the same path, .log
```

`raw` vs `corrected` (from the experiment's `use_corrected`) and `window`
are both part of the shard's filename. Changing either requests a
different shard instead of reusing a stale one. The genotype column names
are `params:`, not part of the filename, and snakemake reruns on mtime, not
params: changing an experiment's `barcode_col_name`/`aa_changes_col_name`/
`edit_distance_col_name` doesn't recut shards that already exist. Delete
the well's shard directory to have them recut with the new names.

The reads table is an input so each sample's `meta.json` can carry the
cell's genotype. It doesn't drag the reads chain into the shards pass: a
dry run against starcall's real Snakefile showed the stitched tile image
and mask aren't among its ancestors.

The rule stitches the tile's image and mask itself, in memory, with
starcall's own `stitch_well_section` and `stitch_segmentation_section`,
called with exactly the arguments starcall's `rule stitch_tile_pt` and
`rule stitch_tile_segmentation` pass. Its inputs are all files starcall
keeps, so starcall's `temp()` whole-tile `raw_pt.tif`/`cells_mask.tif`
are never requested, and recutting a shard regenerates nothing
CellProfiler reads. The exception is `use_corrected`: `corrected_tiles.tif`
(per cycle, whole well) is itself `temp()`, so a missing shard regenerates
it. Its `mem_mb` is starcall's own requests for those two rules plus the
crop's own.

It's a `run:` rule, in the `ops` env's 3.10 snakemake interpreter,
because the stitching functions live in starcall's Snakefile and need
that env (`constitch`, `nd2`). It writes the stitched image and mask as
tifs, the way starcall writes them, to a job-local `tempfile.mkdtemp()`
directory (under `$TMPDIR`, never seen by snakemake), then runs this
module in the pipeline's own Python 3.13 venv (`config['fisseq_python']`,
which `BUILD_CELL_IMAGES` sets to its own `$(command -v python)`) with
`image_tif`/`mask_tif` pointing there. Every child job re-enters the same
image, so that path is valid on a cluster node too. Hydra's run directory
and this module's own log file go to the same directory, so the job's
working directory (the experiment directory) doesn't collect a Hydra
`outputs/` tree per tile; it is removed when the job ends. The console
log lands in the rule's `log:` file. The shard itself is `temp()`,
deleted once `make_well_shards` has packed it.

## Cropping

`write_tile_shard` reads the tile's segmentation table with
`build_cell_images_table.read_segmentation_table`, the same reader
`BUILD_CELL_IMAGES`' table phase uses, so the two can't disagree on which
row is which cell. Each row has a `tile_cell_index` (the CSV's own index)
and a `crop_index` (its 0-based on-disk row position). The module then reads
the tile's image (starcall's `(cycles, channels, H, W)` flattened to
`(C, H, W)`) and the label mask (`(H, W)`) once, and cuts every cell out
with `crop_cell`:

- the crop is `window` x `window`, centred on the bbox midpoint
  (`((bbox_x1 + bbox_x2) // 2, (bbox_y1 + bbox_y2) // 2)`, with `bbox_x*`
  on image axis 0 and `bbox_y*` on axis 1, starcall's own convention);
- where the window runs off the tile it is zero-padded;
- the mask crop is `mask == crop_index + 1` (row *i* of the tile's cell
  table is mask label *i+1*), stored as uint8, so neighbouring cells
  inside the window are masked out.

This is starcall-workflow's own `make_cell_images` crop with the centre
fixed: upstream reads `xpos`/`ypos` columns its cell table doesn't have.
See [Architecture](../architecture.md) decision 17. A tile with no cells
still gets a valid, empty tar (snakemake needs the output to exist), and
its image and mask are never opened. A mask whose shape doesn't match its
image raises.

## Config fields

Extends the [common config fields](#common-config-fields) below.
`output_dir` holds only this run's log; the shard itself goes to
`output_tar`.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `image_tif` | **required** | The tile's stitched phenotype image, `(cycles, channels, H, W)` (written job-locally by `make_cell_shard`). |
| `mask_tif` | **required** | The tile's stitched `(H, W)` segmentation label mask (likewise). |
| `segmentation_csv` | **required** | The tile's `{segmentation_type}.csv`. |
| `reads_csv` | **required** | The tile's `{segmentation_type}_reads{sequencing_reads_params}.csv` (`sequencing_dir`), the source of each cell's genotype. |
| `barcode_col_name` | `"upBarcode"` | The reads table's barcode column. |
| `aa_changes_col_name` | `"aaChanges"` | The reads table's amino-acid-changes column. |
| `edit_distance_col_name` | `"editDistance"` | The reads table's edit-distance column. |
| `well`, `tile` | **required** | This tile's identifiers, written into each sample's key and `meta.json`. |
| `window` | **required** | Side length, in pixels, of each cell's square crop. Must match the loaded Cell-DINO checkpoint's expected input (`cell_dino_crop_size`). Comes from the shard target's filename, which `BUILD_CELL_IMAGES` names from the experiment's `window` (falling back to `params.yaml`'s global default; see [Configuration](../configuration.md)). |
| `output_tar` | **required** | Path of the shard to write. |

## Output

One `.tar` per tile. Each sample is keyed `"{well}_{tile}_{tile_cell_index}"`
and carries:

- `crop.npy`: `(num_phenotyping_cycles × num_channels, window, window)`,
  in the image's dtype;
- `mask.npy`: `(window, window)` uint8 foreground mask;
- `meta.json`: the cell's `CELL_META_SCHEMA` row (`utils/cell_table.py`)
  -- `meta_well`, `meta_tile`, `meta_cell_index`, `meta_barcode`,
  `meta_aa_changes`, `meta_edit_distance`, `meta_variant_class`.

`meta.json` is built by `build_cell_images_table.tile_cell_meta`, the same
function, on the same segmentation and reads CSVs, that builds
`cell_table.parquet`'s leading columns, so the two can't disagree; they
join on (`meta_well`, `meta_tile`, `meta_cell_index`). It carries
everything `EMBED_CELLS` needs except `meta_batch`, which is left out on
purpose: the shards are cached in starcall's tree, and `batch_stem` is a
run-level name that would go stale there when it changes. `EMBED_CELLS`
adds it from its own `batch_stem` (see [Cell Embeddings](embed.md)).

At the defaults (4 uint16 channels, `window` 224) a sample is about
440 KB (`crop.npy` ≈ 392 KB, `mask.npy` ≈ 49 KB, plus `meta.json` and tar
headers), so a tile's shard is roughly 440 KB × its cell count; channel
count (`num_phenotyping_cycles × num_channels`), dtype and `window` scale
it.

`make_well_shards` copies these samples unchanged into the well's
gzipped shards, which is what `BUILD_CELL_IMAGES`' `shards.parquet`
lists.

## Example

Run standalone, `image_tif`/`mask_tif` can be starcall's own whole-tile
files, which have the same layout, where they exist:

```bash
uv run python -m fisseq_embeddings_pipeline.tile_shard \
    output_dir=/tmp/log \
    image_tif=phenotyping/well1_grid4/tile0x0y/raw_pt.tif \
    mask_tif=phenotyping/well1_grid4/tile0x0y/cells_mask.tif \
    segmentation_csv=phenotyping/well1_grid4/tile0x0y/cells.csv \
    reads_csv=sequencing/well1_grid4/tile0x0y/cells_reads.csv \
    well=well1 tile=tile0x0y window=224 \
    output_tar=phenotyping/well1_grid4/tile0x0y/cells_raw_shard_224.tar
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage. |

See [API Reference: tile_shard](../api/tile_shard.md) for full function documentation.
