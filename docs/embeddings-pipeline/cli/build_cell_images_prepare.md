# Cell Images, Phase 1: Prepare (`BUILD_CELL_IMAGES`)

`python -m fisseq_embeddings_pipeline.build_cell_images_prepare` is the
first of `BUILD_CELL_IMAGES`' three phases (Nextflow process
`BUILD_CELL_IMAGES`). It resolves `phenotyping_dir`/`segmentation_dir`/
`sequencing_dir` and writes what has to exist before the nested
`snakemake` (phase 2) starts:

- `resolved_dirs_out` (`resolved_dirs.env`) -- the three resolved data
  dirs as shell-sourceable `key='value'` lines. The process sources it for
  its `phenotyping_dir` output, EMBED_CELLS' bind path.
- `snakemake_config_out` (`snakemake_config.yaml`) -- phase 2's
  `--configfile`: the same three dirs (each with a trailing `/`; they
  override the project's own `config.yaml`), `fisseq_python` (the
  interpreter `make_cell_shard` and `make_well_shards` run this package
  with) and the `fisseq_*` settings of the `fisseq_shards` and
  `fisseq_tiles_manifest` rules (see below) -- among them
  `fisseq_barcode_col`/`fisseq_aa_changes_col`/`fisseq_edit_distance_col`,
  the reads tables' genotype column names, which `make_cell_shard` passes
  to each tile's shard and phase 3 reads back from this same file, so the
  shards' `meta.json` and `cell_table.parquet` rename the same columns.
- `jobscript_out` (`starcall_jobscript.sh`) -- **only** when
  `starcall_job_image` is set, i.e. the run passes a `starcall_profile`:
  the `--jobscript` template every starcall child job runs through (see
  below).

This module and phase 2's `snakemake` are the only places in the pipeline
that read `starcall-workflow`'s tree; see
[Architecture](../architecture.md#cell-images-build_cell_images-output-from-starcall-workflow).

## Tiles and grid size: the `fisseq_tiles_manifest` rule

Phase 2 runs snakemake twice: for `fisseq_shards` (every well's shards,
alone -- see [Architecture](../architecture.md) decision 17 for why), then
for `tiles_manifest.csv` in the task directory. `snakemake/Snakefile`'s
`fisseq_tiles_manifest` rule makes it:
its input function (`snakemake/fisseq_targets.py`) lists every well's
shard directory
(`{well}_grid{N}/{segmentation_type}_{raw|corrected}_shards_{window}_{shard_size|all}/`,
packed by `make_well_shards` from the tile shards `make_cell_shard` cuts
-- see [Well Shards](well_shards.md) and [Cell Shards](tile_shard.md)),
and for every tile of every well the segmentation cell table
(`{segmentation_type}.csv`) and the sequencing reads table
(`{segmentation_type}_reads{params}.csv`), plus the CellProfiler CSV if
`cp_features` is set. So snakemake builds the whole DAG from the grid, the
way starcall's own grid-merging rules do. The rule then writes the two
manifests phase 3 ([`build_cell_images_table`](build_cell_images_table.md))
reads: `tiles_manifest.csv` (`well,tile,segmentation_csv,reads_csv,
cellprofiler_csv`) and `shards_manifest.csv` (`well,shard_tar`, one row
per shard found in each well's directory). The tile shards are
deliberately *not* inputs: they are `temp()`, so snakemake deletes them
once the well is packed. `make_cell_shard` stitches each tile's image and
mask itself, so starcall's `temp()` whole-tile `raw_pt.tif`/mask are
never requested -- see [Architecture](../architecture.md) decision 17.

Every tile of the `grid_size` x `grid_size` grid is listed, in
starcall-workflow's own naming (`{well}_grid{grid_size}/tile{x:02}x{y:02}y`),
whether or not it exists yet -- the nested snakemake is what creates it.
`wells` and `grid_size` default to starcall's own: the `wells` it reads
from the project config (or detects from its input tree), and the project
config's `phenotyping_grid_size`.

## The cluster-mode jobscript

A starcall child job lands on a bare node, but its snakemake command names
the image's own `ops` interpreter, and starcall's `run:` rule bodies
execute inside it -- so it has to run inside the pipeline image.
`render_starcall_jobscript` writes a snakemake 7 `--jobscript` template
that re-executes itself once as

```text
<starcall_container_bin> exec [--nv] --bind <p:p,...> <starcall_job_image> /bin/sh -c "$(cat "$0")" "$0"
```

(guarded by a `FISSEQ_STARCALL_IN_IMAGE` environment variable), then runs
the job's `{exec_job}`. It passes its own text rather than its path: a
scheduler may keep the script in a node-local spool (SGE's
`/var/spool/<cell>/<host>/job_scripts/`) that isn't bound into the image. The binds (`jobscript_bind_paths`) are the three
resolved data dirs, `jobscript_binds` (the process passes
`starcall_workflow_dir`, the snakemake cache dir and `starcall_job_binds`), and the
current working directory, each also at its `os.path.realpath`: snakemake
`cd`s each cluster job into its workdir as `os.getcwd()` recorded it after
entering `--directory`, with symlinks resolved. snakemake fills the template with `str.format`, so a path
containing a brace is rejected up front.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `starcall_workflow_dir` | **required** | The experiment's starcall working directory (snakemake's `--directory`: its `config.yaml` and data trees -- not starcall's code, which comes from the image's pinned commit); the data dirs default under it, and its own `config.yaml`/`default-config.yaml` is consulted first. |
| `phenotyping_dir`, `segmentation_dir`, `sequencing_dir` | `null` | Explicit data dirs; `null` resolves via the project config, then `{starcall_workflow_dir}/{phenotyping,segmentation,sequencing}`. |
| `wells` | `null` | Wells to build cell images for; `null` takes starcall's own `wells`. |
| `grid_size` | `null` | Tile grid size; `null` takes the project config's `phenotyping_grid_size`. |
| `segmentation_type` | `"cells"` | Segmentation type name, threaded into every tile filename. |
| `use_corrected` | `false` | Stitch each shard's tile from starcall's background-corrected `corrected_tiles.tif` instead of the raw input images (mirroring starcall's own `get_phenotyping_pt`; fails at the pinned commit, see [Configuration](../configuration.md)); names the shard directory `..._corrected_shards_...` rather than `..._raw_shards_...`. |
| `window` | **required** | Crop size each cell is cut at, in the shard directory name. Must match `cell_dino_crop_size`. Routed from the experiment's `window` (or the global default). |
| `shard_size` | `null` | Cells per WebDataset shard, each well's counted separately; in the shard directory name (`all` for `null`). `null` gives each well one shard. Routed from the experiment's `shard_size` (or the global default). |
| `barcode_col_name` | `"upBarcode"` | The reads tables' barcode column (starcall's aux tables name it per experiment), renamed `meta_barcode` in the shards' `meta.json` and `cell_table.parquet`. Written as `fisseq_barcode_col`. |
| `aa_changes_col_name` | `"aaChanges"` | Likewise the amino-acid-changes column, renamed `meta_aa_changes` (`fisseq_aa_changes_col`). |
| `edit_distance_col_name` | `"editDistance"` | Likewise the edit-distance column, renamed `meta_edit_distance` (`fisseq_edit_distance_col`). |
| `sequencing_reads_params` | `""` | Suffix threaded into the reads CSV filename (`{segmentation_type}_reads{sequencing_reads_params}.csv`). |
| `cp_features` | `false` | Also build this experiment's CellProfiler CSV. |
| `cellprofiler_cycle` | `""` | Threaded into the CellProfiler CSV filename when `cp_features` is set. |
| `cellprofiler_pipeline` | `""` | Threaded into the CellProfiler CSV filename when `cp_features` is set. |
| `starcall_job_image` | `null` | The `.sif` each starcall child job re-enters; setting it writes `jobscript_out`. |
| `starcall_container_bin` | `"apptainer"` | Runtime the jobscript re-enters the image with. |
| `starcall_job_gpu` | `false` | Pass `--nv` in the jobscript. |
| `jobscript_binds` | `[]` | Extra host paths to bind into each child job. |
| `manifest_out` | `"tiles_manifest.csv"` | The tile manifest the `fisseq_tiles_manifest` rule writes (under `output_dir`; passed to snakemake as an absolute path). |
| `shards_manifest_out` | `"shards_manifest.csv"` | The shard manifest the same rule writes (likewise). |
| `snakemake_config_out` | `"snakemake_config.yaml"` | Output filename (under `output_dir`). |
| `jobscript_out` | `"starcall_jobscript.sh"` | Output filename (under `output_dir`). |
| `resolved_dirs_out` | `"resolved_dirs.env"` | Output filename (under `output_dir`). |

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.build_cell_images_prepare \
    output_dir=./out \
    starcall_workflow_dir=/data/experiment1 \
    'wells=[well1,well2]' \
    window=224 \
    segmentation_type=cells
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage; not consumed by this stage's own logic. |

See [API Reference: build_cell_images_prepare](../api/build_cell_images_prepare.md)
for full function documentation.
