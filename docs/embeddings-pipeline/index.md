# fisseq-embeddings-pipeline

A Nextflow + Python pipeline that scores genetic variants against learned
**Cell-DINO** embeddings from FISSEQ (Fluorescence In-Situ Sequencing)
experiments -- the embedding-space sibling of
[`fisseq-data-pipeline`](../data-pipeline/index.md),
which does the same analysis on hand-engineered CellProfiler features. An
optional, parallel second track (see below) runs that same
CellProfiler-feature analysis directly inside this pipeline too, against
the same cells, for direct comparison against the embedding-space results.

Each cell carries a genetic variant label; the pipeline embeds every cell
with a pretrained Cell-DINO vision transformer and measures how each
variant's cell population differs from wildtype (WT) controls in that
learned embedding space, per experiment.

```text
starcall-workflow's raw tree -> Cell Images (BUILD_CELL_IMAGES, incl. per-tile
    WebDataset shards) -> Cell Embeddings (EMBED_CELLS, Cell-DINO)
    -> QC_FILTER -> NORMALIZE (z-scored against wildtype)
    -> OVWT_BATCHWISE                         -> per-variant distinguishability scores
    -> bootstrap feature selection            -> per-method aggregates, blocklist, output.parquet
```

Downstream of `EMBED_CELLS` it runs fisseq-data-pipeline's stages: the same shared modules
(`fisseq_common.stages`), process names, parameters and publish layout, on the embedding
dimensions instead of CellProfiler features. See [Shared stages](../common/stages.md).

Pooling across experiments (Global Variant Embeddings, global distinguish-ability scores) is
done by `fisseqborn-global` in the fisseqborn package, from these per-experiment outputs.

`BUILD_CELL_IMAGES` is the only stage that reads `starcall-workflow`'s
tree or runs its snakemake (starcall's own Snakefile, at the commit the
image pins, plus this pipeline's `make_cell_shard`
rule). It has every tile's cells cropped straight into a per-tile
WebDataset shard as an ordinary snakemake job, and joins each tile's
segmentation and genotype tables into one self-sufficient
`cell_table.parquet` per experiment (see [Architecture](architecture.md)).

A variant's identity is preserved throughout by its label column
(`meta_aa_changes`). The cells are normalized against the wildtype cells; the per-variant
aggregates are z-scored against the experiment's synonymous variants (same amino acid
before/after).

### CellProfiler-feature track (optional)

Setting `cp_features: true` on one of `params.yaml`'s `experiments:`
entries opts that experiment into a second, parallel track:
`BUILD_CELL_IMAGES` additionally forces that experiment's already-computed
CellProfiler measurements (from `starcall-workflow`) to exist and folds
them into `cell_table.parquet`, `BUILD_CP_FEATURES` selects them back out,
and the same shared stages -- normalize, aggregate, OVWT, without the bootstrap feature
selection -- run again, reusing `QC_FILTER`'s existing output rather than
QC-filtering twice. See [Architecture](architecture.md) and
[Nextflow Workflow](nextflow.md#cellprofiler-feature-track).

## Where to go next

- **[Installation](installation.md)** -- environment setup, Docker image.
- **[Quickstart](quickstart.md)** -- run the pipeline end to end.
- **[Architecture](architecture.md)** -- design decisions, repository
  layout, data contracts, and the Cell-DINO inference internals.
- **[Nextflow Workflow](nextflow.md)** -- how the pipeline's stages
  are orchestrated, running on a cluster with your own profiles, and the
  output directory layout.
- **[Configuration](configuration.md)** -- `params.yaml` reference, Docker
  image versioning, and WebDataset shard sizing.
- **Stage Reference** (sidebar) -- usage, config fields, and outputs for
  each cellDINO-specific stage; the shared stages are in
  [Shared stages](../common/stages.md).

## Related packages and repositories

- [`fisseq-data-pipeline`](../data-pipeline/index.md) -- the CellProfiler-feature version of
  this same analysis; downstream of `EMBED_CELLS` both pipelines run the same stages.
- [`fisseq-common`](../common/index.md) -- the shared stages, Nextflow modules and output
  layout.
- [`fisseqborn`](../fisseqborn/index.md) -- plots and cross-experiment aggregation.
- `starcall-workflow` -- the Snakemake pipeline whose `origin/devel` branch
  produces this pipeline's raw input tree; `BUILD_CELL_IMAGES` is the only
  stage that reads it or runs it -- see
  [Architecture](architecture.md#data-contracts).
