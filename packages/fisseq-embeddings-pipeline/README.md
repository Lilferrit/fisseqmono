# FISSEQ Embeddings Pipeline

A Nextflow + Python workflow for scoring genetic variants against learned
**Cell-DINO** embeddings from FISSEQ (Fluorescence In-Situ Sequencing)
experiments -- the embedding-space sibling of
[`fisseq-data-pipeline`](../fisseq-data-pipeline),
which does the same analysis on hand-engineered CellProfiler features
instead. Downstream of the embeddings, both pipelines run the same stages, from
`fisseq-common`.

## Overview

Each cell carries a genetic variant label; the pipeline embeds every cell
with a pretrained Cell-DINO vision transformer and measures how each
variant's cell population differs from wildtype (WT) controls in that
learned embedding space, per experiment:

```text
starcall-workflow -> Cell Images (per-tile WebDataset shards) -> Cell Embeddings (Cell-DINO)
    -> QC_FILTER -> NORMALIZE (z-scored against wildtype)
    -> OVWT_BATCHWISE                  -> per-variant distinguishability scores
    -> bootstrap feature selection     -> per-method aggregates, blocklist, output.parquet
```

Pooling across experiments (Global Variant Embeddings, global distinguish-ability scores) is
done by `fisseqborn-global` in the fisseqborn package, from these per-experiment outputs.

See **[the documentation site](https://lilferrit.github.io/fisseqmono/embeddings-pipeline/)**
for the full design (architecture decisions, data contracts, per-stage
usage, Nextflow orchestration, running on a cluster, output layout).

## Quick start

Install the environment ([uv](https://docs.astral.sh/uv/)-managed):

```bash
git clone https://github.com/Lilferrit/fisseqmono.git
cd fisseqmono
uv sync
```

Run the full pipeline end to end with
[Nextflow](https://www.nextflow.io/) (needs Java 17+; not installed by
`uv sync`):

```bash
cd packages/fisseq-embeddings-pipeline
nextflow run . -params-file params.yaml \
    --pipeline_dir /path/to/experiment \
    --cell_dino_checkpoint /path/to/checkpoint.pth
```

That runs every task in the published container image under Docker; add
`-profile apptainer` for Apptainer, or `-profile local` to run against
this repo's own venv. On a cluster, pass your own executor settings with
`-c site.config`, and optionally a snakemake profile for the nested
starcall-workflow run with `--starcall_profile` -- nothing
scheduler-specific ships in this repo.

See [Installation](https://lilferrit.github.io/fisseqmono/embeddings-pipeline/installation/)
and [Quickstart](https://lilferrit.github.io/fisseqmono/embeddings-pipeline/quickstart/)
for the full walkthrough, including how to lay out an experiment's inputs
and where to get a Cell-DINO checkpoint.

The workspace's `.devcontainer/` definition is for developing with Claude Code in an
isolated sandbox.

## Documentation

Full documentation is published at
[lilferrit.github.io/fisseqmono/embeddings-pipeline](https://lilferrit.github.io/fisseqmono/embeddings-pipeline/),
built from `docs/embeddings-pipeline/` via [mkdocs](https://www.mkdocs.org/). To build it
locally:

```bash
uv run --group docs mkdocs serve
```

## License

[MIT](LICENSE.txt)
