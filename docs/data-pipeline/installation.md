# Installation

## Requirements

- **Python 3.13** (pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** for Python dependency and environment management
- **[Nextflow](https://www.nextflow.io/) ≥ 26.04** (only needed to run the full
  pipeline via `main.nf`, not for standalone Python CLI usage)
- No required environment variables

## Install

```bash
# Clone the workspace
git clone https://github.com/Lilferrit/fisseqmono.git
cd fisseqmono

# Install every package + the dev tools into .venv
uv sync

# Install pre-commit hooks (one-time)
uv run pre-commit install
```

The packages install in editable mode, so every stage is immediately runnable:
`uv run python -m fisseq_data_pipeline.input` (INPUT) and
`uv run python -m fisseq_common.stages.<stage>` (every other stage; see
[Shared stages](../common/stages.md)).

### Install with pip (no clone)

If you just need the pipeline steps importable — e.g. on a compute node or in a
container — `pip install` the package from a release tag. It pulls in
`fisseq-common[stages]`, which holds the shared stages:

```bash
pip install "fisseq-data-pipeline @ git+https://github.com/Lilferrit/fisseqmono@v2.0.0#subdirectory=packages/fisseq-data-pipeline"
```

Running the Nextflow workflow itself needs a checkout of the workspace: it includes the
shared modules from `packages/fisseq-common/nextflow/`. See
[Nextflow Workflow](nextflow.md#running).

## Cluster / HPC

The package ships a `nextflow.config` with default `params` values and
commented-out profile stubs for `venv`, `conda`, `singularity`, and `sge` executors.
To run on a cluster:

1. Write your own config (or copy and adapt `nextflow.config`).
2. Uncomment and fill in a profile block — pick one `beforeScript` option to make
   the `fisseq_data_pipeline` and `fisseq_common` packages available on each compute node:

   ```groovy
   // Option A: activate a pre-existing venv (recommended for shared clusters)
   beforeScript = 'source /path/to/your/venv/bin/activate'

   // Option B: install from GitHub on each run (simpler, slower)
   beforeScript = 'uv pip install "fisseq-data-pipeline @ git+https://github.com/Lilferrit/fisseqmono@v2.0.0#subdirectory=packages/fisseq-data-pipeline" --system'
   ```

3. Pass it at run time:

   ```bash
   nextflow run . -c your.config -profile sge --pipeline_dir /path/to/experiment -params-file params.yaml
   ```

See [Configuration](configuration.md) for the full parameter reference.
