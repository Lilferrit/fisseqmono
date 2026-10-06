# Quickstart

The fastest path from a fresh checkout to a first pipeline run, including
setting up a cluster config. For more depth, see
[Installation](installation.md) (full environment setup),
[Configuration](configuration.md) (every parameter and the `experiments:`
schema), and
[Walkthrough](walkthrough.md) (stage-by-stage detail on what each part of the
pipeline does).

## 1. Install

```bash
git clone https://github.com/Lilferrit/fisseqmono.git
cd fisseqmono
uv sync
cd packages/fisseq-data-pipeline
```

The run commands below are run from `packages/fisseq-data-pipeline/`. The workflow includes
the shared Nextflow modules from `packages/fisseq-common/nextflow/`, so it needs the whole
workspace checkout. See [Installation](installation.md) for requirements and the pip-only
path.

## 2. Get a `nextflow.config`

The package ships a `nextflow.config` with default `params` and a set
of commented-out profile stubs (`venv`, `conda`, `singularity`, `sge`). You can
either run with it as-is, or copy it and use it as a template for your own
site.

If you've already cloned the repo, you have it at
`packages/fisseq-data-pipeline/nextflow.config`. To fetch just the config file on its own,
download it directly from GitHub:

```bash
wget https://raw.githubusercontent.com/Lilferrit/fisseqmono/main/packages/fisseq-data-pipeline/nextflow.config -O your.config
```

### Making your own profile

Nextflow profiles control *where* and *how* each process runs (executor,
queue, resources, environment setup) — separate from the pipeline `params` in
the same file. To add one:

1. Copy `your.config` (or `nextflow.config`) somewhere you can edit it.
2. Uncomment one of the stub blocks in `profiles { }` — the `sge` stub is the
   right starting point for a Sun Grid Engine cluster:

   ```groovy
   sge {
       process {
           executor       = 'sge'
           queue          = 'all.q'           // SGE queue name
           clusterOptions = '-V'              // forward current environment to each job
           cpus           = 4
           memory         = '16 GB'
           time           = '4h'
           beforeScript   = "source /path/to/.venv/bin/activate"
       }
   }
   ```

3. Rename the block to something specific to your site if you like (e.g.
   `my_sge`) and fill in your actual queue name, resource limits, and a
   `beforeScript` that makes `fisseq_data_pipeline` and `fisseq_common[stages]` importable
   on each compute node — either activating a pre-built venv (recommended) or installing it
   fresh on every run. See [Installation: Cluster / HPC](installation.md#cluster-hpc)
   for both `beforeScript` options in full.
4. Pass your config and profile name at run time with `-c your.config -profile
   my_sge` (see the run commands below).

## 3. Run commands

From `packages/fisseq-data-pipeline/` in your clone:

```bash
nextflow run . -c your.config -profile my_sge --pipeline_dir /path/to/experiment -params-file params.yaml
```

Resume a previously interrupted run instead of starting over:

```bash
nextflow run . -c your.config -profile my_sge --pipeline_dir /path/to/experiment -params-file params.yaml -resume
```

Experiments are declared as a list under `experiments:` in `params.yaml`,
which must be passed with `-params-file` — see
[Configuration](configuration.md#declaring-experiments). `--pipeline_dir` is
just the output root; it needs no pre-existing contents.
See [Configuration: Parameters](configuration.md#parameter-reference) for every
`--param` the pipeline accepts.

### Running a single step directly

Every pipeline stage is also runnable on its own as a Python CLI, without
Nextflow — useful for debugging or rerunning one step against different
inputs. Every stage after INPUT is a shared fisseq-common entry point; for example,
one aggregation method over an experiment's normalized cells:

```bash
uv run python -m fisseq_common.stages.aggregate \
    output_dir=./out \
    cells_file=out/qc_filter/batch1/filtered_cells.parquet \
    filtered_keys_file=out/normalization/batch1/filtered_keys.parquet \
    normalizer_file=out/normalization/batch1/normalizer.parquet \
    aggregator=KS output_name=KS
```

See [Shared stages](../common/stages.md) for every stage's config fields and an example
command, and the [CLI Reference](cli/input.md) for this package's own entry points.

## 4. Example cluster launcher script

A minimal launcher script for submitting the pipeline as an SGE job, kept
up-to-date and re-installable on every run. Save it in the directory you want
the run's working files and `.nextflow` cache in, adjust `SCRIPT_DIR` and
`-profile`, and submit it as your SGE job script:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="/path/to/your/analysis/dir"
module load nextflow
export NXF_HOME="${SCRIPT_DIR}/.nextflow"

# NOTE: for SGE jobs, replace $SCRIPT_DIR with a hard-coded absolute path,
# since SGE may not run from the expected working directory.
cd "${SCRIPT_DIR}"

if [[ ! -d "${SCRIPT_DIR}/.venv" ]]; then
    echo "No venv found at ${SCRIPT_DIR}/.venv — creating one with uv"
    uv venv "${SCRIPT_DIR}/.venv"
fi
source "${SCRIPT_DIR}/.venv/bin/activate"

RESUME_FLAG="-resume"
if [[ "${CLEAN:-false}" == "true" ]]; then
    RESUME_FLAG=""
fi

# Get the latest version of the workspace (the workflow needs fisseq-common's modules)
REPO_DIR="${SCRIPT_DIR}/fisseqmono"
if [[ ! -d "${REPO_DIR}" ]]; then
    git clone https://github.com/Lilferrit/fisseqmono.git "${REPO_DIR}"
fi
git -C "${REPO_DIR}" pull --ff-only
uv pip install --upgrade "${REPO_DIR}/packages/fisseq-data-pipeline"  # + fisseq-common[stages]

nextflow run "${REPO_DIR}/packages/fisseq-data-pipeline" \
    -c "${SCRIPT_DIR}/your.config" \
    -profile my_sge \
    -params-file "${SCRIPT_DIR}/params.yaml" \
    --pipeline_dir "${SCRIPT_DIR}" \
    ${RESUME_FLAG} \
    "$@"
```

Set `CLEAN=true` before running it to force a fresh run instead of resuming
(e.g. `CLEAN=true ./run.sh`). Any extra arguments passed to the script (`"$@"`)
are forwarded straight to `nextflow run`, so you can override any pipeline
parameter without editing the script, e.g. `./run.sh --barcode_count_threshold 15`
(list-valued params like `--feature_select_types` need a `-params-file` override
instead — see
[Configuration: Passing list values on the CLI](configuration.md#passing-list-values-on-the-cli)).

## Next steps

- [Walkthrough](walkthrough.md) — a complete end-to-end run, stage by stage.
- [Nextflow Workflow](nextflow.md) — every process and profile.
- [Configuration](configuration.md) — every parameter and the `experiments:`
  schema.
- [Architecture](architecture.md) — the full pipeline DAG and output layout.
- [Shared stages](../common/stages.md) — config fields and examples for every
  stage after INPUT.
