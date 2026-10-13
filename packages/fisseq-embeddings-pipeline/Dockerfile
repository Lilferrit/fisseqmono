# Dockerfile -- the single image every Nextflow task runs in (Docker by
# default, or Apptainer via `-profile apptainer`). Built from the fisseqmono workspace root,
# which holds the shared fisseq-common package and the one uv.lock:
#
#   docker build -f packages/fisseq-embeddings-pipeline/Dockerfile -t fisseq-embeddings-pipeline .
#
# The image also runs BUILD_CELL_IMAGES' nested starcall snakemake and, in
# cluster mode, every starcall child job it submits (params.starcall_job_image).
# One CUDA-capable base serves the CPU-only stages too, to build and publish a
# single artifact.
#
# Not run with `--gpus all` on an actual GPU host.
#
# The `ops` conda env below isolates starcall-workflow's Python 3.10 stack
# (tensorflow/stardist/cellpose) from this repo's Python 3.13 torch/polars one.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ca-certificates/git: the uv project install. curl: the Miniforge installer
# (the base image ships no conda). build-essential: the `ops` env's snakemake
# dependency `datrie` has no prebuilt wheel for this platform and fails to
# compile without gcc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv as a standalone static binary (Astral's published image), not via `pip
# install uv`: it needs no system Python, and uv provisions its own interpreter
# (jammy's apt only carries 3.10, which doesn't satisfy `requires-python`).
# Pinned, not :latest: a new uv release would invalidate every cached layer
# below it on the next CI build. Bump it deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/
# Put the uv-managed interpreter where every user can read it: under
# /root/.local/share/uv/python/ (the default), .venv/bin/python is a symlink
# into a directory a non-root user (Apptainer always, docker under
# nextflow.config's `-u $(id -u):$(id -g)`) can't traverse, so every task failed
# with "python: Permission denied".
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

# ── starcall-workflow's `ops` conda env ─────────────────────────────────────
# Placed before the uv deps layer: it is the slowest block to build, changes
# least often (only with the starcall-workflow pin or the env itself) and uses
# nothing from the uv layers, so neither a source commit nor a
# uv.lock/pyproject.toml change (every release rewrites both) redoes it.
# Installed to /opt/conda. The `ops` env's bin is deliberately NOT on ENV PATH:
# bare `python`/`python3` must always resolve to the uv-managed 3.13 venv, never
# this env's 3.10. The one process that needs `ops` (the `snakemake` line in
# BUILD_CELL_IMAGES) reaches it by absolute path,
# /opt/conda/envs/ops/bin/<binary>. Conda's base bin (/opt/conda/bin) IS on
# PATH, last: snakemake's --use-conda shells out to a bare `conda`, in the
# nested run and every starcall child job.
# The installer architecture is detected, not hardcoded: an x86_64 installer
# under arm64 fails opaquely ("rosetta error: failed to open elf"/SIGTRAP).
# `uname -m` matches Miniforge's `Miniforge3-Linux-<arch>.sh` asset naming.
RUN miniforge_arch="$(uname -m)" \
    && curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${miniforge_arch}.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

# python 3.10, matching starcall-workflow's own readme.md (its
# `pip3 install -r requirements.txt` step assumes it).
RUN /opt/conda/bin/conda create -y -n ops python=3.10 && /opt/conda/bin/conda clean -afy

# Activate the `ops` env for every following RUN in this block (`conda
# activate` doesn't persist across RUN layers); reset to bash below.
SHELL ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "ops", "/bin/bash", "-c"]

# snakemake plus conda-frontend support for `--use-conda` (materializes the
# CellProfiler env the first time a cp_features: true experiment needs it).
# BUILD_CELL_IMAGES passes --conda-frontend conda for the widest
# compatibility; mamba (on PATH via Miniforge) might be faster, untested.
#
# PINNED to the last release that still supports this env's Python 3.10: every
# snakemake >=8.0.0 requires Python >=3.11. That is load-bearing: snakemake 8
# replaced `--cluster`/`--cluster-cancel`/`--jobscript` with the
# executor-plugin interface, and BUILD_CELL_IMAGES' cluster mode
# (params.starcall_profile, a snakemake 7 profile, plus our --jobscript) uses
# the 7.x spelling. Pinning makes a future `ops` Python bump fail at build time
# instead of silently pulling in an 8.x whose flags no longer parse. Moving to
# 8/9 first needs starcall-workflow's requirements.txt (tensorflow==2.13.0/
# stardist==0.8.5/cellpose) re-resolved against Python >=3.11.
RUN pip install --no-cache-dir "snakemake==7.32.4"

# starcall-workflow, cloned at ONE pinned commit: 62414b0 on its
# origin/sense_analysis branch, which branches off origin/devel. NOT master,
# which has a differently-shaped make_cell_images and no extract_embeddings
# rule; see docs/architecture.md's "A note on branches". 62414b0 stops blank
# well-edge tiles from crashing find_dots (constitch stitches non-image regions
# as NaN, which dot calling ignores); devel's tip dies on them with "cannot
# reshape array of size 0". The same checkout supplies both the `ops` env's
# Python packages (requirements.txt plus the two submodules' setup.py packages,
# installed below) and the Snakefile the nested run executes:
# snakemake/Snakefile (copied in further down) includes
# ../starcall-workflow/workflow/Snakefile, so it is cloned to that relative
# path. Bump the pin by changing the default here (or
# `docker build --build-arg STARCALL_WORKFLOW_COMMIT=<sha>`); see
# docs/architecture.md decision 24. --recursive pulls packages/starcall and
# packages/constitch.
ARG STARCALL_WORKFLOW_GIT_URL=https://github.com/FowlerLab/starcall-workflow.git
ARG STARCALL_WORKFLOW_COMMIT=62414b0b4a818c4ad8c5338c9881ce240272d2fc
WORKDIR /opt/fisseq-embeddings-pipeline
RUN git clone --recursive "${STARCALL_WORKFLOW_GIT_URL}" starcall-workflow \
    && git -C starcall-workflow checkout "${STARCALL_WORKFLOW_COMMIT}" \
    && git -C starcall-workflow submodule update --init --recursive

# SETUPTOOLS_USE_DISTUTILS=stdlib is required: stardist's legacy
# numpy.distutils-based build imports `distutils.msvccompiler`, which
# setuptools' vendored `_distutils` (the default since setuptools>=60) doesn't
# ship, so the build fails with ModuleNotFoundError even on Linux. Exported in
# this RUN only, never ENV: the uv layers below build sdists (antlr4 for hydra)
# on Python 3.13, which has no stdlib distutils, so it would fail `uv sync`.
# NOT editable (-e) installs for the two submodules: both
# packages/starcall/setup.py and packages/constitch/setup.py declare
# `packages=['starcall/']`/`['constitch/']` (stray trailing slash, a bug in
# starcall-workflow's own setup.py), so the editable-install finder's mapping
# key never matches `import starcall` and it raises ModuleNotFoundError. A
# plain install tolerates the slash and needn't stay live against the checkout
# (only workflow/ is kept, below).
RUN export SETUPTOOLS_USE_DISTUTILS=stdlib \
    && pip install --no-cache-dir -r starcall-workflow/requirements.txt \
    && pip install --no-cache-dir starcall-workflow/packages/constitch \
    && pip install --no-cache-dir starcall-workflow/packages/starcall

# Keep only workflow/ (the rules the nested run executes); the rest is
# build-time only.
RUN find starcall-workflow -mindepth 1 -maxdepth 1 ! -name workflow -exec rm -rf {} +

# Reset SHELL: the uv-facing layers below must not run inside the `ops` env.
SHELL ["/bin/bash", "-c"]

# The Python environment lives in the workspace at /opt/fisseqmono (its .venv); the nested
# starcall run's files (snakemake/Snakefile, the starcall-workflow clone) stay at
# /opt/fisseq-embeddings-pipeline, where nextflow.config's ext.fisseq_snakefile points.
WORKDIR /opt/fisseqmono

# .python-version (pins 3.13, not just pyproject.toml's open-ended `>=3.13`)
# must be copied in before the first `uv sync`: otherwise uv picks the newest
# interpreter it can fetch (3.14), where Hydra 1.3.x's `get_args_parser()`
# crashes on the stricter argparse `_check_help`, breaking every stage's
# `python -m fisseq_embeddings_pipeline.<module>`.
#
# Dependencies-only layer, cached independently of source changes: the workspace
# definition (root pyproject.toml, uv.lock and every member's pyproject.toml)
# and nothing else. --no-install-workspace leaves out the workspace members
# (hatchling needs their src/ and README.md, not copied yet); --no-dev excludes
# the root dev group, since every process runs `python -m ...` directly.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/fisseq-common/pyproject.toml packages/fisseq-common/
COPY packages/fisseq-data-pipeline/pyproject.toml packages/fisseq-data-pipeline/
COPY packages/fisseq-embeddings-pipeline/pyproject.toml packages/fisseq-embeddings-pipeline/
COPY packages/fisseqborn/pyproject.toml packages/fisseqborn/
RUN uv sync --frozen --no-dev --package fisseq-embeddings-pipeline --no-install-workspace

# What BUILD_CELL_IMAGES' nested snakemake runs (process.ext.fisseq_snakefile
# in nextflow.config): snakemake/Snakefile, which includes the pinned
# starcall-workflow clone's own (above) by relative path.
COPY packages/fisseq-embeddings-pipeline/snakemake/ /opt/fisseq-embeddings-pipeline/snakemake/

# Now fisseq-common and the pipeline itself; the deps layer above stays cached
# across source-only changes. dinov2 is vendored as pure-torch source under
# src/fisseq_embeddings_pipeline/vendor/dinov2/ (see its VENDORED_FROM.md), so
# it installs with the package.
COPY packages/fisseq-common/README.md packages/fisseq-common/
COPY packages/fisseq-common/src/ packages/fisseq-common/src/
COPY packages/fisseq-embeddings-pipeline/README.md packages/fisseq-embeddings-pipeline/
COPY packages/fisseq-embeddings-pipeline/src/ packages/fisseq-embeddings-pipeline/src/
RUN uv sync --frozen --no-dev --package fisseq-embeddings-pipeline --no-editable

# `uv sync` installs into a project-local .venv/; put it on PATH so the bare
# `python -m fisseq_embeddings_pipeline.<module>` every task invokes resolves
# there. The `ops` env is deliberately never added (no ambiguous bare python).
# /opt/conda/bin goes LAST, for `conda` alone (--use-conda's `conda: cp4`
# rules, e.g. run_cellprofiler): a starcall child job re-enters this image
# with this PATH, not its submitter's. Last, so the venv's `python` wins.
ENV PATH="/opt/fisseqmono/.venv/bin:${PATH}:/opt/conda/bin"

ENTRYPOINT []
