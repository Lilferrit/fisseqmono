# Dockerfile -- the single image every Nextflow task runs in (Docker by
# default, or Apptainer via `-profile apptainer`). Built from the fisseqmono workspace root,
# which holds the shared fisseq-common package and the one uv.lock:
#
#   docker build -f packages/fisseq-embeddings-pipeline/Dockerfile -t fisseq-embeddings-pipeline .
#
# The image runs including
# BUILD_CELL_IMAGES' nested starcall snakemake and, in cluster mode, every
# starcall child job it submits (params.starcall_job_image). One CUDA-capable
# base serves CPU-only stages too (simpler to build/publish as one
# artifact); revisit splitting into a CPU + GPU image later if the pull
# cost matters in practice.
#
# Build-verified for real (docker build + docker run against every stage's
# CLI entry point) against the host's real Docker daemon. Not run with
# `--gpus all` on an actual GPU host.
#
# BUILD_CELL_IMAGES used to run in a wholly separate image
# (docker/starcall.Dockerfile, since deleted) because it's the one stage
# that invokes starcall-workflow's own Snakemake pipeline, whose dependency
# stack (tensorflow/stardist/cellpose) was assumed likely-incompatible with
# this repo's own torch/polars stack. That assumption was never actually
# tested against a real merged build -- the risk was really a Python
# version mismatch (this repo: 3.13, starcall-workflow's own baseline:
# 3.10), which a second, isolated conda env inside this one image resolves
# just as well as a second image would. See the `ops` env block below --
# folded in here almost verbatim from the deleted Dockerfile, including all
# its build-failure-derived rationale comments, which are still true. One
# concrete benefit of merging: this repo's own CI (.github/workflows/
# docker.yml) only ever built this file, never docker/starcall.Dockerfile
# -- the `ops` env now gets real CI build coverage for the first time.
#
# Stages: `base` (CUDA, apt, uv) under everything; `conda` builds /opt/conda
# (Miniforge, the `ops` and `cp4` envs) and the starcall-workflow checkout;
# `venv` builds the uv dependency venv; the final stage copies both in. The
# builder stages exist to split the image into layers of ~1 GB or less
# (compressed): the Fowler cluster's route to ghcr's blob CDN drops long
# downloads, a layer can't resume, and the `ops` pip install and the uv venv
# were ~4 GB layers each that never came through. Each builder ends by moving
# its largest directories out to /layers/<group>/<same absolute path>, and the
# final stage copies each group back to that path as its own layer. If a group
# grows well past 1 GB, split it again.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# ca-certificates/git: needed by the uv project install below. curl: fetches
# the Miniforge installer (this base image ships no conda). build-essential: the `ops`
# env's snakemake dependency `datrie` has no prebuilt wheel for this
# platform and fails to compile ("error: [Errno 2] No such file or
# directory: 'gcc'") without a C compiler present -- confirmed via a real
# `docker build` of the pre-merge docker/starcall.Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv as a standalone static binary (Astral's own published image), not via
# `pip install uv` -- this needs no system Python/pip pre-installed at all,
# since uv provisions its own Python interpreter itself (confirmed
# empirically: `uv sync` pulls a managed CPython even with only an
# incompatible system Python on PATH). ubuntu22.04 (jammy)'s own apt
# archives only carry Python 3.10 by default (3.11 needs a backport/PPA,
# and neither satisfies pyproject.toml's `requires-python = ">=3.13"`
# regardless) -- rather than fight that, let uv manage the interpreter
# version explicitly, which is what actually determines what runs at
# container run time anyway (see the ENV PATH line below).
# Pinned, not :latest: a new uv release would change this layer and so invalidate every cached
# layer below it (the whole image) on the next CI build. Bump it deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/
# Put that managed interpreter somewhere every user can read. By default uv
# installs it under /root/.local/share/uv/python/, and .venv/bin/python is
# only a symlink into there -- so every task failed with "python:
# Permission denied" whenever the container ran as a non-root user, which
# Apptainer always does and docker does under nextflow.config's
# `-u $(id -u):$(id -g)`.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

# ── starcall-workflow's `ops` conda env ─────────────────────────────────────
# Its own stage, independent of the uv layers: it is the slowest block to build and changes
# least often (only with the starcall-workflow pin or the `ops` env itself), and it uses
# nothing from the uv layers, so neither a source commit nor a uv.lock/pyproject.toml change
# (every release rewrites both) makes Docker redo it. Installed
# to /opt/conda. The `ops` env's bin is deliberately NOT on ENV PATH: bare
# `python`/`python3` must always resolve to the uv-managed 3.13 venv (ENV
# PATH line near the bottom), never ambiguously to this env's Python 3.10.
# Every process that needs something from `ops` (just the one `snakemake
# ...` line in BUILD_CELL_IMAGES) reaches it by absolute path,
# /opt/conda/envs/ops/bin/<binary>, instead. Conda's own base bin
# (/opt/conda/bin) IS on PATH, last: snakemake's --use-conda shells out to
# a bare `conda`, in the nested run and in every starcall child job.
# Architecture-detected, not hardcoded to x86_64 -- confirmed via a real
# `docker build` that a hardcoded x86_64 installer run under arm64 (native
# or emulated) fails opaquely ("rosetta error: failed to open elf"/a
# SIGTRAP from the self-extracting installer trying to run x86_64 code),
# not with an obvious "wrong architecture" message. `uname -m` maps
# `x86_64`->`x86_64` and `aarch64`->`aarch64`, matching Miniforge's own
# release asset naming (`Miniforge3-Linux-<arch>.sh`) exactly.
FROM base AS conda
RUN miniforge_arch="$(uname -m)" \
    && curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${miniforge_arch}.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

# python 3.10, matching starcall-workflow's own readme.md exactly (its
# `pip3 install -r requirements.txt` step assumes this interpreter
# version -- not independently re-verified against every pinned dependency
# in requirements.txt, e.g. tensorflow==2.13.0/stardist==0.8.5/
# cellpose==2.2.2, but matching the documented baseline is the safest
# starting point).
RUN /opt/conda/bin/conda create -y -n ops python=3.10 && /opt/conda/bin/conda clean -afy

# Activate the `ops` env for every subsequent RUN in this block -- conda's
# own `conda activate` doesn't persist across Docker RUN layers without
# this SHELL trick. Reset back to plain bash once the `ops`-specific steps
# are done (below), so it doesn't leak into the uv-facing layers that
# follow.
SHELL ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "ops", "/bin/bash", "-c"]

# snakemake (starcall-workflow's own requirement is `snakemake>=7`). Its
# `--use-conda` only activates run_cellprofiler's `conda: 'cp4'`: a named
# env, which snakemake never creates. The image builds it, below.
#
# PINNED, not `>=7`, and specifically to the last release that still
# supports this env's Python 3.10: every snakemake >=8.0.0 declares
# `requires_python >=3.11` (confirmed against PyPI for 8.0.0/8.5.0/9.0.1/
# 9.27.0), so a bare `>=7` silently resolves to 7.32.4 today *only* because
# `conda create -n ops python=3.10` above holds it back. That accident is
# load-bearing: snakemake 8 replaced `--cluster`/`--cluster-cancel`/
# `--jobscript` with the executor-plugin interface, and BUILD_CELL_IMAGES'
# cluster mode (params.starcall_profile, a snakemake 7 profile, plus our
# --jobscript) is written against the 7.x spelling. Pinning here means a future `ops` Python bump
# fails loudly at build time instead of teleporting that invocation into an
# 8.x where its flags no longer parse. Moving to 8/9 means first re-resolving
# starcall-workflow's requirements.txt (tensorflow==2.13.0/stardist==0.8.5/
# cellpose==2.2.1) against Python >=3.11 -- a much larger change than it
# looks, and deliberately not bundled with the cluster work.
RUN pip install --no-cache-dir "snakemake==7.32.4"

# starcall-workflow, cloned at build time at ONE pinned commit: 0edcc1b on
# its origin/devel branch, the commit experiment LMNA_T3_R1_cp111925 ran
# (submodules starcall 97b1952, constitch 3be8729). NOT master, which has a
# differently-shaped make_cell_images and no extract_embeddings rule at all;
# see docs/architecture.md's "A note on branches". This rolls back from
# 62414b0 (origin/sense_analysis), which stitched non-image regions as NaN
# so blank well-edge tiles didn't crash find_dots; at 0edcc1b they die on
# them again with "cannot reshape array of size 0". The same checkout supplies
# both the `ops` env's Python packages (requirements.txt plus the two
# submodules' setup.py packages, installed below) and the Snakefile the
# nested run executes: snakemake/Snakefile (copied in with the project
# source, further down) includes ../starcall-workflow/workflow/Snakefile,
# so it's cloned to that path relative to it. So one pin decides which
# starcall every experiment runs. Bump it by changing the default here (or
# `docker build --build-arg STARCALL_WORKFLOW_COMMIT=<sha>`) -- see
# docs/architecture.md decision 24. --recursive pulls packages/starcall
# and packages/constitch.
ARG STARCALL_WORKFLOW_GIT_URL=https://github.com/FowlerLab/starcall-workflow.git
ARG STARCALL_WORKFLOW_COMMIT=0edcc1b8bb355a3471222ac3b80a159cfafc2fdb
WORKDIR /opt/fisseq-embeddings-pipeline
RUN git clone --recursive "${STARCALL_WORKFLOW_GIT_URL}" starcall-workflow \
    && git -C starcall-workflow checkout "${STARCALL_WORKFLOW_COMMIT}" \
    && git -C starcall-workflow submodule update --init --recursive

# SETUPTOOLS_USE_DISTUTILS=stdlib is required -- confirmed via a real
# `docker build`: stardist's legacy numpy.distutils-based build imports
# `distutils.msvccompiler` (a Windows-only compiler shim) unconditionally
# at module load time; setuptools' own vendored `_distutils` copy (the
# default resolution since setuptools>=60) doesn't ship that submodule at
# all, so the build fails with `ModuleNotFoundError: No module named
# 'distutils.msvccompiler'` even on Linux, where it's never actually used.
# Forcing the real stdlib `distutils` (which does ship that submodule,
# platform-conditional behavior notwithstanding) is the standard fix for
# this well-documented numpy.distutils/setuptools incompatibility.
# Exported in this RUN only, never ENV: the uv layers below build sdists
# (antlr4-python3-runtime, for hydra) on Python 3.13, which has no stdlib
# distutils, so the setting there fails `uv sync` with "No module named
# 'distutils'".
# NOT editable (-e) installs for the two submodules -- confirmed via a real
# `docker build` + `docker run` that `pip install -e` fails at import time
# here: both packages/starcall/setup.py and packages/constitch/setup.py
# declare `packages=['starcall/']`/`packages=['constitch/']` (note the
# stray trailing slash -- a real bug in starcall-workflow's own setup.py,
# out of scope to fix here, it's a read-only sibling repo), which makes
# setuptools' auto-generated editable-install finder's own MAPPING key
# ('starcall/', with the slash) never match `fullname` at actual import
# time ('starcall', without it) -- `import starcall` then raises
# ModuleNotFoundError despite `pip show starcall` reporting it installed. A
# plain (non-editable) install doesn't hit this path at all -- it copies
# the package directory once at install time using setuptools' own legacy
# `packages=[...]`-driven file discovery, which tolerates the trailing
# slash typo fine, and does not need to remain "live" against the checkout
# afterward (only workflow/ is kept, below).
RUN export SETUPTOOLS_USE_DISTUTILS=stdlib \
    && pip install --no-cache-dir -r starcall-workflow/requirements.txt \
    && pip install --no-cache-dir starcall-workflow/packages/constitch \
    && pip install --no-cache-dir starcall-workflow/packages/starcall

# Keep only workflow/ (the rules the nested run executes); the git history,
# packages/ and the rest are build-time only once the pip installs above
# succeed.
RUN find starcall-workflow -mindepth 1 -maxdepth 1 ! -name workflow -exec rm -rf {} +

# Reset SHELL back to plain bash -- the layers below (uv-facing) should
# not run inside the `ops` conda env.
SHELL ["/bin/bash", "-c"]

# starcall's `cp4` env (CellProfiler 4.2.4), for run_cellprofiler's
# `conda: 'cp4'`. A named env has to exist before the run: without it every
# CellProfiler job dies with EnvironmentNameNotFound. Built from the pinned
# checkout's own workflow/envs/cp4.yaml, minus its `anaconda` and `defaults`
# channels (repo.anaconda.com, which conda refuses non-interactively until
# its terms of service are accepted); conda-forge and bioconda carry every
# package. cellprofiler 4.2.4 pins centrosome 1.2.1, an sdist whose Cython
# no longer compiles under Cython 3 (`/` became true division: "Cannot
# assign type 'double' to 'unsigned int'"), and pip's isolated build pulls
# the newest Cython; PIP_CONSTRAINT reaches those build envs too. Above the
# uv.lock layer: slow, and changes only with the pin.
RUN sed '/^- anaconda$/d; /^- defaults$/d' \
        /opt/fisseq-embeddings-pipeline/starcall-workflow/workflow/envs/cp4.yaml \
        > /tmp/cp4.yaml \
    && echo 'cython<3' > /tmp/cp4-constraints.txt \
    && PIP_CONSTRAINT=/tmp/cp4-constraints.txt \
        /opt/conda/bin/conda env create -y -n cp4 -f /tmp/cp4.yaml \
    && /opt/conda/bin/conda run -n cp4 cellprofiler --version \
    && /opt/conda/bin/conda clean -afy \
    && rm /tmp/cp4.yaml /tmp/cp4-constraints.txt

# Split /opt/conda for the final stage's layers (header). Each argument is
# `<group>=<path> ...`: those paths move to /layers/<group><path>; a path a
# wheel no longer installs is skipped, leaving its group smaller. Groups are
# bins packed to ~0.7-1 GB compressed each, not tidy categories (measured:
# `tar -c <paths> | gzip | wc -c` inside the image); repack them when a
# torch/tensorflow/CellProfiler change moves the sizes. conda's
# package cache goes first: what `conda clean` keeps there is hardlinked to
# the envs' files, stored once while it all shared a layer, twice once split.
RUN rm -rf /opt/conda/pkgs \
    && sp=/opt/conda/envs/ops/lib/python3.10/site-packages \
    && for spec in \
        "ops-cudnn=${sp}/nvidia/cudnn" \
        "ops-torch=${sp}/torch ${sp}/triton" \
        "ops-nvidia=${sp}/nvidia/cublas ${sp}/nvidia/nccl ${sp}/nvidia/cusparse" \
        "ops-tensorflow=${sp}/tensorflow ${sp}/nvidia/cusolver ${sp}/nvidia/cufft" \
        "cp4=/opt/conda/envs/cp4"; do \
        group="${spec%%=*}"; mkdir -p "/layers/${group}"; \
        for path in ${spec#*=}; do \
            [ -e "${path}" ] || continue; \
            mkdir -p "/layers/${group}$(dirname "${path}")"; \
            mv "${path}" "/layers/${group}${path}"; \
        done; \
    done

# ── uv dependency venv ──────────────────────────────────────────────────────
# The Python environment lives in the workspace at /opt/fisseqmono (its .venv); the nested
# starcall run's files (snakemake/Snakefile, the starcall-workflow clone) stay at
# /opt/fisseq-embeddings-pipeline, where nextflow.config's ext.fisseq_snakefile points.
FROM base AS venv
WORKDIR /opt/fisseqmono

# .python-version (pins 3.13, not just pyproject.toml's open-ended
# `>=3.13`) has to be copied in before the first `uv sync` -- caught by a
# real `docker build`: without it, uv has nothing here to prefer 3.13 over
# any other >=3.13 interpreter, so it silently resolved the newest one it
# could fetch instead (3.14 as of this build) -- which turned out to be
# genuinely incompatible: Hydra 1.3.x's `get_args_parser()` crashes on
# Python 3.14's stricter argparse `_check_help` (`TypeError: argument of
# type 'LazyCompletionHelp' is not a container or iterable`), breaking
# every single stage's `python -m fisseq_embeddings_pipeline.<module>`
# invocation outright, not just `--help`. Confirmed fixed by copying
# .python-version in before either `uv sync` call below, restoring 3.13.
#
# Dependencies-only layer, cached independently of source changes: the workspace definition
# (root pyproject.toml, uv.lock and every member's pyproject.toml) and nothing else.
# --package installs this pipeline's dependencies only; --no-install-workspace leaves out the
# workspace members themselves (hatchling needs their src/ and README.md, not copied yet);
# --no-dev excludes the root dev group (tests, docs, linters) -- every process runs
# `python -m fisseq_embeddings_pipeline.<module>` directly, never `pytest`/`ruff`.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/fisseq-common/pyproject.toml packages/fisseq-common/
COPY packages/fisseq-data-pipeline/pyproject.toml packages/fisseq-data-pipeline/
COPY packages/fisseq-embeddings-pipeline/pyproject.toml packages/fisseq-embeddings-pipeline/
COPY packages/fisseqborn/pyproject.toml packages/fisseqborn/
RUN uv sync --frozen --no-dev --package fisseq-embeddings-pipeline --no-install-workspace

# Split the venv for the final stage's layers, as the `conda` stage does. The
# uv cache stays behind in this stage.
RUN sp=/opt/fisseqmono/.venv/lib/python3.13/site-packages \
    && for spec in \
        "venv-torch=${sp}/torch" \
        "venv-cudnn=${sp}/nvidia/cudnn ${sp}/nvidia/nvshmem" \
        "venv-nvidia-a=${sp}/nvidia/cublas ${sp}/nvidia/nccl ${sp}/nvidia/cufft" \
        "venv-nvidia-b=${sp}/nvidia/cusparselt ${sp}/nvidia/cusparse ${sp}/nvidia/cusolver ${sp}/triton"; do \
        group="${spec%%=*}"; mkdir -p "/layers/${group}"; \
        for path in ${spec#*=}; do \
            [ -e "${path}" ] || continue; \
            mkdir -p "/layers/${group}$(dirname "${path}")"; \
            mv "${path}" "/layers/${group}${path}"; \
        done; \
    done

# ── Final image ─────────────────────────────────────────────────────────────
# One COPY per group, each its own layer, back at its original path; then
# what's left of each tree. The `conda` stage's first: they change only with
# the starcall pin, the venv's with every uv.lock.
FROM base
COPY --from=conda /layers/ops-cudnn/ /
COPY --from=conda /layers/ops-torch/ /
COPY --from=conda /layers/ops-nvidia/ /
COPY --from=conda /layers/ops-tensorflow/ /
COPY --from=conda /layers/cp4/ /
COPY --from=conda /opt/conda /opt/conda
COPY --from=conda /opt/fisseq-embeddings-pipeline/starcall-workflow /opt/fisseq-embeddings-pipeline/starcall-workflow
COPY --from=venv /opt/uv-python /opt/uv-python
COPY --from=venv /layers/venv-torch/ /
COPY --from=venv /layers/venv-cudnn/ /
COPY --from=venv /layers/venv-nvidia-a/ /
COPY --from=venv /layers/venv-nvidia-b/ /
COPY --from=venv /opt/fisseqmono /opt/fisseqmono
WORKDIR /opt/fisseqmono

# What BUILD_CELL_IMAGES' nested snakemake runs (process.ext.fisseq_snakefile
# in nextflow.config): snakemake/Snakefile, which includes the pinned
# starcall-workflow clone's own (above) by relative path.
COPY packages/fisseq-embeddings-pipeline/snakemake/ /opt/fisseq-embeddings-pipeline/snakemake/

# Now fisseq-common and the pipeline itself -- the deps layer above stays cached across
# source-only changes. dinov2 needs no separate vendoring/installation step
# here: it's vendored as pure-torch source directly under
# src/fisseq_embeddings_pipeline/vendor/dinov2/ (see that directory's
# VENDORED_FROM.md for why and what), so it's part of this package and
# installs with it -- no xformers/cuml/git-dependency wrangling needed.
COPY packages/fisseq-common/README.md packages/fisseq-common/
COPY packages/fisseq-common/src/ packages/fisseq-common/src/
COPY packages/fisseq-embeddings-pipeline/README.md packages/fisseq-embeddings-pipeline/
COPY packages/fisseq-embeddings-pipeline/src/ packages/fisseq-embeddings-pipeline/src/
RUN uv sync --frozen --no-dev --package fisseq-embeddings-pipeline --no-editable

# `uv sync` installs into a project-local .venv/, not any system Python --
# put it on PATH so the bare `python -m fisseq_embeddings_pipeline.<module>`
# every task's script invokes actually resolves there.
# Confirmed empirically this matters: without it, `python`/`python3` would
# fall through to whatever's first on the base image's own PATH (nothing,
# now that no system Python is installed above -- but even with one, none
# of the deps installed above would be on its sys.path), and every process
# would fail with ModuleNotFoundError/`python: command not found`. The
# `ops` conda env above is deliberately never added here, for the same
# "no ambiguous bare python" reason in reverse -- see that block's own
# comment. /opt/conda/bin goes LAST, for `conda` alone (--use-conda
# activates the `cp4` env built above for run_cellprofiler): a starcall child job
# re-enters this image with this PATH, not its submitter's, so conda has
# to be here rather than prepended by BUILD_CELL_IMAGES. Last, so the
# venv's `python` still wins over Miniforge's.
ENV PATH="/opt/fisseqmono/.venv/bin:${PATH}:/opt/conda/bin"

ENTRYPOINT []
