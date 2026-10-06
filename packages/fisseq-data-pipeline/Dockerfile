# Dockerfile -- the single image every Nextflow process runs in.
#
# Built from the fisseqmono workspace root, which holds the shared fisseq-common package and
# the one uv.lock:
#
#   docker build -f packages/fisseq-data-pipeline/Dockerfile -t fisseq-data-pipeline .
#
# CPU-only: this pipeline is polars + XGBoost + pycytominer, with no CUDA
# dependency anywhere, so a plain ubuntu base is enough (the sibling
# fisseq-embeddings-pipeline needs nvidia/cuda for its Cell-DINO stage; nothing
# here does).
FROM ubuntu:22.04

# ca-certificates/git: needed by the uv project install below. No compiler
# toolchain is required -- every runtime dependency resolves to a prebuilt
# wheel on both amd64 and arm64.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv as a standalone static binary (Astral's own published image), not via
# `pip install uv` -- this needs no system Python or pip pre-installed at all,
# since uv provisions its own interpreter. ubuntu 22.04 (jammy) only ships
# Python 3.10 by default, which does not satisfy pyproject.toml's
# `requires-python` regardless.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
# Put the managed interpreter where every user can read it (the default is under /root), so a
# container run as a non-root user can still execute .venv/bin/python.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /opt/fisseqmono

# .python-version (which pins 3.13 exactly, not just pyproject.toml's
# `>=3.13,<3.14` range) MUST be copied in before the first `uv sync`. Without
# it uv has nothing here to prefer 3.13 over any newer interpreter it can
# fetch, and Hydra 1.3.x's `get_args_parser()` crashes on Python 3.14's
# stricter argparse -- which breaks every stage's `python -m` invocation.
#
# Dependencies-only layer, cached independently of source changes: the workspace definition
# (root pyproject.toml, uv.lock and every member's pyproject.toml) and nothing else.
# --package installs this pipeline's dependencies only; --no-install-workspace leaves out the
# workspace members themselves, whose sources aren't here yet; --no-dev leaves out the root
# dev group (tests, docs, linters).
COPY pyproject.toml uv.lock .python-version ./
COPY packages/fisseq-common/pyproject.toml packages/fisseq-common/
COPY packages/fisseq-data-pipeline/pyproject.toml packages/fisseq-data-pipeline/
COPY packages/fisseq-embeddings-pipeline/pyproject.toml packages/fisseq-embeddings-pipeline/
COPY packages/fisseqborn/pyproject.toml packages/fisseqborn/
RUN uv sync --frozen --no-dev --package fisseq-data-pipeline --no-install-workspace

# Now fisseq-common and the pipeline itself -- the deps layer above stays cached across
# source-only changes.
COPY packages/fisseq-common/README.md packages/fisseq-common/
COPY packages/fisseq-common/src/ packages/fisseq-common/src/
COPY packages/fisseq-data-pipeline/README.md packages/fisseq-data-pipeline/
COPY packages/fisseq-data-pipeline/src/ packages/fisseq-data-pipeline/src/
RUN uv sync --frozen --no-dev --package fisseq-data-pipeline --no-editable

# `uv sync` installs into the workspace's .venv/, not any system Python -- put
# it on PATH so the bare `python -m fisseq_data_pipeline.<module>` that every
# process script invokes actually resolves there. Without
# this, `python` would not resolve at all (no system Python is installed
# above), and every process would fail with `python: command not found`.
ENV PATH="/opt/fisseqmono/.venv/bin:${PATH}"

ENTRYPOINT []
