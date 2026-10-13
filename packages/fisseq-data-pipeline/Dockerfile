# Dockerfile -- the single image every Nextflow process runs in.
#
# Built from the fisseqmono workspace root, which holds the shared fisseq-common package and
# the one uv.lock:
#
#   docker build -f packages/fisseq-data-pipeline/Dockerfile -t fisseq-data-pipeline .
#
# CPU-only (polars + XGBoost, no CUDA), so a plain ubuntu base is enough.
FROM ubuntu:22.04

# ca-certificates/git: needed by the uv project install below. No compiler
# toolchain is required -- every runtime dependency resolves to a prebuilt
# wheel on both amd64 and arm64.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv as a standalone static binary, not via `pip install uv`: no system Python is needed,
# since uv provisions its own interpreter (jammy's default 3.10 would not satisfy
# `requires-python` anyway).
# Pinned, not :latest: a new uv release would change this layer and so invalidate every cached
# layer below it (the whole image) on the next CI build. Bump it deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /usr/local/bin/
# Put the managed interpreter where every user can read it (the default is under /root), so a
# container run as a non-root user can still execute .venv/bin/python.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

WORKDIR /opt/fisseqmono

# .python-version (pins 3.13 exactly, not just pyproject.toml's `>=3.13,<3.14` range) MUST
# be copied in before the first `uv sync`; otherwise uv may fetch a newer interpreter, and
# Hydra 1.3.x's `get_args_parser()` crashes on Python 3.14's stricter argparse, breaking
# every stage's `python -m` invocation.
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

# uv installs into the workspace's .venv/; with no system Python, every process's bare
# `python -m ...` fails with `python: command not found` unless that venv is on PATH.
ENV PATH="/opt/fisseqmono/.venv/bin:${PATH}"

ENTRYPOINT []
