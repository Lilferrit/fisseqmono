# AGENTS.md — fisseqmono

One uv workspace, four packages. Each package keeps its own detailed guide; this file covers
what spans them.

| Package | Guide |
|---|---|
| `packages/fisseq-common` | this file, [Shared code](#shared-code-fisseq-common) |
| `packages/fisseq-data-pipeline` | [`packages/fisseq-data-pipeline/AGENTS.md`](packages/fisseq-data-pipeline/AGENTS.md) |
| `packages/fisseq-embeddings-pipeline` | [`packages/fisseq-embeddings-pipeline/AGENTS.md`](packages/fisseq-embeddings-pipeline/AGENTS.md) |
| `packages/fisseqborn` | [`packages/fisseqborn/README.md`](packages/fisseqborn/README.md) and `docs/fisseqborn/` |

Documentation for all four is one MkDocs site: `mkdocs.yml` and `docs/` at the root
(`docs/<package>/`, `docs/common/`, `docs/diff-reports/`).

## Layout and dependencies

- Python 3.13 only (`>=3.13,<3.14`: Hydra 1.3 crashes on 3.14), pinned by `.python-version`.
  One `uv.lock`, one ruff config (root `pyproject.toml`), one `.pre-commit-config.yaml`.
- The root is a virtual workspace (not a package). `uv sync` installs its dev group: all four
  packages plus pytest, ruff and pre-commit; `--group docs` adds MkDocs.
- `fisseq-common`'s base install is polars, pyarrow, numpy. The heavy dependencies
  (scikit-learn, xgboost, hydra, scipy, pycytominer) are its `stages` extra. The pipelines
  depend on `fisseq-common[stages]`, fisseqborn on plain `fisseq-common`; keep it that way —
  nothing in the base modules may import a `stages` dependency.
- Every package lists `fisseq-common` by git URL pinned to the release tag, with
  `[tool.uv.sources] fisseq-common = { workspace = true }` so the workspace uses its own copy.
- A package imports no other workspace package except `fisseq-common`.

## Shared code (fisseq-common)

- `schema`, `variant`, `normalizer`, `layout` (where each pipeline publishes its
  per-experiment outputs), `global_aggregation` (cross-experiment methods), `utils/`.
- `stages/`: every stage both pipelines run. A pipeline's `python -m fisseq_<pipeline>.<stage>`
  module is a thin Hydra entry point: its config subclasses the shared one to set the
  pipeline's defaults (module paths, ConfigStore names and config keys never change).
  Differences between the pipelines are explicit parameters, each set by the pipeline's
  wrapper — see `docs/common/index.md` for the table.
- Nextflow: the processes both pipelines run have one copy each,
  `nextflow/modules/local/<stage>/main.nf`. A module carries only what both pipelines pass
  alike; each pipeline's `conf/modules.config` sets `ext.entry`, `ext.args` and `publishDir`.
  Publish paths must match `fisseq_common.layout` (each pipeline's
  `tests/unit/test_publish_layout.py` checks).
- Cross-experiment aggregation is fisseqborn's job (`fisseqborn-global`, on
  `fisseq_common.global_aggregation`); the pipelines write per-experiment outputs only.

## Tests

```bash
uv run --package <package> pytest packages/<package>/tests   # one package
uv run pytest tests                                          # cross-package (needs Nextflow)
```

- `packages/<package>/tests/unit/` and `tests/integration/` must pass with only that package
  installed — CI runs them in a venv built with
  `uv sync --frozen --exact --package <package> --group dev`. Test directories have no
  `__init__.py`; pytest runs with `--import-mode=importlib`.
- Code moved into `fisseq-common` takes its unit tests with it.
- Root `tests/`:
  - `test_reference_outputs.py` reruns each pipeline's integration fixture and compares every
    published parquet with `tests/reference/<scenario>/`. **Output changes must be
    deliberate**: a commit that changes a pipeline's outputs regenerates the references
    (`uv run python tests/reference/capture.py <scenario>`) and adds a report to
    `docs/diff-reports/`. The embeddings pipeline's per-experiment outputs must not change in
    a refactor.
  - `test_global_parity.py` checks fisseqborn's cross-experiment aggregation against the
    outputs of the embeddings pipeline's former global stages (`tests/reference/*/global/`,
    which `capture.py` keeps).
  - `test_layout_load.py`: both pipelines' outputs are where `layout` says and load in
    fisseqborn.
  - `test_release.py`.

## CI (`.github/workflows/`)

- `pkg-<package>.yml` (via `_package.yml`): that package's suite, alone, when the package or
  `fisseq-common` changes (plus `nextflow/` for the pipelines).
- `root.yml`: lint, `uv lock --check`, Nextflow lint, root tests — every change.
- `docker.yml`: both pipeline images, from the root context
  (`docker build -f packages/<pipeline>/Dockerfile .`), pushed to `ghcr.io/<owner>/<pipeline>`
  from `main` and `v*` tags.
- `docs.yml`: strict build on PRs, gh-pages deploy from `main`.

## Releases

One version for the whole workspace. `uv run python scripts/release.py X.Y.Z` sets it in the
five `pyproject.toml` files and the three `fisseq-common` git URLs, then runs `uv lock`;
commit and tag `vX.Y.Z`.
