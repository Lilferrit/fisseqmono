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
  (scikit-learn, xgboost, hydra, scipy) are its `stages` extra. The pipelines
  depend on `fisseq-common[stages]`, fisseqborn on plain `fisseq-common`; keep it that way —
  nothing in the base modules may import a `stages` dependency.
- Every package lists `fisseq-common` by git URL pinned to the release tag, with
  `[tool.uv.sources] fisseq-common = { workspace = true }` so the workspace uses its own copy.
- A package imports no other workspace package except `fisseq-common`.

## Shared code (fisseq-common)

- `schema`, `variant`, `normalizer`, `layout` (where each pipeline publishes its
  per-experiment outputs), `global_aggregation` (cross-experiment methods), `utils/`.
- `stages/`: every stage both pipelines run, whole: the algorithm, the Hydra config and the
  entry point `python -m fisseq_common.stages.<stage>` (`stage_main` in `stages/config.py`):
  `qcfilter`, `filter`, `ovwt`, `aggregate`, `generatesplit`, `correlatefeatures`,
  `blocklist`, `combineblocklists`, `finalize`. The pipelines have no wrapper modules.
  Differences between the pipelines are config fields (`join_keys`, `feature_selector`, QC
  column names, ...) set in each pipeline's `conf/modules.config` — see
  `docs/common/index.md` for the table and `docs/common/stages.md` for each stage.
- A cell is identified by the pipeline's `join_keys`; rows are sorted and split on
  `row_keys(join_keys)` (plus `meta_variant_tag`, so a QC pseudo-variant row stays distinct
  from its source cell). Stages take `meta_*` columns from QC_FILTER's side and features from
  the cell table.
- Nextflow: the processes of the shared stages have one copy each,
  `packages/fisseq-common/nextflow/modules/local/<stage>/main.nf` (+ `functions.nf`;
  modules only, no workflow). Each module hardcodes its entry point and carries what both
  pipelines pass alike; each pipeline's `conf/modules.config` sets only `ext.args`, `ext.seed`
  and `publishDir`. Publish paths must match `fisseq_common.layout` (each pipeline's
  `tests/unit/test_publish_layout.py` checks).
- Downstream of their own cell tables the two pipelines run the same graph, process names,
  parameters, controls (wildtype) and publish layout; only the cell table differs
  (CellProfiler features vs. Cell-DINO embeddings, plus the embeddings pipeline's CellProfiler
  track).
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
    `docs/diff-reports/`. A pipeline's outputs change only in a commit that changes them
    deliberately, with a report.
  - `test_layout_load.py`: both pipelines' outputs are where `layout` says, load in
    fisseqborn and go through `fisseqborn.write_global`.
  - `test_release.py`, `test_workspace.py`.

## CI (`.github/workflows/`)

Pull requests run lint and unit tests; pushes to `main` build the docs and the images. The
integration and root tests run in CI only on a manual run (`workflow_dispatch`, the "Run
workflow" button), so run them locally before merging.

- `pkg-<package>.yml` (via `_package.yml`): that package's unit tests, alone, on a PR that
  changes the package or `fisseq-common` (which holds the shared Nextflow modules). A manual
  run adds the pipelines' integration tests.
- `root.yml`: lint, `uv lock --check`, Nextflow lint (`packages/fisseq-common/nextflow` and the
  pipelines) on every PR; a manual run adds the root tests.
- `docker-<pipeline>.yml` (via `_docker.yml`): one pipeline image, from the root context
  (`docker build -f packages/<pipeline>/Dockerfile .`), built and pushed to
  `ghcr.io/<owner>/fisseqmono/<pipeline>` from `main` and `v*` tags (never on PRs). Its
  `paths` filter lists exactly the files the Dockerfile copies (Nextflow files, tests and docs
  never reach an image; keep the filter in step when a Dockerfile changes). Tag pushes always
  build. Layers are cached in ghcr at `<image>:buildcache` (BuildKit registry cache,
  `mode=max`), so a source-only change rebuilds only the source layers; keep the uv image
  pinned and the slow, rarely-changing layers (the embeddings image's `ops` env) above the
  `uv.lock` layer, or that cache stops paying off.
- `docs.yml`: strict build and gh-pages deploy on every push to `main`.

## Releases

One version for the whole workspace. `uv run python scripts/release.py X.Y.Z` sets it in the
five `pyproject.toml` files and the three `fisseq-common` git URLs, then runs `uv lock`;
commit and tag `vX.Y.Z`.
