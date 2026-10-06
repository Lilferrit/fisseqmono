# AGENTS.md — fisseq-embeddings-pipeline

> **This repo is implemented.** `docs/embeddings-pipeline/` (built with mkdocs, published to
> GitHub Pages on every push to `main` — see **CI** below) is the
> authoritative reference: architecture decisions, data contracts,
> per-stage config/usage, Nextflow wiring, and output layout. Read it
> before `SPEC.md`/`IMPLEMENTATION_CHECKLIST.md`, which no longer exist —
> their content was folded into `docs/embeddings-pipeline/` once implementation caught up to
> the design.

---

## Project overview

`fisseq-embeddings-pipeline` is the embedding-space sibling of
`fisseq-data-pipeline`: it scores genetic variants against a pretrained **Cell-DINO** vision
transformer's learned embeddings instead of hand-engineered CellProfiler features. Upstream
of the embeddings it is its own (starcall-workflow run, cell crops, Cell-DINO); **downstream
of `EMBED_CELLS` it runs the data pipeline's stages**: the same `fisseq_common.stages` entry
points and `packages/fisseq-common/nextflow` modules, the same process names
(`QC_FILTER` → `NORMALIZE` → `OVWT_BATCHWISE` + the `*_BATCHWISE` bootstrap feature
selection ending in `FINALIZE_FEATURE_SELECT_BATCHWISE`), the same `params.yaml` names and
defaults, and the same publish layout (`fisseq_common.layout`). Pooling across experiments is
fisseqborn's `fisseqborn-global`. See
[`docs/embeddings-pipeline/architecture.md`](../../docs/embeddings-pipeline/architecture.md)
and [`docs/common/stages.md`](../../docs/common/stages.md).

**This package's code** (everything else is fisseq-common's):

```text
src/fisseq_embeddings_pipeline/
  config/__init__.py              re-exports fisseq_common.stages.config's AppConfig & co.
  config/experiments.py           PLAN_EXPERIMENTS: params validation, per-experiment routing,
                                  RENAMED_PARAMS / removed-param warnings
  build_cell_images_enumerate.py  BUILD_CELL_IMAGES phase 1 (tiles, starcall targets, jobscript)
  tile_shard.py                   make_cell_shard rule body (phase 2, inside the nested snakemake)
  build_cell_images_table.py      BUILD_CELL_IMAGES phase 3 (cell_table.parquet, tiles.parquet)
  cell_metadata.py                BUILD_CELL_METADATA (QC_FILTER's input)
  embed.py                        EMBED_CELLS (Cell-DINO)
  cp_features.py                  BUILD_CP_FEATURES (CellProfiler track's cell table)
  utils/cell_table.py             cell_table.parquet -> meta_* projection
  vendor/dinov2/                  vendored dinov2 subset
modules/local/                    PLAN_EXPERIMENTS, BUILD_CELL_IMAGES, BUILD_CELL_METADATA,
                                  EMBED_CELLS, BUILD_CP_FEATURES
conf/modules.config               ext.args / ext.seed / publishDir of the shared modules
snakemake/Snakefile               BUILD_CELL_IMAGES' nested run (starcall + make_cell_shard)
```

**Related code:**
- `fisseq-data-pipeline` (`packages/fisseq-data-pipeline/` in this workspace) — the
  CellProfiler-feature version of this same analysis, running the same shared stages.
- `fisseq-common` (`packages/fisseq-common/`) — `fisseq_common.stages` (algorithm, Hydra
  config and entry point of every shared stage) and their Nextflow modules.
- `starcall-workflow` — the Snakemake pipeline whose `origin/devel` branch
  produces this pipeline's raw input. This
  package's `Dockerfile` clones it at **one pinned commit**
  (`ARG STARCALL_WORKFLOW_COMMIT`): the code `BUILD_CELL_IMAGES`' nested
  snakemake actually runs, through `snakemake/Snakefile`, and what the
  image's `ops` env is built from. See `docs/embeddings-pipeline/architecture.md`'s Data
  contracts section and decision 24.

The workspace's `.devcontainer/devcontainer.json` bind-mounts starcall-workflow read-only
into the sandbox at `/workspaces/starcall-workflow`. It's mounted read-only on purpose —
don't write into it; if starcall needs a fix upstream, note that in your commit message.

**`starcall-workflow` gotcha:** this pipeline tracks `starcall-workflow`'s
`origin/devel` branch, not `master`. The authoritative version is the
Dockerfile's `STARCALL_WORKFLOW_COMMIT` — read starcall's code at that
commit (`git -C /workspaces/starcall-workflow show
<commit>:workflow/rules/phenotyping.smk`). The bind-mounted sibling's own
checkout may be on a different branch (check `git -C
/workspaces/starcall-workflow branch --show-current` before trusting its
working tree; on `master`, the phenotyping layout this pipeline depends on
— `workflow/rules/phenotyping.smk`'s whole-tile outputs under
`phenotyping_dir` — won't be there at all). starcall is never patched:
anything this pipeline adds to starcall's DAG goes in
`snakemake/Snakefile` instead. Upstream's own
`make_cell_images` is broken against its cell table, which is why the
per-tile `make_cell_shard` rule there does the cropping itself
(`tile_shard.py`; `docs/embeddings-pipeline/architecture.md` decision 17). Bumping the pin:
change `STARCALL_WORKFLOW_COMMIT`'s default to the new devel commit,
rebuild the image, and run `tests/integration --container`.

## Repo conventions

- **Python stages**: each cellDINO-specific `src/fisseq_embeddings_pipeline/<stage>.py` is a
  Hydra entry point invoked as `python -m fisseq_embeddings_pipeline.<stage>`, with a
  `@dataclasses.dataclass class <Stage>Config(AppConfig)` registered via `ConfigStore`.
  Don't add a module for a stage the data pipeline also runs: change the shared one in
  `fisseq_common.stages` (both pipelines' outputs move with it), or add a config field there
  and set it in `conf/modules.config`.
- **Every config extends `AppConfig`** (`fisseq_common.stages.config`, re-exported by
  `config/__init__.py`), which carries the one shared `random_seed` field every stochastic
  stage reads from — never add a stage-local `random_state`/seed field.
- **polars, not pandas**, for all tabular data except where the pipeline
  explicitly uses pandas (`build_cell_images_table.py`'s per-tile CSV
  reads — matching `starcall-workflow`'s own CSV-reading convention there;
  it still writes its final `cell_table.parquet` via polars, though, to
  keep everything downstream of `BUILD_CELL_IMAGES` in the usual
  convention; `tile_shard.py` reuses its `read_segmentation_table` rather
  than reading the same CSV another way).
- **`meta_*` column convention**: metadata columns are prefixed `meta_*`;
  `FEATURE_SELECTOR` (`cs.exclude("^meta_.*$")`) and `EMBEDDING_SELECTOR`
  (`cs.matches(r"^emb_\d+$")`) key off this — see
  [`docs/embeddings-pipeline/api/utils.md`](../../docs/embeddings-pipeline/api/utils.md) before adding any new
  non-`meta_*` column.
- **No stage copies another stage's data wholesale.** If you're about to
  write a full copy of another stage's table to disk (rather than a join
  key + something new), stop and check whether that violates the no-copy
  principle — see `docs/embeddings-pipeline/architecture.md`'s architecture decisions.
- **Nextflow processes**: this pipeline's own in `modules/local/<name>/main.nf`; the shared
  ones in `packages/fisseq-common/nextflow/modules/local/<stage>/main.nf`, included under the
  data pipeline's process names and configured only by `conf/modules.config` (`ext.args`:
  `join_keys`, `feature_selector`, ...; `ext.seed`; `publishDir`, which must match
  `fisseq_common.layout.EmbeddingsPipelineLayout` — `tests/unit/test_publish_layout.py`
  checks). All wired together in `workflows/embeddings.nf`. Each carries
  `errorStrategy 'ignore'`, a `process_*` label, `container "${params.container_image}"`, a
  `publishDir ..., mode: 'copy'` into `pipeline_dir`, and a `script:` of
  `${threadEnv(task.cpus)}` (fisseq-common's `functions.nf`) plus one `python -m` invocation
  with `output_dir=.` and a trailing `random_seed=...` — see `EMBED_CELLS` for the
  worked example, and `BUILD_CELL_IMAGES` for the one genuine exception (a nested starcall
  `snakemake`, run against `snakemake/Snakefile`, whose `make_cell_shard` rule calls
  `python -m fisseq_embeddings_pipeline.tile_shard` once per tile). `PLAN_EXPERIMENTS` runs
  first and owns validation/routing (`config/experiments.py`) — add new per-experiment
  routing there, in Python, not in Groovy. See
  [`docs/embeddings-pipeline/nextflow.md`](../../docs/embeddings-pipeline/nextflow.md#modules).
- **No scheduler-specific code.** Cluster settings are the user's: a
  `-c site.config` for Nextflow and a snakemake 7 `starcall_profile` for
  the nested starcall run. Only the generic image re-entry jobscript
  (`render_starcall_jobscript`) lives here.
- **Config**: defaults belong in `params.yaml` (package root), never in
  `nextflow.config` or a profile — see
  [`docs/embeddings-pipeline/configuration.md`](../../docs/embeddings-pipeline/configuration.md).
  The parameters the shared stages read have the data pipeline's names and defaults; keep
  them identical in both `params.yaml` files. A renamed parameter goes in `RENAMED_PARAMS`
  (warned about and ignored), a removed one in the removed-param warnings.

## Gotchas

- **Cell identity is `(meta_batch, meta_well, meta_tile, meta_cell_index)`**:
  `meta_cell_index` is per tile. Every shared process reading the normalized cells needs
  `join_keys` set to that in `conf/modules.config`; the shared default is the data
  pipeline's `(meta_cell_index, meta_variant_tag)`. Rows are sorted and split on
  `row_keys(join_keys)`, which adds `meta_variant_tag` so QC pseudo-variant rows stay
  distinct from their source cells.
- **`feature_selector`**: `embeddings` (`emb_NNNN` only) on the cellDINO track, `features`
  on the CP track. The embeddings table has no other non-`meta_` columns today, but don't
  rely on that.
- **Controls are the wildtype cells** (NORMALIZE), and every per-method aggregate is
  z-scored against the synonymous variants: an experiment needs at least two synonymous
  variants, or those columns come out null.
- **The CP track has no feature selection**: no splits, blocklists or `output.parquet`
  under `feature_select_batchwise_cp_features/`, only `aggregates/`.
- **Output changes are deliberate**: the root `tests/test_reference_outputs.py` compares
  every published parquet with `tests/reference/embeddings*/`. A change to a shared stage
  moves both pipelines' outputs; regenerate the references and add a diff report (root
  `AGENTS.md`).

## Git workflow

`main` is this repo's default/integration branch, with a GitHub remote —
see **CI** below for what runs where.

- **One branch per unit of work.** Branch off `main`:
  `git checkout -b <short-slug>`. Merge back to `main` when the work is
  done and tested (see below), then delete the branch.
- **Before every commit**, run the relevant tests and linters and don't
  commit a red state:
  ```bash
  uv run pytest tests/unit
  uv run ruff check --fix . && uv run ruff format .
  ```
  (Full commands in **Testing** below — `tests/integration` doesn't need
  to pass for every single commit, but do run it before merging a branch
  back to `main`.)
- **Finishing a branch:** once `tests/unit` (plus `tests/integration` if
  the branch touches workflow wiring) pass, merge it back to `main`
  (`git checkout main && git merge --no-ff <slug>`; `--no-ff` keeps the
  branch boundary visible in `git log --graph`). Delete the branch after
  merging.
- **Never rewrite history already merged into `main`** (no
  `push --force`/rebase of a merged branch) — even solo, this keeps
  `git log` a trustworthy record of what happened and when.
- If `docs/embeddings-pipeline/` diverges from what actually ended up on disk, fix the docs in
  the same commit as the code change that caused the divergence —
  `docs/embeddings-pipeline/` should always reflect what's actually implemented, never what's
  planned or in progress.

## Testing

```bash
uv run --package fisseq-embeddings-pipeline pytest packages/fisseq-embeddings-pipeline/tests/unit
uv run --package fisseq-embeddings-pipeline pytest packages/fisseq-embeddings-pipeline/tests/integration   # real `nextflow run -profile local`; needs nextflow + java
uv run --package fisseq-embeddings-pipeline pytest packages/fisseq-embeddings-pipeline/tests/integration --container   # real starcall; needs docker + testing_data/lmna_t3_mini
uv run pre-commit run --all-files
```

`tests/unit/` has one test module per cellDINO-specific stage, plus
`test_experiments.py` (params validation/routing) and `test_publish_layout.py`; the shared
stages' unit tests are fisseq-common's. `tests/integration/test_integration.py` is a
synthetic fixture, a
`subprocess`-driven end-to-end `nextflow run -profile local`, and
output-file/column assertions. BUILD_CELL_IMAGES' nested `snakemake` is a
stub on PATH that records its argv; the fixture pre-writes the
starcall-shaped outputs it would have produced — per-tile cell/reads
tables plus each tile's shard, cut by `tile_shard.write_tile_shard` (the
same code `make_cell_shard` runs).

`EMBED_CELLS`' GPU/checkpoint dependency is handled in the integration
fixture by building a tiny, from-scratch, randomly-initialized
`vit_small` checkpoint (`_write_tiny_checkpoint`) and running
`EMBED_CELLS` against it with `device=cpu`, rather than stubbing
`load_cell_dino` out entirely. No GPU or real checkpoint is needed to run
`tests/integration` anywhere, including CI.

`--container` is mutually exclusive with the synthetic suite (see
`tests/integration/conftest.py`) and runs only the `container`-marked
tests: real starcall-workflow — the pinned commit cloned into a real
build of the root Dockerfile — plus
the real `make_cell_shard` rule (under Docker), on the tiny
`testing_data/lmna_t3_mini/` fixture
(`uv run python scripts/prepare_real_starcall_test_data.py --minimal`),
stopping after `EMBED_CELLS`. Each experiment's `starcall_workflow_dir` is
just the fixture's `config.yaml` plus its `input/` tree; no starcall
checkout is cloned. `test_real_starcall_local` runs the nested
starcall in local mode; `test_real_starcall_profile_mode` runs it through
a throwaway `starcall_profile` with a fake cluster and fake container
runtime — the real `apptainer` re-entry on a compute node is only
verifiable on a real cluster. Self-skips without the fixture or `docker`;
`FISSEQ_TEST_IMAGE=<tag>` skips the image build; not run in CI. Docker
Desktop's file-sharing allowlist can silently block bind-mounting this
repo's temp directories (see the test module's container-section comment).

## CI

CI (the workspace's `.github/workflows/`; see the root `AGENTS.md`):
- `pkg-<package>.yml` — that package's unit tests, alone in its own venv, on a PR that
  changes it or `fisseq-common`; a manual run (`workflow_dispatch`) adds the integration
  tests
- `root.yml` — lint, `uv lock --check` and Nextflow lint on every PR; a manual run adds the
  cross-package tests (`tests/`)
- `docker-fisseq-embeddings-pipeline.yml` (via the reusable `_docker.yml`) — builds this
  image (never on PRs) and pushes `:latest` + `:<short-sha>` from `main`, `:<version>` from a `v*`
  tag; its `paths` filter lists exactly the files the `Dockerfile` copies (Nextflow files,
  tests and docs never reach the image), and every tag push builds
- `docs.yml` — builds and deploys the one MkDocs site on push to `main`

Release with `scripts/release.py X.Y.Z` (one version for the whole workspace), then tag
`vX.Y.Z`.
The image tagging convention is in
[`docs/embeddings-pipeline/configuration.md`](../../docs/embeddings-pipeline/configuration.md#docker-image-versioning-publishing).

## Docker / devcontainer

`Dockerfile` (this package's) is the single image every task runs
in — build it locally from the workspace root with `docker build -f
packages/fisseq-embeddings-pipeline/Dockerfile -t fisseq-embeddings-pipeline:latest .`
and point `params.yaml`'s
`container_image` at wherever you publish it (see
[`docs/embeddings-pipeline/configuration.md`](../../docs/embeddings-pipeline/configuration.md)). The workspace's
`.devcontainer/` is a Claude-Code-in-a-sandbox dev environment — not the
pipeline's runtime container, just where you edit code.
