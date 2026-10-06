# AGENTS.md — fisseq-data-pipeline

> This file plus the source and `pyproject.toml` are the authoritative references;
> `docs/data-pipeline/` (and, for the shared stages, `docs/common/stages.md`) is the
> user-facing version of the same material.

---

## Project overview

The **FISSEQ Data Pipeline** is a Nextflow + Python workflow for processing
single-cell CellProfiler morphological profiling data from FISSEQ (Fluorescence
In-Situ Sequencing) experiments. Each cell carries a genetic variant label; the
pipeline measures how each variant's cell population differs from wildtype (WT)
controls using morphological features.

**End-to-end data flow:**

```
params.yaml (experiments: [...])
      │
      ▼
   INPUT            (per experiment)  ──►  input/<batch_stem>.parquet
      │
      ▼
   QC_FILTER        (per experiment)   ← edit distance, barcode count,
      │                                   variant barcode count
      ▼
   NORMALIZE        (per experiment)   ← z-score fit on WT control cells
      │
      ├──► OVWT_BATCHWISE  (per experiment; params.run_ovwt)
      │              (fold scheme: params.ovwt_cv_mode,
      │               fold count/cap: params.ovwt_n_folds)
      │
      └──► Feature selection, batchwise (params.run_feature_selection):
             AGGREGATE_FEATURE_TYPE_BATCHWISE   (per type; synonymous z-score)
             AGGREGATE_FEATURE_TYPE_PASSTHROUGH (per passthrough type; raw)
             GENERATE_SPLIT_BATCHWISE           (per bootstrap replicate)
               └─► AGGREGATE_HALF_BATCHWISE     (per bootstrap × type × half)
                     └─► CORRELATE_FEATURES_BATCHWISE (per bootstrap × type)
                           └─► BLOCKLIST_BATCHWISE  (gathers all bootstraps —
                                                     the one sync point)
                                 └─► COMBINE_BLOCKLISTS_BATCHWISE (all types)
                                       └─► FINALIZE_FEATURE_SELECT_BATCHWISE
```

Only INPUT is this package's own. Every other stage is fisseq-common's, whole: the algorithm,
the Hydra config and the entry point `python -m fisseq_common.stages.<stage>`, run from the
shared Nextflow modules in `packages/fisseq-common/nextflow/modules/local/<stage>/main.nf`.
The embeddings pipeline runs the same stages, process names, params and publish layout.

There is a single pipeline mode — `main.nf` includes one workflow and runs it.
Every output is per experiment. Cross-experiment aggregation (combining
blocklists, per-variant medians across experiments, re-centering OvWT AUROCs
against synonymous variants) is done downstream by
fisseqborn (`fisseqborn-global`), which reads
`feature_select_batchwise/<batch>/{aggregates,passthrough_aggregates,blocklists}/<type>.parquet`
and `ovwt_batchwise/<batch>/results.parquet`.

**Main components:**
- `src/fisseq_data_pipeline/` — `input` (INPUT), the standalone `aggregate` entry point,
  `config`
- `modules/local/input.nf` — this pipeline's own Nextflow process; the shared ones are
  included from `../../fisseq-common/nextflow/modules/local/<stage>/main` and configured per
  process by `conf/modules.config` (`ext.args`, `ext.seed`, `publishDir`)
- `workflows/fisseq.nf` — the DAG
- `main.nf` — entry point
- `params.yaml` — every parameter default
- `nextflow.config` — executor/profile/container settings only
- `Dockerfile` — the single image every process runs in

---

## Setup & environment

- **Python 3.13** (pinned in the workspace's `.python-version`; `requires-python` is
  `>=3.13,<3.14` — Hydra 1.3.x crashes on 3.14's argparse)
- **uv** for dependency and environment management
- **Nextflow ≥ 26.04** to run the pipeline (not needed for Python-only work)
- **Docker** only to build the image

```bash
uv sync                      # from the workspace root: every package + the dev tools
uv run pre-commit install
```

This package lives in the fisseqmono uv workspace (`packages/fisseq-data-pipeline/`); its
shared stages and their Nextflow modules are in `packages/fisseq-common/`
(`src/fisseq_common/stages/`, `nextflow/modules/local/`). The workflow needs the whole
checkout.

---

## Build, run, and test commands

All Python commands run via `uv run` — never bare `python`, `pytest`, `ruff`.

```bash
# Install
uv sync --group dev

# Run the pipeline (-params-file is effectively mandatory)
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -profile local   # no container
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -resume

# Run one stage directly (every stage but INPUT is fisseq-common's)
uv run python -m fisseq_common.stages.ovwt output_dir=./out \
    cells_file=out/qc_filter/b1/filtered_cells.parquet \
    filtered_keys_file=out/normalization/b1/filtered_keys.parquet \
    normalizer_file=out/normalization/b1/normalizer.parquet

# Tests
uv run pytest tests/unit          # fast, no external deps
uv run pytest tests/integration   # slow, needs `nextflow` on PATH
uv run pytest                     # everything

# Lint & format
uv run ruff check .
uv run ruff format --check .
uv run ruff format .

# Nextflow lint (CI runs it in root.yml) — after touching any .nf or .config
nextflow lint . ../fisseq-common/nextflow

# Docs
uv run --group docs mkdocs build --strict
uv run --group docs mkdocs serve

# Container
docker build -f packages/fisseq-data-pipeline/Dockerfile -t fisseq-data-pipeline:dev .   # from the workspace root
```

---

## Code architecture

### Directory map

```
fisseq-data-pipeline/
├── src/fisseq_data_pipeline/
│   ├── config/__init__.py         # re-exports AppConfig, InputConfig, LabeledInputConfig
│   ├── input.py                   # INPUT
│   └── aggregate.py               # standalone entry point (not in the workflow)
├── conf/modules.config            # ext.args / ext.seed / publishDir of the shared modules
├── modules/local/input.nf         # INPUT
├── workflows/fisseq.nf
├── main.nf
├── params.yaml
├── nextflow.config
├── Dockerfile
└── tests/{unit,integration}/
```

The shared stages (`packages/fisseq-common/src/fisseq_common/stages/`): `qcfilter`
(QC_FILTER), `filter` (NORMALIZE), `ovwt` + `xgbparams` (OVWT_BATCHWISE), `aggregate`
(AGGREGATE_FEATURE_TYPE_BATCHWISE / _PASSTHROUGH / AGGREGATE_HALF_BATCHWISE), `generatesplit`,
`correlatefeatures`, `blocklist`, `combineblocklists`, `finalize`
(FINALIZE_FEATURE_SELECT_BATCHWISE), and `config` (base classes, `CellsInput`, `row_keys`,
`stage_main`). Their unit tests are in `packages/fisseq-common/tests/unit/`.

### Key abstractions

**Config** (`fisseq_common.stages.config`) — Hydra structured config hierarchy:
```
AppConfig  (output_dir, output_root, log_level, random_seed)
  └── InputConfig (adds input_file)
        └── LabeledInputConfig (adds label_column)
CellsInput (cells_file, filtered_keys_file, normalizer_file, join_keys, feature_selector)
```
Each shared stage's config extends these; `stage_main` registers it in the `ConfigStore` and
makes the entry point.

**What this pipeline sets** (`conf/modules.config` `ext.args`): QC_FILTER's raw starcall
column names (`upBarcode`, `aaChanges`, `editDistance`),
`sort_output_by=[meta_cell_index,meta_variant_tag]` and `assign_cell_index=true`; NORMALIZE's
`batch_name`; the aggregate processes' `downsample_wt` and `normalize_to_synonymous`;
BLOCKLIST's `minimum_correlation`; `ext.seed` for AGGREGATE_HALF_BATCHWISE. The stages'
defaults (`join_keys = [meta_cell_index, meta_variant_tag]`, `feature_selector = "features"`,
`control` = WT) are this pipeline's.

**`Normalizer`** (`fisseq_common.normalizer`) — fits per-feature z-score stats on a
LazyFrame and applies them. Stats persist to Parquet (not pickle) and reload via
`Normalizer.load`. Zero-variance features produce `null`. NORMALIZE publishes only
`filtered_keys.parquet` and `normalizer.parquet`; every later stage rebuilds the normalized
cells (`fisseq_common.stages.filter.load_cells`).

**`BaseAggregator`** (`fisseq_common.stages.aggregate`) — the aggregation strategies.
Combining feature types happens in Nextflow: AGGREGATE runs once per
`params.feature_select_types` entry — and once per `params.feature_select_passthrough_types`
entry, publishing to `passthrough_aggregates/` — and `finalize` joins the per-type outputs on
the label column. `aggregates/` are z-scored against the synonymous variants
(`normalize_to_synonymous=true`); `passthrough_aggregates/` and the bootstrap halves stay raw.

**`fisseq_common.stages.xgbparams`** — shared XGBoost infrastructure: `XGBoostParams` /
`XGBoostConfig`, `get_feature_cols`, `get_dmatrix`, `split_indices_stratified`,
`train_binary_xgboost`.

---

## Conventions

### Column naming

| Pattern | Meaning |
|---------|---------|
| `meta_*` | Metadata (barcode, batch, labels, QC flags, scores, cell index) |
| `UPPERCASE_WITH_UNDERSCORE` | CellProfiler feature columns |
| `tmp_*` | Ephemeral intermediates, dropped before output |

The `meta_` prefix is load-bearing: `FEATURE_SELECTOR` is
`cs.exclude("^meta_.*$")`, so any new non-`meta_` column is treated as a feature.

Key constants (`fisseq_common.schema`): `CONTROL_COLUMN_NAME`
(`meta_is_control`), `META_BARCODE_COL`, `META_BATCH_COL`,
`META_CELL_INDEX_COL`, `META_EDIT_DISTANCE_COL`, `META_VARIANT_TAG_COL`,
`IMPACT_SCORE_COL`, `FEATURE_SELECTOR`, `META_SELECTOR`.

### DataFrame conventions

- Always use `pl.LazyFrame`; `.collect()` only at I/O boundaries.
- Convert NaN → null with `.fill_nan(None)` before statistical operations.
- Null-containing rows/columns are excluded from aggregations intentionally.

### Configuration pattern

Each module defines a `@dataclasses.dataclass` config. The shared stages end with
`main = stage_main("<name>_main", Config, run)`; `input` and `aggregate` use
`@hydra.main(version_base=None, config_path=None, config_name="<name>_main")` directly.
Overrides are `key=value` CLI pairs (dot-notation for nested fields:
`xgboost.params.max_depth=5`).

### Logging

Every `main()` calls `setup_logging(cfg, name)` from `fisseq_common.utils.log` first.

### Error handling

- `ValueError` for bad inputs or configuration.
- Avoid bare `except` — the sanctioned exception is per-variant failure
  isolation in `fisseq_common.stages.ovwt.ovwt_batchwise`, which is load-bearing (see gotcha 4).

### Docstring style

NumPy style, enforced by mkdocstrings under `--strict`. Note that prose placed
inside a `Parameters` or `Attributes` block is parsed as a malformed parameter
entry and **fails the docs build** — put narrative text in a `Notes` section
instead.

### Commit style

Lowercase verb, optional scope, PR number in parentheses:
`fix NaN handling, performance improvements (#10)`.

---

## Gotchas & known issues

1. **There is exactly one seed.** `AppConfig.random_seed`. Never add a
   stage-local `random_state`/`seed` field — `tests/unit/test_config.py` (and
   fisseq-common's, for the shared stages) sweeps every config class and fails if one
   appears. A stage that must differ from its siblings derives a fixed offset
   (`GENERATE_SPLIT` uses `random_seed + bootstrap_idx`; `AGGREGATE_HALF_BATCHWISE`'s
   `ext.seed` is `random_seed + rep * 2 + half`).

2. **Row order is part of reproducibility.** Polars inner joins are not
   order-preserving under multithreaded execution. `QC_FILTER` assigns
   `meta_cell_index` over the raw input order and sorts on
   `(meta_cell_index, meta_variant_tag)` before publishing, and every stage that rebuilds
   the cells sorts on `row_keys(join_keys)` (the same two columns). Without that, the
   same input yields the same rows in a different order every run and every
   seeded step downstream silently diverges. If you add a stage that reorders
   published cell-level output, restore a deterministic order before writing.
   `meta_variant_tag` is part of the identity: a QC pseudo-variant row shares its source
   cell's `meta_cell_index` but has its own tag (`downsample-<amount>`), and the stages take
   `meta_*` columns from the QC side and features from the cell table.

3. **Never put a `params { }` block in `nextflow.config`**, not even an empty
   one. Combined with `-params-file`, Nextflow's `ConfigBuilder` before 26.04.6
   misattributes every `params.yaml` key into `process{}` scope and raises
   `Unknown config attribute 'params'`. All defaults live in `params.yaml`.

4. **`fisseq_common.stages.ovwt`'s per-variant `try/except` is load-bearing.** A variant can
   still be too small for the outer `StratifiedKFold` or produce a degenerate
   fold, even after `_stratification_key`'s rare-bucket collapse. Small
   variants legitimately hit this; the alternative is losing every other
   variant's results. fisseq-common's `test_variant_failure_is_isolated_not_fatal` asserts a
   failing variant is silently dropped while its peers succeed. (Singleton
   strata inside a fold are *not* one of these cases any more — see gotcha 5.)

5. **`split_indices_stratified` is a two-way split that tolerates singleton
   strata.** It splits each outer fold's fit rows 80/20 into train and
   calibration — no third slot. It was an 80/10/10 train/test/val split whose
   only caller used two of the three slots, silently discarding 10% of every
   fold. Rows in a 1-member stratum can't be stratified, so they are assigned
   to the train half with a logged warning rather than raising (the production
   `least populated class in y has only 1 member` failure) or being thrown
   away. Test fixtures still want comfortably more than `n_folds` cells per
   stratum: an undersized fixture produces degenerate folds, and the test then
   passes against near-empty output while asserting nothing.

6. **OvWT trains on WT-normalized features, not synonymous-normalized ones** (both
   pipelines). The synonymous re-centering happens on the AUROCs, downstream in
   fisseqborn. This is deliberate — do not "fix" it by adding a second normalizer fit in
   the OvWT stage.

7. **Two different control baselines.** NORMALIZE (`fisseq_common.stages.filter`, `control`)
   uses **WT** cells. `fisseq_common.stages.filter.variant_classification` flags
   **synonymous** variants as `meta_is_control`, which is what the aggregate stage's
   `normalize_to_synonymous` and `finalize` normalize against. (The aggregators' own
   reference group is still the WT `meta_is_control` from NORMALIZE; the synonymous flag is
   computed on the aggregated output.) Not interchangeable.

8. **Per-experiment synonymous normalization needs ≥2 synonymous variants.**
   `AGGREGATE_FEATURE_TYPE_BATCHWISE`'s `normalize_to_synonymous` and
   `FINALIZE_FEATURE_SELECT_BATCHWISE` both fit a normalizer on the experiment's
   synonymous rows; a single one makes std (ddof=1) undefined and nulls every
   feature. Test fixtures must include ≥2 synonymous labels (the integration
   fixture uses `A1A`/`A2A`/`A3A`).

9. **Never bind a Nextflow variable named `channel`.** It is a reserved
   lowercase alias for the `Channel` class and silently resolves to
   `nextflow.Channel` instead of failing. Use `chan`. (The
   `channel.fromList(...)` factory call is fine.)

10. **DSL2 forbids bare statements at script scope.** A top-level helper must be
    `def someFunction() { ... }`, not `def x = { ... }`.

11. **`.join()` is not a broadcast operator.** For a many-to-one key
    relationship it silently keeps one match per key and drops the rest. Use
    `.combine(other, by: 0)` to broadcast; reserve `.join()` for cases where
    both sides are already one-per-key.

12. **Pass real path channels to collecting processes, not glob strings.** A
    `val` glob hashes only the glob text, so `-resume` fails to invalidate when
    the underlying files change.

13. **`output_root` takes priority over `output_dir`** in the standalone `aggregate.py` —
    the output lands at `{output_root}.{stem}.parquet` regardless of `output_dir`. INPUT's
    process uses `output_root=<batch>` plus an `mv`; the shared modules pass `output_dir=.`
    and name per-method outputs with `output_name`. If you change output naming, update
    `conf/modules.config`'s `publishDir`/`saveAs` and `fisseq_common.layout` together.

14. **`errorStrategy 'ignore'` is on every process.** A failed stage drops that
    experiment rather than aborting the run, so a missing output may mean its
    task failed. Check the run log.

15. **OvWT has two cross-validation schemes**, `OvwtParams.cv_mode`
    (`params.ovwt_cv_mode`). `"kfold"` cuts `n_folds` folds stratified on
    `(meta_barcode, is_wt)`, so every fold's model has seen every barcode.
    `"barcode_holdout"` holds whole barcodes out of training a fold at a time
    (wildtype is still split across the folds) and skips single-barcode
    variants. There `n_folds` *caps* the folds instead of fixing them: `null`
    is one fold per barcode, an integer packs the barcodes into that many
    cell-count-balanced groups, and a value above the barcode count degrades
    back to one per barcode — so fold counts differ per variant, and
    `len(models[variant])` is not `n_folds`. Both modes emit the same columns,
    so downstream consumers are mode-blind — but `auroc_median_barcode` means
    *in-sample separability* under the first and *generalization to an unseen
    barcode* under the second. Never compare the two modes' numbers.
    `auroc_folds` (a `List(Float64)`, per-fold test AUROC) and
    `auroc_median_fold` score each fold under its own model only. Unlike
    `auroc_pooled`/`auroc_median_barcode`, they never mix scores from different
    fold models into a single ROC curve. `auroc_folds` is a non-`meta_` list
    column, so any downstream consumer (fisseqborn) must drop or handle it
    before normalizing `results.parquet` as features — the `Normalizer` would
    otherwise treat it as a feature.

16. **`workflows/fisseq.nf` duplicates two Python allowlists on purpose.**
    `aggregatorKeys()` mirrors `fisseq_common.stages.aggregate._AGGREGATORS` and
    `ovwtCvModes()` mirrors `fisseq_common.stages.ovwt.CV_MODES`, because both values are interpolated straight
    into a process's shell script — a malformed entry breaks the generated
    `.command.sh` at the bash level before any Python validation can fire, and
    `errorStrategy 'ignore'` then hides it. `tests/unit/test_nextflow_params.py`
    parses those two functions and fails if either copy drifts; update both
    sides when you add an aggregator or a CV mode. `aggregatorKeys()` validates
    both `feature_select_types` and `feature_select_passthrough_types`, and the
    two lists must be disjoint.

17. **A `.join()` that may have an empty right side needs `remainder: true`.**
    `workflows/fisseq.nf`'s stage-4 `finalize_input_ch` joins the passthrough
    aggregates, and `params.feature_select_passthrough_types` defaults to `[]`
    — an empty channel. Without `remainder: true` that join emits nothing and
    `FINALIZE_FEATURE_SELECT_BATCHWISE` never runs for any batch, silently (see gotcha
    14). This is the complement of gotcha 11: `.join()` drops on the many side
    and starves on the empty side.

18. **`output.parquet` is not "the selected features" any more.**
    `params.feature_select_passthrough_types` puts non-`meta_` columns into
    `feature_select_batchwise/<batch>/output.parquet` that were never
    blocklisted or normalized — that
    is the entire point of the second list. Nothing in-pipeline reads that file
    (it is terminal), but any new stage that does must not assume
    `FEATURE_SELECTOR` over it yields selected features: its passthrough columns
    are raw and un-normalized, while the selected ones are synonymous-z-scored.
    The `aggregates/` vs `passthrough_aggregates/` publish split keeps
    normalized and raw-scale values apart for downstream readers (fisseqborn
    globs `aggregates/`) — never publish both into one directory.

19. **`.python-version` must be copied into the image before the first
    `uv sync`.** Without it uv resolves the newest `>=3.13` interpreter, and
    Hydra 1.3.x crashes on Python 3.14's argparse, breaking every stage's CLI.

---

## PR / commit workflow

```bash
git checkout main && git pull
git checkout -b <descriptive-branch-name>
# ... make changes, commit ...
git push -u origin <descriptive-branch-name>
```

Claude pushes the branch but does **not** open the PR — the user does that.

Before merging: `uv run pytest tests/unit`, `uv run ruff check .`,
`uv run ruff format --check .`, `nextflow lint . ../fisseq-common/nextflow`, and
`uv run --group docs mkdocs build --strict`.

CI (the workspace's `.github/workflows/`; see the root `AGENTS.md`):
- `pkg-<package>.yml` — that package's unit tests, alone in its own venv, on a PR that
  changes it or `fisseq-common`; a manual run (`workflow_dispatch`) adds the integration
  tests
- `root.yml` — lint, `uv lock --check` and Nextflow lint on every PR; a manual run adds the
  cross-package tests (`tests/`)
- `docker-fisseq-data-pipeline.yml` (via the reusable `_docker.yml`) — builds this image
  on a push to `main` that changes a file its Dockerfile copies (a `paths` filter; Nextflow
  files, tests and docs never reach the image) and on every `v*` tag, never on PRs; pushes `:latest` + `:<short-sha>` from
  `main` and `:<version>` from a tag
- `docs.yml` — builds and deploys the one MkDocs site on push to `main`

Release with `scripts/release.py X.Y.Z` (one version for the whole workspace), then tag
`vX.Y.Z`.

---

## Documentation maintenance

- Any change to CLI flags / Hydra config fields / Nextflow processes / module
  responsibilities **must** update the relevant `docs/data-pipeline/` page in the same change.
  Start from `docs/data-pipeline/architecture.md`, `docs/data-pipeline/nextflow.md`,
  `docs/data-pipeline/configuration.md`, and `docs/common/stages.md` for a shared stage (or
  `docs/data-pipeline/cli/<module>.md` + `api/<module>.md` for `input` / `aggregate`).
- Any new source file needs a file-level docstring (Python) or top `//` comment
  block (`.nf`), except `__init__.py`.
- Adding or removing a module means updating the repository root's `mkdocs.yml` nav — `docs.yml`
  runs `mkdocs build --strict`, so a dangling reference fails the build.
- `README.md` stays a thin pointer (overview + quick start + docs link).

---

## Safety / do-not-touch list

| Path | Reason |
|------|--------|
| `site/` | Generated MkDocs output, gitignored. CI publishes it to `gh-pages`. |
| `uv.lock` | Auto-managed by uv. Edit `pyproject.toml`, then `uv sync`. |
| `.venv/` | Managed by uv. |
| `<pipeline_dir>/work/` | Nextflow task working directories. Delete only via `nextflow clean`. |
| Any `*.parquet` under `<pipeline_dir>/` | Pipeline output data. |
| `.github/workflows/docs.yml` | Deploys live documentation. |
