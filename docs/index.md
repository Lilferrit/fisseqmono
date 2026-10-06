# FISSEQ

One repository, [`Lilferrit/fisseqmono`](https://github.com/Lilferrit/fisseqmono), a
[uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) of four packages:

| Package | What it is |
|---|---|
| [fisseq-common](common/index.md) | Code shared by the other three: the column schema, variant classification, the normalizer, the per-experiment output layout, cross-experiment aggregation, and (with the `stages` extra) the pipeline stages both pipelines run. |
| [fisseq-data-pipeline](data-pipeline/index.md) | Nextflow pipeline scoring variants on CellProfiler features, per experiment. |
| [fisseq-embeddings-pipeline](embeddings-pipeline/index.md) | Nextflow pipeline scoring variants on Cell-DINO embeddings (plus a CellProfiler track), per experiment. |
| [fisseqborn](fisseqborn/index.md) | Plots, and aggregation across experiments (`fisseqborn-global`), of either pipeline's outputs. |

Both pipelines produce per-experiment outputs only; `fisseqborn-global` pools experiments.
The [diff reports](diff-reports/README.md) record every deliberate change to a pipeline's
outputs made while merging the repositories.

## Working in the repository

```bash
uv sync                                    # every package + test/lint tools (the root dev group)
uv run pytest tests                        # cross-package tests
uv run --package fisseq-data-pipeline pytest packages/fisseq-data-pipeline/tests
uv run ruff check . && uv run ruff format --check .
uv run --group docs mkdocs serve           # this site
```

All four packages share one version. `scripts/release.py X.Y.Z` sets it everywhere (and the
`fisseq-common` git URL each package depends on); tag the commit `vX.Y.Z`.
