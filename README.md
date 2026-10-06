# fisseqmono

The FISSEQ analysis code, as one [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
of four packages (documentation: <https://lilferrit.github.io/fisseqmono/>):

| Package | |
|---|---|
| [`fisseq-common`](packages/fisseq-common) | Shared code: column schema, variant classification, the normalizer, the pipelines' output layout, cross-experiment aggregation, and (`[stages]` extra) the per-experiment stages both pipelines run. |
| [`fisseq-data-pipeline`](packages/fisseq-data-pipeline) | Nextflow pipeline scoring variants on CellProfiler features, per experiment. |
| [`fisseq-embeddings-pipeline`](packages/fisseq-embeddings-pipeline) | Nextflow pipeline scoring variants on Cell-DINO embeddings (plus a CellProfiler track), per experiment. |
| [`fisseqborn`](packages/fisseqborn) | Plots, and cross-experiment aggregation (`fisseqborn-global`), of either pipeline's outputs. |

```text
packages/<package>/          # src/, tests/{unit,integration}/, pyproject.toml (+ pipelines: Nextflow, Dockerfile)
nextflow/modules/local/      # the Nextflow processes both pipelines run (one copy each)
tests/                       # cross-package tests: reference outputs, layout, global parity, release
docs/                        # the one MkDocs site (mkdocs.yml); docs/diff-reports/: output changes
scripts/release.py           # set the one workspace version
```

```bash
uv sync                                                    # every package + dev tools
uv run pytest tests                                        # cross-package tests (needs Nextflow)
uv run --package fisseqborn pytest packages/fisseqborn/tests
uv run ruff check . && uv run ruff format --check .
uv run --group docs mkdocs serve
docker build -f packages/fisseq-data-pipeline/Dockerfile -t fisseq-data-pipeline .
```

Install one package outside the workspace from its tag, e.g.
`pip install "fisseqborn @ git+https://github.com/Lilferrit/fisseqmono@v2.0.0#subdirectory=packages/fisseqborn"`.
