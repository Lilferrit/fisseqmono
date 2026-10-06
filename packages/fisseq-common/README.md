# fisseq-common

Code shared by `fisseq-data-pipeline`, `fisseq-embeddings-pipeline` and `fisseqborn`.

The base install (polars, pyarrow, numpy) covers the column schema, the per-experiment output
layout, variant classification and the cross-experiment aggregation helpers. The `stages` extra
adds the per-experiment stages both pipelines run (`fisseq_common.stages`, each an entry point
`python -m fisseq_common.stages.<stage>`) and their dependencies. Their Nextflow modules are in
this package's `nextflow/` directory.
