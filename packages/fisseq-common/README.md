# fisseq-common

Code shared by `fisseq-data-pipeline`, `fisseq-embeddings-pipeline` and `fisseqborn`.

The base install (polars, pyarrow, numpy) covers the column schema, the per-experiment output
layout, variant classification and the cross-experiment aggregation helpers. The `stages` extra
adds the dependencies of the per-experiment pipeline stages (`fisseq_common.stages`).
