# API: config

Hydra structured config hierarchy shared by every entry point:
`AppConfig` → `InputConfig` → `LabeledInputConfig`. See
[Architecture: Key abstractions](../architecture.md#components) for how
these compose.

These live in `fisseq-common`, shared with the embeddings pipeline.

::: fisseq_common.stages.config
