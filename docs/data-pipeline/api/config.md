# API: config

Hydra structured config hierarchy shared by every entry point:
`AppConfig` → `InputConfig` → `LabeledInputConfig`. `fisseq_data_pipeline.config`
re-exports these three; they live in `fisseq-common`, shared with the embeddings pipeline, next
to `CellsInput` (the inputs of the stages that read the normalized cells). See
[Shared stages: Common config fields](../../common/stages.md#common-config-fields).

See [`fisseq_common.stages.config`](../../common/api.md#fisseq_common.stages.config) (fisseq-common).
