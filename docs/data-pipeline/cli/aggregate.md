# Standalone aggregate

`python -m fisseq_data_pipeline.aggregate` aggregates a cell-level table to one row per
variant with one aggregator, z-scores the result against the synonymous variants, and attaches
per-variant metadata and an impact score. It is not wired into the Nextflow workflow: the
pipeline aggregates with the shared stage
[`fisseq_common.stages.aggregate`](../../common/stages.md#aggregate), one task per method, and
FINALIZE_FEATURE_SELECT joins the methods.

The aggregators, feature chunking and the synonymous z-score are the shared stage's; see
[Shared stages: aggregate](../../common/stages.md#aggregate). The aggregators exclude the
control rows, so the input needs a boolean `meta_is_control` column.

## Config fields

Extends `LabeledInputConfig` (`input_file`, `label_column`) and the
[common config fields](../../common/stages.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data (read with `load_batches`; each file is one batch, `meta_batch` = its stem). |
| `label_column` | `"meta_aa_changes"` | Variant label column. |
| `aggregator` | **required** | One of `mean`, `median`, `MAD`, `std`, `KS`, `signedKS`, `QQ`, `AUROC`, `KSnegLogP`, `AUROCnegLogP`. |
| `block_list_file` | `null` | Parquet with `feature` and `feature_ok` columns; a statistic whose `feature_ok` is false is not computed. |
| `compute_impact_score` | `true` | Add `meta_impact_score` (cosine distance from the synonymous variants' median). |
| `save_normalizer` | `true` | Also write the synonymous-fitted normalizer, `normalizer.parquet`. |
| `feature_chunk_size` | `32` | Feature columns per Polars query; `null` disables chunking. See [Feature chunking](../../common/stages.md#feature-chunking). |

## Output

- Glob input: `{output_dir}/output.parquet`, or `{output_root}.output.parquet`.
- Single-file input: `{output_dir}/<stem>.parquet`, or `{output_root}.<stem>.parquet`.
- With `save_normalizer`: `normalizer.parquet` (or `{output_root}.normalizer.parquet`).

`output_root` takes priority over `output_dir`: the prefixed file is written relative to the
working directory.

## Example

```bash
uv run python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    aggregator=KS
```

See [API Reference: aggregate](../api/aggregate.md) for the function documentation.
