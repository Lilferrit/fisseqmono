# Normalize

`python -m fisseq_data_pipeline.normalize` (Nextflow process `NORMALIZE`) fits per-feature z-score
statistics on control (WT) cells. Control rows are identified via a SQL WHERE clause evaluated
against the input frame, making it easy to adapt to non-standard control labels without code
changes. Features with zero variance are stored as `null` in the normalizer.

The stage writes no normalized copy of the cells. It publishes the QC-passed cells' keys and the
fitted normalizer, and every downstream stage (OvWT, the feature-type aggregations, the
bootstrap splits, the final feature selection) rebuilds the normalized table from QC_FILTER's
`filtered_cells.parquet` and these two files (`fisseq_data_pipeline.cells`). The stage itself is
shared with fisseq-embeddings-pipeline (`fisseq_common.stages.filter`); there the controls are
untagged synonymous variants instead of wildtype cells.

## Config fields

Extends the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | QC_FILTER's `filtered_cells.parquet`. |
| `control_sample_query` | `"meta_aa_changes = 'WT'"` | SQL-like WHERE clause identifying control rows used to fit the normalizer. |
| `batch_name` | the input file's stem | The experiment's name, stored as `meta_batch` in `filtered_keys.parquet`. Nextflow passes the `batch_stem`. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `save_normalizer` | `true` | Deprecated and ignored: `normalizer.parquet` is always written. |

## Output files

With `prefix` = `{output_root}.` when `output_root` is set:

- `{prefix}filtered_keys.parquet` — every `meta_*` column of the QC-passed cells, plus
  `meta_is_control` and `meta_batch`; no feature columns. Sorted on
  `(meta_cell_index, meta_variant_tag)`, the cell identity the consumers join on.
- `{prefix}normalizer.parquet` — the fitted normalizer.

Nextflow publishes both to `normalization/<batch_stem>/`.

## Example

```bash
uv run python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=out/qc_filter/batch1/filtered_cells.parquet \
    batch_name=batch1
```

See [API Reference: normalize](../api/normalize.md) for full function
documentation, including the reusable `Normalizer` class.
