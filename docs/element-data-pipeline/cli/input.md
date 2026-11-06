# Input (INPUT)

```bash
uv run python -m fisseq_element_data_pipeline.input \
    output_dir=./out \
    config_path=/path/to/config.yaml
```

`config_path` is a YAML file:

```yaml
input_paths: [/path/to/cellprofiler_Cells_with_variants.parquet]
feature_allowlist_file: null      # optional: one fnmatch pattern per line
feature_blocklist_file: null      # optional: one fnmatch pattern per line
csv_schema_scan_rows: 100         # optional, CSV sources only
target_col: MostAbundantTarget    # optional
variant_col: variant              # optional
wt_label: WT                      # optional
```

Writes `output.parquet` (`<output_root>.output.parquet` with `output_root` set): the feature
columns, `upBarcode`, `editDistance`, `aaChanges`, then the `meta_*` columns. Cells with a null
target or variant are dropped. See the [API reference](../api/input.md) for details.

NaN and ±inf in every float column (features and metadata) are written as null. Only values
that load as floats are affected: polars reads `NaN`, `inf`, `+inf`, `-inf` and overflowing
numbers like `1e999` in a CSV as floats, but other spellings (`nan`, `Inf`, `Infinity`, `NA`,
...) make the column String, and it stays String.
