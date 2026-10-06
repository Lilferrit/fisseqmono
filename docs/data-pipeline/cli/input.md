# Input

`python -m fisseq_data_pipeline.input` (Nextflow process `INPUT`, runs once per
`experiments:` entry — see
[Configuration](../configuration.md#declaring-experiments))
reads a YAML config describing one or more raw cell-score files
(CSV or Parquet) and merges them into a single `input/`-ready cell-level
Parquet file. The Nextflow process writes that YAML from the experiment's entry. INPUT is the
only stage this package owns; every later stage is a
[shared stage](../../common/stages.md).

Variant-class/count-based restriction (previously done here via
`top_n_missense`) now happens downstream, in `QC_FILTER`'s `n_variants` /
`variant_downsample_classes` / `variant_downsample_mode` (see
[Shared stages: qcfilter](../../common/stages.md#qcfilter)), so it applies uniformly to every
batch.

## Config fields

Extends the common `output_dir` / `output_root` / `log_level` fields (see
[Common config fields](#common-config-fields) below).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `config_path` | **required** | Path to a separate YAML file (see below) describing the input files and variant selection behavior. Parsed independently of the Hydra CLI config. |

### `config_path` YAML schema

```yaml
input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
feature_allowlist_file: null      # optional, default null (no allowlist)
feature_blocklist_file: null      # optional, default null (no blocklist)
csv_schema_scan_rows: 100         # optional, default 100
```

- `input_paths` — one or more raw cell-score files (CSV or Parquet), concatenated.
  **Required, and per-experiment only** — there is no pipeline-wide default for a
  per-experiment list of raw data files (see
  [Declaring experiments](../configuration.md#declaring-experiments)).
- `feature_allowlist_file` / `feature_blocklist_file` — optional paths to plain
  text files, one fnmatch-style glob pattern per line (e.g.
  `Cells_AreaShape_*`), matched against feature column names. If an allowlist
  is given, only feature columns matching at least one of its patterns are
  kept; if a blocklist is also given, matching columns are then dropped from
  what remains (allowlist is applied first). Identity columns (`upBarcode`,
  `editDistance`, `aaChanges`) and metadata columns are unaffected.
- `csv_schema_scan_rows` — optional, number of rows scanned from each CSV
  `input_paths` source to infer column dtypes (forwarded to polars
  `scan_csv`'s `infer_schema_length`). Default `100`. `null` scans every row
  instead — slower, but avoids mis-inferred dtypes on columns whose
  non-null/non-integer values only appear after the scanned prefix. Has no
  effect on parquet sources.
Except for `input_paths`, every field above is also a
plain `params.yaml` pipeline-wide default (`params.feature_allowlist_file`,
`params.feature_blocklist_file`, `params.csv_schema_scan_rows`) — set one on
the command line or in `params.yaml` to apply it to every experiment, and/or
override it for one experiment in its `experiments:` entry. See
[Declaring experiments](../configuration.md#declaring-experiments).

## Output files

Written to `output_dir`, prefixed `{output_root}.` when `output_root` is set:

- `output.parquet` — the selected/filtered cells. The Nextflow process publishes it as
  `<pipeline_dir>/input/<batch_stem>.parquet`.

## Example

```bash
uv run python -m fisseq_data_pipeline.input \
    output_dir=./out \
    config_path=configs/batch1.yaml
```

## Common config fields

Every config extends `AppConfig` (`output_dir`, `output_root`, `log_level`, `random_seed`); see
[Shared stages: Common config fields](../../common/stages.md#common-config-fields).

See [API Reference: input](../api/input.md) for full function documentation.
