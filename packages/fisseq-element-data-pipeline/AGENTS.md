# AGENTS.md — fisseq-element-data-pipeline

A copy of [`fisseq-data-pipeline`](../fisseq-data-pipeline/AGENTS.md) for Element Biosciences
cell tables. Everything in that guide applies here (the DAG, conventions, gotchas, commands,
with `fisseq_element_data_pipeline` for `fisseq_data_pipeline`) except INPUT, below. There is
no standalone `aggregate` entry point.

`workflows/fisseq.nf`, `conf/modules.config`, `nextflow.config`, `main.nf` and `params.yaml`
are the data pipeline's apart from the INPUT keys. A change to one of them in the data
pipeline (a new param, a process, a publish path) usually belongs here too.

## INPUT (`src/fisseq_element_data_pipeline/input.py`, `modules/local/input.nf`)

Builds the data pipeline's INPUT output from an Element table, so QC_FILTER's
`conf/modules.config` args (`upBarcode`/`aaChanges`/`editDistance`) are unchanged:

- `upBarcode` = the whole `target_col` (`MostAbundantTarget`) string. Element target names
  are `<variant>_<NNNN>`, unique, with `|` → `_` and `*` → `.`; a multi-variant target
  (`A502L_R644H`) has no barcode id and is kept as its own barcode. Don't strip the variant
  prefix: QC_FILTER counts cells per barcode alone, so bare ids (`0007`) would collide across
  variants.
- `aaChanges` = `variant_col` (`variant`), `wt_label` relabelled `WT`.
- `editDistance` = 0 (Int64) for every cell.
- Null target or variant: dropped.
- `meta_*`: `origin_file`/`origin_row_idx`, `ELEMENT_METADATA_COLUMNS` and any other
  non-numeric column (`tile` is a String; left as a feature it would break the normalizer).

The per-experiment keys are `batch_stem`, `input_paths`, `feature_allowlist_file`,
`feature_blocklist_file`, `csv_schema_scan_rows`, `target_col`, `variant_col`, `wt_label`
(`experimentKeys()` in `workflows/fisseq.nf`).

## Tests and CI

```bash
uv run --package fisseq-element-data-pipeline pytest packages/fisseq-element-data-pipeline/tests
```

The integration fixture writes Element-format tables. This package lives on the
`element-pipeline` branch, which is not merged into `main`, so its CI runs on every push to
that branch: `pkg-fisseq-element-data-pipeline.yml` (unit and integration tests),
`docker-fisseq-element-data-pipeline.yml` (the image, `:latest` and `:<sha7>`, no paths
filter, so every branch commit has an image) and `root.yml` (lint, lockfile, Nextflow lint).
Docs: `docs/element-data-pipeline/`.
