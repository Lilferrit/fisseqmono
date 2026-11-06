# TODO

## Storage: keep cells × features in one table per track

Goal: the only cells × features table on disk is the one the ingest stage writes. Every later
stage writes keys (row identity + `meta_*`), fitted parameters or per-variant tables, and
rebuilds the normalized cells on demand (`fisseq_common.stages.filter.load_cells`).

Already done: NORMALIZE writes only `filtered_keys.parquet` and `normalizer.parquet`
(per-feature mean/std), and the embeddings track's QC_FILTER reads the meta-only
`metadata.parquet`.

Each item changes published outputs: regenerate `tests/reference/` and add a
`docs/diff-reports/` entry showing the aggregates, OvWT `results.parquet` and `output.parquet`
are unchanged.

### 1. Data pipeline: QC_FILTER writes keys only (largest saving)

`qc_filter/<batch>/filtered_cells.parquet` copies INPUT's whole cell table, plus another full
row for every pseudo-variant.

- [ ] INPUT (`fisseq_data_pipeline/input.py`, `select_output_columns`): write the identity
      columns as `meta_barcode`, `meta_aa_changes` (unsplit) and `meta_edit_distance`, not
      `upBarcode`/`aaChanges`/`editDistance`. Otherwise the `"features"` selector picks them up
      as features. Assign `meta_cell_index` (Int64, input row order).
- [ ] QC_FILTER (`fisseq_common/stages/qcfilter.py`, `filter_columns`): keep `meta_*` columns
      only (drop the CellProfiler-column rule). The lazy scan then reads only those columns.
- [ ] Data pipeline `conf/modules.config`: QC_FILTER uses the default column names and
      `assign_cell_index=false`.
- [ ] Join on `[meta_cell_index]`: make it the shared CellsInput / `DATA_JOIN_KEYS` default
      (`fisseq_common.stages.config`). `row_keys` still adds `meta_variant_tag`, so a
      pseudo-variant row joins its source cell's features, as in the embeddings pipeline.
- [ ] `workflows/fisseq.nf`: `norm_ch` carries INPUT's parquet as `cells`. NORMALIZE gets
      `(batch_stem, input_parquet, filtered_cells)`.
- [ ] Docs and tests: the `layout.filtered_cells` docstring, `docs/common/stages.md` (the
      `cells_file` field and the examples), the INPUT docs and
      `test_cells_input_defaults_to_the_data_join_keys`.

### 2. Embeddings CP track: read `cell_table.parquet` directly

`cp_features/<batch>/cp_features.parquet` copies `cell_table.parquet`'s CellProfiler columns.

- [ ] Drop BUILD_CP_FEATURES (`modules/local/build_cp_features`, `cp_features.py`).
      NORMALIZE_CP_FEATURES, OVWT_BATCHWISE_CP_FEATURES and
      AGGREGATE_FEATURE_TYPE_CP_FEATURES take `cell_table.parquet` with
      `feature_selector=features`.
- [ ] `cell_table.parquet` has no `meta_batch`, so set `join_keys` on the CP processes to
      `[meta_well, meta_tile, meta_cell_index]` (one batch per file). Check that the key
      dtypes in `CELL_META_SCHEMA` and `CELL_METADATA_SCHEMA` match.
- [ ] Layout: `EmbeddingsPipelineLayout(track="cp_features").features()` →
      `cell_images/<batch>/cell_table.parquet`. Also update `_MARKERS`,
      `test_publish_layout.py`, `docs/embeddings-pipeline/` and the Docker workflow's `paths`
      filter.

### 3. OvWT `cell_scores.parquet`: keys + score only

It holds every `meta_*` column per scored row, and every wildtype cell once per variant.

- [ ] `stages/ovwt.py`: write `row_keys(join_keys)` + `score` +
      `meta_variant_scored_against` and drop the other `META_SELECTOR` columns. Consumers
      rejoin `filtered_keys.parquet`. fisseqborn doesn't read this file today.
- [ ] Maybe: a `params.ovwt_save_models` gate for `models.pkl` (variants × folds boosters).

### 4. Bootstrap splits: no per-replicate key files

`splits/bootstrap_<N>/half{1,2}.parquet` add up to cells × reps × keys.

- [ ] Compute the half inside AGGREGATE from `filtered_keys` with seed
      `random_seed + bootstrap_idx` (replacing `split_file` with `bootstrap_idx`/`half`).
      Then drop GENERATE_SPLIT and `splits/` from both pipelines and from the layout.
      Alternative: one file per experiment with one Int8 column per replicate.

### Run settings (no code)

- Keep `publish_mode: link`: `copy` doubles every published file against `work/`.
- `nextflow clean -f -k` after a successful run (or `cleanup = true` for runs that won't be
  resumed).
- Maybe: cast CellProfiler features to Float32 in INPUT. This halves the remaining table but
  changes numerics, so it needs its own diff report.
