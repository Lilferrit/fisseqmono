# Feature-selection stages switch to the shared implementations

**Commit:** sharing GENERATE_SPLIT, CORRELATE_FEATURES, BLOCKLIST, COMBINE_BLOCKLISTS and the
blocklist / passthrough / pycytominer steps of FINALIZE_FEATURE_SELECT.

**Code differences (data pipeline).**

1. GENERATE_SPLIT writes each half as cell keys `(meta_cell_index, meta_variant_tag)` instead
   of positional row indices (`tmp_cell_idx`), and AGGREGATE_HALF selects a half with a
   semi-join on those keys. The seed is `random_seed + bootstrap_idx` either way, now computed
   in Python. Because NORMALIZE's keys file lists the cells in the old normalized table's order,
   the same seed picks the same cells.
2. CORRELATE_FEATURES stores an undefined (NaN) correlation as null, and returns an empty table
   when there are no feature columns.
3. BLOCKLIST blocks a feature whose median correlation is null (before: `feature_ok` was null,
   and FINALIZE's `filter(~feature_ok)` then kept the feature). BLOCKLIST and
   COMBINE_BLOCKLISTS sort their output by `feature`.
4. FINALIZE drops features with `feature_ok` false or null; passthrough aggregates are
   left-joined file by file (before: inner-joined to each other first).

**Output change (data pipeline).**

| File | Change |
|---|---|
| `feature_select_batchwise/<batch>/splits/bootstrap_<n>/half{1,2}.parquet` | columns `meta_cell_index`, `meta_variant_tag` instead of `tmp_cell_idx`. Each half names exactly the cells the old one selected (checked for all 12 halves). |
| `…/blocklists/<type>.parquet`, `…/blocklist.parquet` | row order (now sorted by `feature`); same rows and values |

No other file changed: half aggregates, correlations, blocklist verdicts and `output.parquet`
are identical. Differences 2–4 change values only for a feature whose correlation is undefined
in some replicate (e.g. a constant column) or when passthrough aggregates cover different
variants; the reference fixture has neither.
