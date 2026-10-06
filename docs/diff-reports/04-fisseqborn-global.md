# fisseqborn's cross-experiment aggregation switches to the embeddings pipeline's global methods

**Commit:** `write_global` / `fisseqborn-global` reproduce the embeddings pipeline's former
GLOBAL_* stages (`fisseq_common.global_aggregation`), checked by `tests/test_global_parity.py`.
This report is about a **data-pipeline** run's global output, which changes because fisseqborn
switched methods. Per-experiment pipeline outputs are untouched.

**Compared on:** the data pipeline's reference run (`tests/reference/data`, two experiments),
old fisseqborn (`fa34a21`) against new. Both runs got every method the run aggregated
(`types` = AUROC, KS, MAD, QQ, mean, median, std) and `min_batches=1`. With the defaults, every
median feature fails the vote in one of the two experiments: the old version then has no
feature to pass to pycytominer and raises, and the new one has no feature to pool and raises
too. That is a property of the small fixture, not a difference between the versions.

## Method changes

| Step | Old fisseqborn | New (embeddings pipeline's method) |
|---|---|---|
| Vote input | per-method `blocklists/<type>.parquet` of `types` | each experiment's combined `blocklist.parquet` (per-method files if absent) |
| Vote rule (default) | `missing="fail"`: OK in every experiment loaded | `missing="ignore"`: OK in every experiment that reports it, or `>= min_batches` (`min_batches_ok`) |
| Vote output order | first appearance | sorted by `feature` |
| Applying the vote | keep only features voted OK | drop only features voted not OK (unmentioned features are kept) |
| Default `types` | `median` | every method the run aggregated |
| Pooling | union of features; first value of other `meta_` columns, `meta_n_experiments` | features every experiment has; label column only |
| pycytominer selection | on by default | opt-in (`operations=`) |
| Pre-PCA impact score, `meta_variant_type`, `meta_is_control` | always | opt-in (`impact_score=True`); `meta_is_control` and `meta_impact_score` are now in `pca_reduced` |
| PCA | optional, `pca=N` components added to the aggregate | always, full rank, `random_state=seed`; `pca=` is ignored (deprecated) |
| OvWT | z-score per experiment against its synonymous variants, then median | same; `meta_num_experiments` counts the non-null z-scored first score |

## Output files

| Old | New |
|---|---|
| `feature_select/aggregate.parquet` | `feature_select/median_aggregate.parquet` |
| `feature_select/blocklist.parquet` | `feature_select/blocklist.parquet` (sorted) |
| `feature_select/pca_components.parquet` (with `pca=N`) | `feature_select/pca_{scores,components,variance_explained,reduced}.parquet` |
| `ovwt_distinguishability/global_scores.parquet` | unchanged name |

## Value changes on the reference run (`min_batches=1`, every type)

- `blocklist.parquet`: the same 35 rows and verdicts (13 OK). Both experiments report every
  feature, so the two vote rules agree here; only the row order changed.
- `median_aggregate.parquet` vs old `aggregate.parquet`: same four variants. The 8 features
  both versions keep have identical values (max |old - new| = 0). New keeps 13 features (every
  feature voted OK); old kept 8 of those, after pycytominer's variance/correlation filters.
  New has no `meta_n_experiments`, `meta_variant_type`, `meta_is_control`, `meta_impact_score`
  columns (the last two are in `pca_reduced.parquet`).
- `global_scores.parquet`: equal up to floating-point summation order (max |old - new| =
  1.3e-14).

A feature missing from one experiment's blocklist, or a variant missing from one experiment,
would also change values: the old default counted the missing blocklist against the feature,
and pooled over the union of features. The reference run has neither.
