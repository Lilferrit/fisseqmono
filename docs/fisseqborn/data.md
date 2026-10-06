# Loading pipeline outputs

The data classes read the output directory of either pipeline, fisseq-data-pipeline or
fisseq-embeddings-pipeline, and prepare it for plotting. They follow the same pattern as the plots: a class wraps a polars frame, and chained methods
each return a **new** object. So a notebook's loading code becomes a single chain:

```python
import polars as pl
import fisseqborn as fb

run = "/path/to/pipeline_dir"

profiles = (
    fb.Profiles.from_pipeline(run, types=["median"], passthrough=["KSnegLogP"])
      .variant_type()                                  # meta_variant_type + meta_is_control
      .median_across_batches(paired={"_median": "_KSnegLogP"})  # already z-scored per batch
      .clinvar("clinvar/clinvar_converted.parquet")    # meta_clinvar_annotation
      .domain()                                        # meta_domain
      .distinguishability(fb.OvwtScores.from_pipeline(run))
)

(fb.VolcanoPlot.from_wide(profiles)                    # plots take a dataset directly
   .layer(pl.col("meta_variant_type") == "Single Missense", label="Single Missense")
   .layer(pl.col("meta_variant_type") == "Synonymous", label="Synonymous")
   .save("vis/volcano.png"))
```

| Class | Holds |
|---|---|
| `Dataset` | Any frame keyed by `meta_aa_changes`. Provides the polars passthroughs, the variant annotations and saving. |
| `Profiles` | Feature profiles. Normalization, the median across batches, feature selection, the impact score, PCA/UMAP and clustering. |
| `OvwtScores` | OvWT distinguishability AUROCs. Batch correction and per-variant aggregation. |
| `Blocklists` | Per-batch feature reproducibility, the consensus feature list and its per-feature table. |

## The pipeline layout

`from_pipeline` finds each file through `fisseq_common.layout`, which both pipelines publish
with. The layout is detected from the run's top-level directories (`input/`: the data
pipeline; `cell_images/`, `cell_metadata/`, `embeddings/`, `cp_features/`: the embeddings pipeline), or
passed as `layout="data"` / `layout="embeddings"`. A run with neither, e.g. one with only
`feature_select_batchwise/` and `ovwt_batchwise/` copied out, is read as a data-pipeline run.
`track="cp_features"` reads the embeddings pipeline's CellProfiler track
(`*_cp_features/` directories), which has only aggregates and OvWT scores: no blocklists,
passthrough aggregates or `output.parquet`.

Both pipelines write the same per-experiment layout:

```text
<pipeline_dir>/
  feature_select_batchwise/<batch>/
    aggregates/<type>.parquet              # median, KS, AUROC: meta_aa_changes + <feature>_<type>,
                                           #   already z-scored to the batch's synonymous controls
    passthrough_aggregates/<type>.parquet  # KSnegLogP, AUROCnegLogP (raw)
    blocklists/<type>.parquet              # feature, median_r, feature_ok (= median_r >= min_correlation)
    output.parquet                         # per-batch selected profile + meta_num_cells etc.
  ovwt_batchwise/<batch>/results.parquet   # auroc_pooled, auroc_median_barcode, auroc_folds,
                                           #   auroc_median_fold, meta_n_barcodes, meta_n_cells
```

In the embeddings pipeline the features are the `emb_NNNN` dimensions, so the aggregate columns
are `emb_NNNN_<type>`. Its CellProfiler track writes only
`feature_select_batchwise_cp_features/<batch>/aggregates/<type>.parquet` (also z-scored).

There is no `global/` directory any more: the pipeline stopped running its global stages, so the
aggregation across experiments happens here. See
[Aggregate across experiments](#aggregate-across-experiments).

Batches load in natural order (`T2_R1` before `T10_R1`). Each row is tagged with its batch
in `meta_experiment`. A missing directory or file raises `FileNotFoundError` with the full path.
Every `from_pipeline` takes the same options for choosing batches:

- `batches=[...]` loads only those batches, in that order.
- `exclude=` drops the batches whose names match. Pass an `fnmatch` glob, a compiled regex, or a
  list of both. A glob has to match the whole name, while a regex can match anywhere in it:

```python
import re

fb.Profiles.from_pipeline(run, exclude="T10_*")                       # glob
fb.OvwtScores.from_pipeline(run, exclude=[re.compile(r"_R3$"), "T2_R1"])  # regex + name
```

`Profiles.from_pipeline` has two more options:

- `features="intersection"` keeps only the features present in every batch. The default,
  `"union"`, keeps every batch's features and fills in null where a batch lacks one. The
  intersection logs one warning per batch, listing the columns it dropped, and raises if no
  feature is shared.
- `metadata=True` (or a list of column names) left-joins the per-variant `meta_` columns of each
  batch's `output.parquet`, such as `meta_num_cells` and the barcode counts. Without it, the counts
  only come through `OvwtScores`. Sum them across batches with
  `median_across_batches(sum_cols=["meta_num_cells", ...])`. The embeddings pipeline's
  CellProfiler track writes no `output.parquet`, so it has no metadata to join.

A single file that isn't in this layout, such as a parquet a notebook wrote, can be loaded with
`fb.Profiles.read(path)`.

## Remote runs (scp)

A run that lives on a remote machine can be loaded in place. Pass an scp-style
`"user@host:/path/to/run"` instead of a local directory:

```python
run = "me@cluster:/data/fisseq/2026-10-01"
profiles = fb.Profiles.from_pipeline(run, types=["median"], download_dir="data/2026-10-01")
scores = fb.OvwtScores.from_pipeline(run, exclude="T10_*", download_dir="data/2026-10-01")
```

The loaders list the run with `ssh ... ls`, then copy **only the files they read** with scp. They
never copy whole directories. A run directory also holds pseudo-replicate aggregates and other
outputs that can add up to tens of GB, so this matters.

| Call | Copies, per selected batch |
|---|---|
| `Profiles.from_pipeline` | `aggregates/<type>.parquet` for each of `types`, `passthrough_aggregates/<type>.parquet` for each of `passthrough`, and `output.parquet` only with `metadata=` |
| `Blocklists.from_pipeline` | `blocklists/<type>.parquet` for each of `types`. With `types=None`, it first lists the `blocklists/` folders in one ssh call. |
| `OvwtScores.from_pipeline` | `results.parquet` |
| `Dataset.read("host:/path/file.parquet")` | that single file (globs aren't supported remotely) |

- `batches=` and `exclude=` are applied before anything is copied.
- Files land in `download_dir`, which mirrors the run's layout:
  `data/2026-10-01/feature_select_batchwise/T1_R1/aggregates/median.parquet`.
- Without `download_dir`, they go to a temporary directory (`fisseqborn-*`). The same run reuses that
  directory for the rest of the Python session. It isn't deleted automatically, because the lazy
  frames read from it. Its path is logged at `INFO` level.
- A file that is already in the download directory is reused. Pass `refresh=True` to copy it again.
  Each copy lands in a staging folder first, so an interrupted download is never mistaken for a
  cached file.
- A file that doesn't exist on the remote raises `FileNotFoundError`. Any other scp failure raises
  `RuntimeError` with scp's message.
- All the ssh and scp calls share one connection (`ControlMaster`), so you're prompted for a password
  or 2FA code at most once per minute of activity. Prompts need a terminal. In a notebook, set up an
  ssh key or agent, or open a connection to the host from a shell first.
- Paths are passed to scp unquoted, so remote paths with spaces aren't supported.

`fisseqborn-global` and `fb.write_global` accept a remote run too. All their loaders share
`--download-dir` / `download_dir`:

```bash
fisseqborn-global me@cluster:/data/fisseq/2026-10-01 --out global --download-dir data/2026-10-01
```

## Laziness

A dataset wraps a polars `LazyFrame`, so a chain only builds a query. The query runs when you need
the data:

- `.df` (cached)
- `.save(path)`
- passing the dataset to a plot

The methods that have to see the numbers run the query themselves: `drop_nonfinite`,
`feature_select`, the PCA,
UMAP and clustering methods, and `Blocklists.consensus`. Because every step returns a copy, a base
dataset can be branched without recomputing or mutating it:

```python
base = fb.Profiles.from_pipeline(run).variant_type()
per_variant = base.median_across_batches()

per_variant.impact_score().save("impact.parquet")
per_variant.drop_nonfinite().umap().cluster(n_neighbors=30).save("clusters.parquet")
```

## Profiles

### Normalization and aggregation

- Both pipelines write the aggregates **already z-scored** against each batch's synonymous controls,
  so you don't need `normalize(by="meta_experiment")` on current runs. On them it is a no-op, up to
  rounding. It is still useful for older runs that wrote raw aggregates, and for re-normalizing
  after other steps.
- `normalize(by=...)` z-scores every feature against the control rows: `(x - mean) / std`. It does
  this within each `by` group, or across all rows when `by` is not given.
    - Controls are the synonymous variants without a `:tag` suffix, flagged by `variant_type()`.
    - This matches the pipeline's `Normalizer`. NaN is treated as missing, and a feature whose
      controls don't vary becomes null.
    - Passthrough columns (the p-values) are never normalized.
- `median_across_batches(paired={"_median": "_KSnegLogP"})` collapses the data to one row per
  variant.
    - For each paired feature, the value and its p-value come from the same batch: the one holding
      the lower-middle value. Batches with tied values are ordered by p-value, so the result doesn't
      depend on the order the batches were loaded in.
    - A variant with no value in any batch gets a null p-value too.
    - `meta_n_experiments` counts the batches each variant was measured in.
    - Other `meta_` columns keep their first value, except `sum_cols`, which are summed.
    - `features="intersection"` first drops the features that have no value in some batch, with
      the same warnings as in `from_pipeline`.

### Feature selection

```python
ok = fb.Blocklists.from_pipeline(run).rethreshold(0.7).consensus()   # passes in every batch
profiles = profiles.keep_features(ok)          # p-value columns follow their feature
profiles = profiles.drop_features("*CH2*")     # names or fnmatch globs
profiles = profiles.drop_nonfinite()           # drop columns with any null / NaN / inf
```

- `rethreshold(min_r)` recomputes `feature_ok` as `median_r >= min_r`, the pipeline's
  `min_correlation` rule.
- `consensus(min_batches=None, missing="ignore")` returns the features that are OK in every batch
  that reports them, or in at least `min_batches` batches: the vote `fisseqborn-global` uses.
    - With `missing="fail"`, a batch whose blocklist doesn't list a feature counts as a failure
      for that feature.
- `table(min_batches=None, missing="ignore")` returns the counts behind the consensus, one row per
  feature, sorted by feature: `feature`, `n_batches` (the batches that report it), `n_ok` and
  `feature_ok`.

`feature_select()` runs `pycytominer.feature_select` over the profile values. It needs
`fisseqborn[select]`. The default operations are the ones the data pipeline used to run: drop near-zero-variance features,
apply the blocklist, then drop one of every pair of features correlated above `corr_threshold`.

```python
selected = (profiles
    .drop_nonfinite()
    .feature_select(corr_threshold=0.8))         # or operations=[...], blocklist_file=..., ...
selected.feature_selection                       # feature / kept, one row per input feature
```

- Every pycytominer knob is a keyword argument (`freq_cut`, `unique_cut`, `na_cutoff`,
  `outlier_cutoff`, ...). You can also pass other operations: `"drop_na_columns"`,
  `"drop_outliers"` and `"noise_removal"`.
- The `"blocklist"` operation uses pycytominer's built-in Cell Painting blocklist unless you give
  `blocklist_file`.
- P-value (passthrough) columns aren't used as inputs. They are kept whenever their feature is.

`profiles.feature_info()` parses feature names into a table with these columns: `statistic`,
`compartment`, `category` and `channels`. Use it to build `FeatureGroup`s or volcano selections:

```python
info = profiles.feature_info()
lamin = info.filter(pl.col("channels") == "CH1", pl.col("statistic") == "median")["feature"]
```

### Scores and embeddings

| Method | Adds |
|---|---|
| `impact_score()` | `meta_impact_score`: the cosine distance to the median control profile, halved to run from 0 to 1 |
| `impact_scores([0.7, 0.9])` | `meta_impact_score_0.7`, `meta_impact_score_0.9`: the impact score on the fewest PCs explaining each share of the variance (PCA is fitted once; the features are kept) |
| `pca(n_components=5)` | `meta_pc_1 … meta_pc_5`, with the loadings on `.pca_loadings` (write them with `.save_pca_loadings(path)`) |
| `pca_reduce(variance=0.9)` / `pca_reduce(noise_floor=True)` | replaces the features with `X_0 … X_k` |
| `umap(n_neighbors=30)` / `umap(n_components=3)` | `meta_notebook_umap_1 … _<n>` (or `output_cols=`); needs `fisseqborn[umap]` |
| `cluster("leiden", n_neighbors=15)` / `cluster("kmeans", n_clusters=8)` | `meta_cluster_idx`; Leiden needs `fisseqborn[cluster]` |
| `distinguishability(ovwt)` | `meta_distinguishability_score` |

For example, this pipeline from `notebooks/2026-10-01/dist-vs-impact` computes the two scores
that the [correlation](plots/correlation.md#distinguishability-vs-impact-score) and
[ROC](plots/roc.md) pages compare:

```python
ok_features = fb.Blocklists.from_pipeline(PIPELINE_DIR).rethreshold(0.8).consensus()
scored = (
    fb.Profiles.from_pipeline(PIPELINE_DIR, types=["median", "KS", "AUROC"])
    .variant_type()
    .median_across_batches()               # the aggregates are already z-scored per batch
    .keep_features(ok_features)
    .drop_nonfinite()
    .impact_scores([1.0])                  # -> meta_impact_score_1.0
    .distinguishability(fb.OvwtScores.from_pipeline(OVWT_PIPELINE_DIR),
                        score="auroc_median_fold")   # -> meta_distinguishability_score
    .clinvar(CLINVAR_PATH)
)
```

`ExplainedVariancePlot(scored)` then shows the PCA fit behind the impact scores (see
[Explained variance](plots/explained-variance.md)).

## Variant annotations

Every dataset has these methods. They read the `meta_aa_changes` labels:

| Method | Adds |
|---|---|
| `variant_type()` | `meta_variant_type` (Synonymous, Single Missense, Frameshift, Nonsense, 3nt Deletion, WT, Other) and `meta_is_control` |
| `position()` | `meta_position`, the leading position of the first codon. Use `strict=True` for single substitutions only. |
| `domain()` | `meta_domain`, from `fisseq.LMNA_DOMAIN_REGIONS`, for single substitutions |
| `tile()` | `meta_tile`, from `fisseq.LMNA_TILES`. Use `allow_multiple=True` to list both tiles in an overlap. |
| `clinvar(path)` | `meta_clinvar_*` columns and `meta_clinvar_annotation`: the ClinVar call, falling back to the variant type |

## OvWT distinguishability

```python
scores = fb.OvwtScores.from_pipeline(run)       # one row per (variant, batch)
per_variant = (scores
    .correct("auroc_pooled")                     # -> auroc_pooled_corrected
    .per_variant("auroc_pooled_corrected"))      # median across batches
```

`correct` takes one score column or a list. By default it corrects every score in
`fb.ovwt.SCORES` that is present: `auroc_pooled`, `auroc_median_barcode` and `auroc_median_fold`.
Each score `<score>` gets its own `<score>_corrected` column. `auroc_folds` holds the per-fold
lists, so it is never treated as a score, and passing it raises an error.

- With `rescale=True` (the default), `correct` shifts and scales each batch so that the mean and
  standard deviation of its synonymous variants equal the median of those statistics across
  batches.
    - Scores stay on the AUROC scale, so 0.5 still marks chance, but a corrected score can land
      slightly outside [0, 1].
    - Use `reference="all"` to base the correction on every variant instead.
- With `rescale=False`, the result is the plain z-score of each experiment against its synonymous
  controls, `(x - mean) / std`. This is the old `GLOBAL_OVWT` math.
- In both modes, the standard deviation uses ddof=1. If it is below float32 epsilon, that
  experiment's corrected scores are null.
- An experiment with fewer than two synonymous controls also gets all-null scores, and `correct`
  logs a warning about it.

`per_variant` also takes a list of scores. `meta_n_experiments` then counts the batches where any of
them is present.

`Profiles.distinguishability(scores)` runs the correction (`reference=`, `rescale=`) and the median
for you. It also joins per-variant scores as they are. For example, it can read the
`global_scores.parquet` that `fisseqborn-global` writes:

```python
global_scores = fb.OvwtScores.read("global/ovwt_distinguishability/global_scores.parquet")  # data pipeline
profiles.distinguishability(global_scores, score="meta_median_auroc_pooled")
```

`Profiles.from_global` and `OvwtScores.from_global` still read the `global/` directory of older runs.
They now raise a `DeprecationWarning`, because current runs don't have that directory.

## Aggregate across experiments

Neither pipeline aggregates across experiments. The `fisseqborn-global` command does it from the
per-experiment outputs of either pipeline, with the methods of the embeddings pipeline's former
global stages (GLOBAL_BLOCKLIST, GLOBAL_VARIANT_EMBEDDINGS, GLOBAL_VARIANT_DISTINGUISHABILITY and
their CellProfiler-track twins; `fisseq_common.global_aggregation`).

```bash
fisseqborn-global /path/to/pipeline_dir --out /path/to/global \
    --exclude 'T10_*' --exclude-regex '_R3$'
```

```text
<out>/<track>/
  blocklist.parquet              # the vote: feature, n_batches, n_ok, feature_ok
  median_aggregate.parquet       # per variant, each reproducible feature's median across experiments
  pca_scores.parquet             # full-rank PCA: meta_pc_1 .. meta_pc_n
  pca_components.parquet         # meta_component_idx + one loading column per feature
  pca_variance_explained.parquet # meta_variance_explained, meta_cumulative_variance_explained
  pca_reduced.parquet            # the leading PCs reaching --cumulative-variance-explained (0.9),
                                 #   meta_is_control, meta_impact_score on those PCs
<out>/<ovwt dir>/global_scores.parquet  # meta_median_<score>, meta_num_experiments
```

The track directories are `feature_select/` and `ovwt_distinguishability/` for a data-pipeline run;
`embeddings/` and `distinguishability/` (Cell-DINO track) plus `cp_features/` and
`distinguishability_cp_features/` (CellProfiler track, no blocklist) for an embeddings-pipeline run.

Per track:

1. **Vote.** Each experiment's combined `blocklist.parquet` (or its per-method blocklists, in an
   older run). A feature is reproducible when every experiment that reports it marks it OK, or at
   least `--min-batches N` do. `--min-correlation R` first recomputes each experiment's verdict as
   `median_r >= R`; `--missing fail` counts an experiment that doesn't report a feature against it.
2. **Pool.** Each experiment's aggregates of `--types` (default: every method the run aggregated)
   and `--passthrough`, minus the features voted not OK, medianed per variant over the features
   every experiment has.
3. **PCA** at full rank, seeded with `--seed` (default 0).
4. **Distinguishability.** Each experiment's OvWT scores z-scored against its own synonymous
   variants, then the median across experiments (`--scores`, `--no-ovwt`).

Off by default, applied to `median_aggregate` before the PCA: `--operations ...` (pycytominer
feature selection, e.g. `variance_threshold blocklist correlation_threshold`) with `--corr-threshold`, `--impact-score` (a pre-PCA `meta_impact_score`),
`--umap N`, `--metadata` (integer `meta_` counts summed across experiments) and
`--paired median:KSnegLogP` (each companion from the experiment holding its value's median).
`--no-blocklist` pools every feature; `--no-cp-features` skips the CellProfiler track.

Run `fisseqborn-global --help` for the full list. `fb.write_global(run, out, ...)` does the same
from Python and returns the paths it wrote, keyed `<dir>/<stem>`. The pooling steps are
`fisseq_common.global_aggregation`'s `blocklist_vote`, `drop_blocked`, `median_across_batches`
and `ovwt_global_scores`, and the PCA is `fisseqborn.global_aggregate.full_rank_pca`.

To aggregate a subset of experiments (the old pipeline's per-channel global stages), run the command
once per group, leaving the others out with `--exclude`.

## From the notebooks

| Notebook pattern | Now |
|---|---|
| `for curr in root.glob("*/aggregates/median.parquet"): ... pl.lit(curr.parent.parent.name)` | `Profiles.from_pipeline(run)` |
| `Normalizer.from_lazyframe(lf, fit_only_on_control=True).apply(lf)` per batch | not needed on current runs (aggregates are pre-normalized); `.variant_type().normalize(by="meta_experiment")` for older ones |
| `group_by("meta_aa_changes").agg((~cs.starts_with("meta")).median())` | `.median_across_batches()` |
| `median_with_p(feature)` | `.median_across_batches(paired={"_median": "_KSnegLogP"})` |
| per-batch blocklists → `n_ok == 15` | `Blocklists.from_pipeline(run).consensus()` |
| `scp -r cluster:/run .` | `from_pipeline("me@cluster:/run", download_dir=...)` (only the needed files) |
| `global/<channel>/...` from the pipeline | `fisseqborn-global run --out global` or `fb.write_global(run, "global")` |
| `bad_mask = ... is_null() \| is_nan() \| is_infinite()` | `.drop_nonfinite()` |
| `pycytominer.feature_select(profiles=..., operation=[...])` | `.feature_select(operations=[...])` |
| `map_elements(variant_classification, ...)` | `.variant_type()` |
| `add_clinvar_annotation(df, path)` / `add_domain(df, LMNA_DOMAIN_REGIONS)` / `get_tile` | `.clinvar(path)` / `.domain()` / `.tile()` |
| `add_distinguishability_score(df, ovwt_dir)` | `.distinguishability(fb.OvwtScores.from_pipeline(run))` |
| `compute_impact_score(lf)` | `.impact_score()` |
| `PCA(n_components=None)` + `searchsorted(...)` | `.pca_reduce(variance=0.9)` |
| `add_cluster_idx(df, method="leiden", ...)` | `.cluster("leiden", ...)` |
