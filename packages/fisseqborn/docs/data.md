# Loading pipeline outputs

The data classes read the output directory of
[fisseq-data-pipeline](https://github.com/Lilferrit/fisseq-data-pipeline) and prepare it for
plotting. They follow the same pattern as the plots: a class wraps a polars frame, and chained methods
each return a **new** object. So a notebook's loading code becomes a single chain:

```python
import polars as pl
import fisseqborn as fb

run = "/path/to/pipeline_dir"

profiles = (
    fb.Profiles.from_pipeline(run, types=["median"], passthrough=["KSnegLogP"])
      .variant_type()                                  # meta_variant_type + meta_is_control
      .normalize(by="meta_experiment")                 # z-score each batch vs its synonymous variants
      .median_across_batches(paired={"_median": "_KSnegLogP"})
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
| `Blocklists` | Per-batch feature reproducibility, and the consensus feature list. |

## The pipeline layout

`from_pipeline` expects the layout that the pipeline writes (see its *Architecture → Output layout*
docs):

```text
<pipeline_dir>/
  feature_select_batchwise/<batch>/
    aggregates/<type>.parquet              # median, KS, AUROC: meta_aa_changes + <feature>_<type>
    passthrough_aggregates/<type>.parquet  # KSnegLogP, AUROCnegLogP
    blocklists/<type>.parquet              # feature, median_r, feature_ok
  ovwt_batchwise/<batch>/results.parquet   # auroc_pooled, auroc_median_barcode, ...
  global/<channel>/
    feature_select/aggregate.parquet             # Profiles.from_global
    ovwt_distinguishability/global_scores.parquet # OvwtScores.from_global
```

Batches load in natural order (`T2_R1` before `T10_R1`). Each row is tagged with its batch
in `meta_experiment`. Use `batches=[...]` to load a subset. A missing directory or file raises
`FileNotFoundError` with the full path.

A single file that isn't in this layout, such as a parquet a notebook wrote, can be loaded with
`fb.Profiles.read(path)`.

## Laziness

A dataset wraps a polars `LazyFrame`, so a chain only builds a query. The query runs when you need
the data:

- `.df` (cached)
- `.save(path)`
- passing the dataset to a plot

The methods that have to see the numbers run the query themselves: `drop_nonfinite`, the PCA,
UMAP and clustering methods, and `Blocklists.consensus`. Because every step returns a copy, a base
dataset can be branched without recomputing or mutating it:

```python
base = fb.Profiles.from_pipeline(run).variant_type().normalize(by="meta_experiment")
per_variant = base.median_across_batches()

per_variant.impact_score().save("impact.parquet")
per_variant.drop_nonfinite().umap().cluster(n_neighbors=30).save("clusters.parquet")
```

## Profiles

### Normalization and aggregation

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

### Feature selection

```python
ok = fb.Blocklists.from_pipeline(run).rethreshold(0.7).consensus()   # passes in every batch
profiles = profiles.keep_features(ok)          # p-value columns follow their feature
profiles = profiles.drop_features("*CH2*")     # names or fnmatch globs
profiles = profiles.drop_nonfinite()           # drop columns with any null / NaN / inf
```

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
| `pca(n_components=5)` | `meta_pc_1 … meta_pc_5`, with the loadings on `.pca_loadings` |
| `pca_reduce(variance=0.9)` / `pca_reduce(noise_floor=True)` | replaces the features with `X_0 … X_k` |
| `umap(n_neighbors=30)` | `meta_notebook_umap_1/2`; needs `fisseqborn[umap]` |
| `cluster("leiden", n_neighbors=15)` / `cluster("kmeans", n_clusters=8)` | `meta_cluster_idx`; Leiden needs `fisseqborn[cluster]` |
| `distinguishability(ovwt)` | `meta_distinguishability_score` |

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

`correct` shifts and scales each batch so that the mean and standard deviation of its synonymous
variants equal the median of those statistics across batches. Use `reference="all"` to base the
correction on every variant instead. Scores stay on the AUROC scale, so 0.5 still marks chance, but a
corrected score can land slightly outside [0, 1].

`Profiles.distinguishability(scores)` runs the correction and the median for you. It also accepts
per-variant scores as they are, for example the pipeline's own cross-experiment scores:

```python
profiles.distinguishability(fb.OvwtScores.from_global(run, "main"), score="meta_median_auroc_pooled")
```

## From the notebooks

| Notebook pattern | Now |
|---|---|
| `for curr in root.glob("*/aggregates/median.parquet"): ... pl.lit(curr.parent.parent.name)` | `Profiles.from_pipeline(run)` |
| `Normalizer.from_lazyframe(lf, fit_only_on_control=True).apply(lf)` per batch | `.variant_type().normalize(by="meta_experiment")` |
| `group_by("meta_aa_changes").agg((~cs.starts_with("meta")).median())` | `.median_across_batches()` |
| `median_with_p(feature)` | `.median_across_batches(paired={"_median": "_KSnegLogP"})` |
| per-batch blocklists → `n_ok == 15` | `Blocklists.from_pipeline(run).consensus()` |
| `bad_mask = ... is_null() \| is_nan() \| is_infinite()` | `.drop_nonfinite()` |
| `map_elements(variant_classification, ...)` | `.variant_type()` |
| `add_clinvar_annotation(df, path)` / `add_domain(df, LMNA_DOMAIN_REGIONS)` / `get_tile` | `.clinvar(path)` / `.domain()` / `.tile()` |
| `add_distinguishability_score(df, ovwt_dir)` | `.distinguishability(fb.OvwtScores.from_pipeline(run))` |
| `compute_impact_score(lf)` | `.impact_score()` |
| `PCA(n_components=None)` + `searchsorted(...)` | `.pca_reduce(variance=0.9)` |
| `add_cluster_idx(df, method="leiden", ...)` | `.cluster("leiden", ...)` |
