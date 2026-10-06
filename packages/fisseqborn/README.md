# fisseqborn

Seaborn-style, object-oriented plots for the fisseq project. Pass a **polars** DataFrame and
column names, optionally chain extra layers, then `plot()` or `save()`.

```python
import polars as pl
import fisseqborn as fb
from fisseqborn import fisseq

# Boxplot with Mann-Whitney stars, n= tick labels and a chance line.
(fb.BoxPlot(df, x="meta_variant_type", y="meta_distinguishability_score", points="strip")
   .annotate_pairs()
   .refline(y=0.5)
   .set(ylim=(0, 1))
   .save("vis/distinguishability.png"))

# UMAP hexbin of a score, with synonymous / pathogenic variants overlaid.
(fb.EmbeddingPlot(df, x="meta_notebook_umap_1", y="meta_notebook_umap_2",
                  hue="meta_distinguishability_score", kind="hexbin", cmap="Blues", vmin=0.5, vmax=1)
   .highlight(pl.col("meta_variant_type") == "Synonymous", label="Synonymous", color="green")
   .highlight(pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC, label=fisseq.PATHOGENIC,
              color="red", marker="^")
   .save("vis/umap_distinguishability.png"))
```

## Plots

| Class | For |
|---|---|
| `BoxPlot(df, x, y, hue=None, points=None/"strip"/"density")` | score by variant class / experiment; `.annotate_pairs()` adds significance stars |
| `EmbeddingPlot(df, x, y, hue=None, kind="scatter"/"hexbin")` | UMAP / PCA; categorical or numeric hue, `center=0` for z-scores; `.highlight(expr, ...)` overlays a subset, `hue=` colors it by a column (the base palette when the levels match) |
| `CorrelationPlot(df, x, y, stat="pearson", fit=None/"linear"/"lowess", identity=False)` | replicate vs replicate, score vs cell count; `kind="kde"`, `count_sides=True` |
| `RocPlot(df, label, positive, score, negative="Synonymous", group=None)` | one curve per score column or per group; `.aucs()` returns the AUC table |
| `VolcanoPlot(df, x, y, alpha=0.05, bonferroni=True)` | long-form effect size vs −log10 p; `.layer(expr, label=...)` adds a group of points, drawn in call order; `VolcanoPlot.from_wide(df)` takes one row per variant with `_median` / `_KSnegLogP` column pairs |
| `Heatmap(df, index, columns, values=None)` / `Heatmap.correlation(df, cols)` | pairwise matrices (long or wide form), correlation matrices |
| `ExplainedVariancePlot(profiles.pca_reduce(...), kind="cumulative"/"scree"/"both", thresholds=[...])` | cumulative explained variance / scree plot, marking thresholds and the noise floor |
| `PairPlot(df, vars, hue=None, diag_kind="kde")` | pairwise scatter grid (e.g. the first PCs) with the fisseq palettes / orders (figure-level) |
| `ClusterMap(df, groups=None, orientation="horizontal"/"vertical", row_colors=None, row_labels=None)` | clustered heatmaps with per-group color scales (figure-level, see below) |

## Loading pipeline outputs

The data classes read a fisseq-data-pipeline or fisseq-embeddings-pipeline output directory
(through `fisseq_common.layout`) and chain the same way plots do. Each
method returns a new object, and plots accept the result directly.

```python
profiles = (
    fb.Profiles.from_pipeline(run_dir, types=["median"], passthrough=["KSnegLogP"])
      .variant_type()                                  # meta_variant_type + meta_is_control
      .median_across_batches(paired={"_median": "_KSnegLogP"})  # already z-scored per batch
      .keep_features(fb.Blocklists.from_pipeline(run_dir).consensus())
      .clinvar("clinvar_converted.parquet")
      .distinguishability(fb.OvwtScores.from_pipeline(run_dir))
)
fb.VolcanoPlot.from_wide(profiles).save("vis/volcano.png")
profiles.drop_nonfinite().umap().cluster(n_neighbors=30).save("profiles.parquet")
```

A remote run can be loaded in place: `from_pipeline("me@cluster:/path/to/run", download_dir="data")`
copies only the files that call reads, over scp.

Neither pipeline aggregates across experiments. `fisseqborn-global <pipeline_dir> --out <dir>` does it
from the per-batch outputs: per track, the blocklist vote, `median_aggregate.parquet`, a full-rank
PCA (`pca_*.parquet`) and the distinguishability `global_scores.parquet` (see
[Aggregate across experiments](docs/data.md#aggregate-across-experiments)).

| Class | For |
|---|---|
| `Profiles` | per-variant feature profiles: `normalize`, `median_across_batches`, `keep_features`, `impact_score`, `impact_scores` (one column per PCA variance threshold, one fit), `pca`, `pca_reduce` (full `pca_explained_variance` table kept), `umap`, `cluster`, `distinguishability` |
| `OvwtScores` | OvWT AUROCs: `correct` (per-batch rescale, or with `rescale=False` a z-score, against synonymous variants) and `per_variant`; `from_pipeline` also reads the older `variant` / `test_auroc` results schema (pass `score="test_auroc"`) |
| `Blocklists` | per-batch feature reproducibility: `rethreshold`, `consensus`, `table` |
| `Dataset` | the base class: `filter`/`with_columns`/`join`/`pipe`, `variant_type`, `position`, `domain`, `tile`, `clinvar`, `save`, `cluster_summary` |
| `ClusterSummary` | one row per cluster from `cluster_summary`: medians (optionally z-scored vs controls), per-level shares, `n` and a `"<id> (n=…)"` label; `.group(key)` gives a ready `FeatureGroup` |

`umap()` needs `fisseqborn[umap]` and Leiden clustering needs `fisseqborn[cluster]`.

## Clustermaps with feature groups

`ClusterMap` splits the features into `FeatureGroup` blocks. Each block has its own colormap and colorbar.
`cluster=True/False` sets whether a block's features are used when clustering the rows. `palette=` (or a `cmap` / limit mapping) gives each feature of a block its own color scale and a small colorbar aligned with it. Blocks with
`cluster=False` are only drawn, in the row order that the other blocks produce. `standardize=True` z-scores the
clustering features first, so blocks on different scales count about equally.

```python
from fisseqborn import FeatureGroup, fisseq

summary = profiles.cluster_summary(         # one row per cluster, from per-variant profiles
    medians=[*landmark_cols, "meta_distinguishability_score"],
    zscore=landmark_cols,                   # z-scored vs the synonymous controls first
    shares={"class": {"Synonymous": pl.col("meta_variant_type") == "Synonymous",
                      fisseq.PATHOGENIC: pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC},
            "domain": "meta_domain"},       # share of each level falling in each cluster
)
fb.ClusterMap(
    summary,
    row_labels="label",                     # "6 (n=737)"
    standardize=True,
    orientation="vertical",                 # clusters as columns, groups stacked
    groups=[
        FeatureGroup("Landmark z-score", landmark_cols, cmap="RdBu_r", center=0, clip=3,
                     labels=landmark_names),
        summary.group("domain", "Share of domain", palette="tab10"),  # one hue + colorbar per domain
        summary.group("class", "Share of class", palette=True, annot=True),
        FeatureGroup("Median distinguishability", "meta_distinguishability_score", cluster=False,
                     cmap="Reds", vmin=0.5, vmax=1, annot=True),
    ],
).save("vis/cluster_summary.png")
```

`plot()` returns `(fig, ClusterMapAxes)`. It holds the per-group heatmap and colorbar axes, `row_order`,
`feature_order` and the row linkage. `ClusterMap(...).row_order()` gives the order without drawing anything.
Without `groups`, all non-`meta_` columns form one block that is clustered on both axes.

## Layers (every plot)

- `.refline(x=, y=, diagonal=)`: draws dashed reference lines.
- `.set(**ax_kwargs)`: runs `ax.set(...)` after drawing.
- `.legend(outside=True, ...)`: restyles or moves the legend.

Each layer call returns a **new** plot, so you can reuse a base plot in a loop without layers building up.
`plot(ax=...)` draws onto an existing axes, for example to build subplot grids.

## Theme

`fisseqborn.fisseq` holds:

- `VARIANT_TYPE_PALETTE`, `CLINVAR_PALETTE` and `PATHOGENIC`
- the canonical category orders
- `LMNA_DOMAIN_REGIONS`, `LMNA_TILES` and `LMNA_LANDMARK_FEATURES`
- `batch_palette(experiments)`, which gives replicates of the same tile the same hue

When every level of a hue column is known to the theme, its palette and order are used automatically.
Other levels are sorted naturally (so `"2"` comes before `"10"`).
Figure defaults (`dpi=150`) live in `fb.config`.

## Development

```sh
uv sync
uv run pytest
```

## Documentation

The docs are built with MkDocs (Material theme, API reference via mkdocstrings) from `docs/`:

```sh
uv sync --group docs
uv run mkdocs serve      # live preview at http://127.0.0.1:8000
```

Every push to `main` rebuilds the site and pushes it to the `gh-pages` branch
(`.github/workflows/docs.yml`).
