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
| `EmbeddingPlot(df, x, y, hue=None, kind="scatter"/"hexbin")` | UMAP / PCA; categorical or numeric hue, `center=0` for z-scores; `.highlight(expr, ...)` overlays a subset |
| `CorrelationPlot(df, x, y, stat="pearson", fit=None/"linear"/"lowess", identity=False)` | replicate vs replicate, score vs cell count; `kind="kde"`, `count_sides=True` |
| `RocPlot(df, label, positive, score, negative="Synonymous", group=None)` | one curve per score column or per group; `.aucs()` returns the AUC table |
| `VolcanoPlot(df, x, y, alpha=0.05, bonferroni=True)` | long-form effect size vs −log10 p; `.layer(expr, label=...)` adds a group of points, drawn in call order; `VolcanoPlot.from_wide(df)` takes one row per variant with `_median` / `_KSnegLogP` column pairs |
| `Heatmap(df, index, columns, values=None)` / `Heatmap.correlation(df, cols)` | pairwise matrices (long or wide form), correlation matrices |
| `ClusterMap(df, groups=None, row_colors=None, row_labels=None)` | clustered heatmaps with per-group color scales (figure-level, see below) |

## Clustermaps with feature groups

`ClusterMap` splits the features into `FeatureGroup` blocks. Each block has its own colormap and colorbar.
`cluster=True/False` sets whether a block's features are used when clustering the rows. Blocks with
`cluster=False` are only drawn, in the row order that the other blocks produce. `standardize=True` z-scores the
clustering features first, so blocks on different scales count about equally.

```python
from fisseqborn import FeatureGroup, fisseq

fb.ClusterMap(
    cluster_df,                      # one row per cluster
    row_labels="label",
    standardize=True,
    groups=[
        FeatureGroup("Landmark z-score", landmark_cols, cmap="RdBu_r", center=0, clip=3,
                     labels=landmark_names),
        FeatureGroup("Share of class", ["Synonymous", "Frameshift", fisseq.PATHOGENIC],
                     cmap="Greens", vmin=0, annot=True),
        FeatureGroup("Median distinguishability", "median_distinguishability", cluster=False,
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
- `LMNA_DOMAIN_REGIONS` and `LMNA_LANDMARK_FEATURES`
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
