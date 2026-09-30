# Clustermaps

`ClusterMap` draws a hierarchically clustered heatmap with one row per DataFrame row. It is
figure-level, so it builds its own figure and `plot()` doesn't accept `ax=`.

## All features at once

By default every column that doesn't start with `meta_` is a feature. The features form a single block
on a diverging `RdBu_r` scale clipped at ±3, and both rows and features are clustered. `row_colors` adds
a color strip and legend from a categorical column:

```python
import polars as pl
import fisseqborn as fb

synonymous = profiles.filter(pl.col("meta_clinvar_annotation") == "Synonymous")
(fb.ClusterMap(synonymous, row_colors="meta_cluster_idx",
               row_palette={"0": "orange", "1": "green", "9": "blue"})
   .save("vis/synonymous_clustermap.png"))
```

![Synonymous clustermap](../images/clustermap-synonymous.png){ width="560" }

*From `notebooks/2026-09-11/synonymous-clustering`, where it was drawn with `sns.clustermap`.*

Feature columns containing null, NaN or inf are dropped with a logged warning, because clustering
can't handle them.

## Feature groups: choosing what drives the clustering

A summary figure often mixes features on different scales, such as landmark z-scores, class
proportions and a score, and only some of them should decide the row order. Split the features into
`FeatureGroup`s. Each group:

- is drawn as its own block, with its own colormap and colorbar;
- with `cluster=True` (the default), is used when clustering the rows;
- with `cluster=False`, is only displayed, in the row order the other groups produce.

```python
from fisseqborn import FeatureGroup, fisseq

landmarks = {f"{k}_median": v for k, v in fisseq.LMNA_LANDMARK_FEATURES.items()}

# one row per cluster
fb.ClusterMap(
    cluster_summary,
    row_labels="label",                 # e.g. "6 (n=737)"
    standardize=True,
    title="Cluster summary",
    groups=[
        FeatureGroup("Landmark z-score vs. synonymous", list(landmarks),
                     cmap="RdBu_r", center=0, clip=3, labels=landmarks, annot=True),
        FeatureGroup("Share of class", ["Synonymous", "Frameshift"],
                     cmap="Greens", vmin=0, annot=True),
        FeatureGroup("Median distinguishability", "median_distinguishability",
                     cluster=False, cmap="Reds", vmin=0.5, vmax=1, labels=False, annot=True),
    ],
).save("vis/cluster_summary.png")
```

![Cluster summary drawn with fisseqborn](../images/clustermap-summary.png)

*Drawn by fisseqborn on real FISSEQ profiles (k-means clusters of the T1_R1 median aggregates). The
median distinguishability block has `cluster=False`, so it doesn't affect the order of the rows.*

`standardize=True` z-scores each clustering feature before distances are computed, so blocks on
different scales weigh in comparably. Zero-variance features are then left out. This matches the
original notebook helper. It only changes the ordering, not the colors.

Other options per group:

| Option | Does |
|---|---|
| `cluster_features=True` | also reorder the features inside the block by clustering them |
| `labels=` | `True`/`False`, or a mapping to display names (e.g. `fisseq.LMNA_LANDMARK_FEATURES`) |
| `annot=True`, `fmt=` | write each cell's value, with text color chosen for contrast |
| `cmap`, `vmin`, `vmax`, `center`, `clip` | the block's color scale |
| `palette=` | a white → color scale **per feature**, e.g. `{"Head": "tab:blue", …}`, `"tab10"` or `True` (fisseq colors when known) |
| `cmap={feature: cmap}`, `vmin={feature: v}`, … | per-feature colormaps and limits |

Groups with missing values or `cluster=False` keep their missing values, drawn in light grey.

## One color scale per feature

When the features of a block are separate quantities, such as the share of each domain, give each
one its own hue and color scale with `palette=`. Each feature then gets a small colorbar of its own,
next to its heatmap column (or row, when vertical) and aligned with it. A scalar `vmin`/`vmax`
applies to every feature, a missing limit comes from that feature's values, and a mapping sets
limits feature by feature. `cmap` also accepts a `{feature: colormap}` mapping.

`Dataset.cluster_summary` builds the one-row-per-cluster input, and `ClusterSummary.group` turns a
share block into a `FeatureGroup` with `"(n=…)"` labels:

```python
is_pathogenic = pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC
summary = profiles.cluster_summary(
    medians={**{c: c for c in landmarks}, "meta_distinguishability_score": "Median distinguishability"},
    zscore=list(landmarks),               # vs. the synonymous controls
    shares={
        "class": {fisseq.PATHOGENIC: is_pathogenic,
                  "Synonymous": pl.col("meta_variant_type") == "Synonymous",
                  "Frameshift": pl.col("meta_variant_type") == "Frameshift"},
        "domain": "meta_domain",
    },
)
fb.ClusterMap(
    summary,
    row_labels="label",
    standardize=True,
    orientation="vertical",
    groups=[
        FeatureGroup("Landmark z-score (vs. synonymous)", list(landmarks), cmap="RdBu_r",
                     center=0, clip=3, labels=landmarks),
        summary.group("domain", "Share of domain", palette="tab10"),
        summary.group("class", "Share of class", palette=True, annot=True),
        FeatureGroup("Median distinguishability", "Median distinguishability", cluster=False,
                     cmap="Reds", vmin=0.5, vmax=1, annot=True),
    ],
).save("vis/cluster_summary.png")
```

## Orientation

By default (`orientation="horizontal"`) each DataFrame row is a heatmap row, the groups sit side by
side, the dendrogram is on the left and `row_labels` are on the right. `orientation="vertical"`
transposes the figure: each DataFrame row is a heatmap column, the groups are stacked from top to
bottom with their features as heatmap rows (labelled on the right), the dendrogram runs along the
top, and `row_labels` become the x tick labels under the bottom group. Each group's title and
colorbar sit on its left.

`row_colors`, `row_labels`, `row_order` and the other `row_*` names always refer to DataFrame
rows, whichever way they are drawn.

This is the layout of the notebook figure that feature groups replace:

![Original cluster summary](../images/clustermap-summary-original.png)

*From `notebooks/2026-09-22/pca` (`fisseqstuff.vis.plot_cluster_summary`).*

Group titles wrap between words to fit their block. Horizontal blocks are made wide enough for
their longest title word, and vertical blocks tall enough for the whole title, so narrow groups'
titles never collide.

## What `plot()` returns

`plot()` returns `(fig, ClusterMapAxes)`, which holds:

- `heatmap_axes`, `colorbar_axes` and `title_axes`, dicts keyed by group name;
- `feature_colorbar_axes`, each per-feature group's colorbars keyed by feature;
- `row_dendrogram_ax` and `row_colors_ax`;
- `row_order`, the input row positions in plotting order (top to bottom, or left to right when
  vertical);
- `feature_order` for each group;
- `row_linkage`, the scipy linkage used to order the rows.

`ClusterMap(...).row_order()` computes the order without drawing, for example to reuse the same row
order in a table.
