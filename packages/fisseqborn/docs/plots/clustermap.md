# Clustermaps

`ClusterMap` draws a hierarchically clustered heatmap with one heatmap row per DataFrame row. It
is figure-level, so it builds its own figure and `plot()` doesn't accept `ax=`.

It works on two kinds of input:

- **Per-variant rows** (thousands of rows, a handful to a few dozen features): you mostly care about
  the dendrogram and the `row_colors` strip.
- **Per-cluster summaries** (one row per cluster, built by `Dataset.cluster_summary`): you split the
  columns into `FeatureGroup` blocks, each with its own color scale, and choose which blocks decide
  the row order.

Every figure on this page is drawn from the `spatial` profiles of `notebooks/2026-09-30/pca`.
Those are the PCs retained at 100% variance, Leiden-clustered with `cluster(n_neighbors=75)`, and
annotated with `domain()`, ClinVar and the OvWT distinguishability score. The landmark medians are
joined in from before the blocklist filter.

```python
import polars as pl
import fisseqborn as fb
from fisseqborn import FeatureGroup, fisseq

spatial = fb.Profiles.read("filtered_pca_profiles.parquet")

landmark_cols = [f"{k}_median" for k in fisseq.LMNA_LANDMARK_FEATURES]
landmark_cols.remove("AreaShape_Area_median")
landmarks = {f"{k}_median": v for k, v in fisseq.LMNA_LANDMARK_FEATURES.items()}

is_pathogenic = pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC
is_synonymous = pl.col("meta_variant_type") == "Synonymous"
is_frameshift = pl.col("meta_variant_type") == "Frameshift"
```

## Per-variant clustermaps

Without `groups`, every column that doesn't start with `meta_` is a feature, or just the columns
you pass as `features=`. They form a single block on a diverging `RdBu_r` scale clipped at ±3.
Both rows and features are clustered, and feature names are hidden. `row_colors` adds a color strip
and a legend from a categorical column, using the fisseq palette when it knows the levels:

```python
controls = (
    spatial
    .normalize(types=["median"])               # landmark z-scores vs. synonymous
    .filter(is_synonymous | is_pathogenic | is_frameshift)
    .with_columns(
        pl.when(is_pathogenic).then(pl.lit(fisseq.PATHOGENIC))
          .otherwise(pl.col("meta_variant_type"))
          .alias("meta_class")
    )
)
(fb.ClusterMap(controls.df, features=landmark_cols, row_colors="meta_class",
               title="Landmark z-scores of synonymous, frameshift and pathogenic variants")
   .save("vis/cm_variants_landmarks.png"))
```

![Per-variant clustermap of landmark z-scores](../images/cm-variants-landmarks.png)

The pathogenic and frameshift variants collect at the top and bottom, where nuclear intensity is
extreme. The synonymous controls fill the near-zero middle.

With many features, z-score them yourself first so no one feature dominates the distances. Here 400
random variants on their first 20 PCs, with Ward linkage and a categorical `row_palette`, show how
well the Leiden clusters line up with hierarchical clustering on the same space:

```python
pcs = [f"X_{i}" for i in range(20)]
sample = spatial.df.sample(400, seed=0).with_columns(
    (pl.col(c) - pl.col(c).mean()) / pl.col(c).std() for c in pcs
)
(fb.ClusterMap(sample, features=pcs, row_colors="meta_cluster_idx", row_palette="tab20",
               method="ward", title="400 variants on 20 PCs, colored by Leiden cluster")
   .save("vis/cm_variants_pcs.png"))
```

<div class="grid" markdown>

![Per-variant PC clustermap](../images/cm-variants-pcs.png)

![Same, with PuOr_r clipped at ±2](../images/cm-variants-pcs-cmap.png)

</div>

On the right is the same call with `cmap="PuOr_r", vmin=-2, vmax=2`. Those, plus `center=`, are
the knobs for the single default block.

!!! note "Missing values"
    Feature columns containing null, NaN or inf are dropped with a logged warning
    (`drop_nonfinite=True`), because the clustering can't handle them. Use a display-only group
    (below) to show such columns anyway.

## Summarizing clusters: `Dataset.cluster_summary`

A per-cluster summary mixes quantities on very different scales: landmark z-scores, the share of
each domain or variant class that falls in each cluster, and a median score.
`Dataset.cluster_summary` builds that one-row-per-cluster table:

```python
summary = spatial.cluster_summary(
    medians=[*landmark_cols, "meta_distinguishability_score"],
    zscore=landmark_cols,           # vs. the synonymous controls, before taking the medians
    shares={
        "domain": "meta_domain",    # one share per level of a column
        "class": {                  # or one share per boolean expression
            fisseq.PATHOGENIC: is_pathogenic,
            "Synonymous": is_synonymous,
            "Frameshift": is_frameshift,
        },
    },
    levels={"domain": list(fisseq.LMNA_DOMAIN_REGIONS)},  # fix the levels and their order
)
```

| Argument | Does |
|---|---|
| `by` | the cluster id column (default `meta_cluster_idx`). Rows come out in natural order (`"2"` before `"10"`) |
| `medians` | columns to take the per-cluster median of. A `{column: new name}` mapping renames them |
| `zscore` | z-score these `medians` columns (`True`: all of them) against the control rows (`meta_is_control`) *before* taking the medians |
| `shares` | share blocks by name: a column name (one share per level, nulls ignored) or `{level: boolean expr}` |
| `levels` | for column share blocks, the levels to keep and their order. Levels with no variants are kept as NaN |

Each share is the fraction of **that level's** variants that fall in the cluster, so every share
column sums to 1 over the clusters (it is normalized per level, not per cluster). The first rows
of the summary (some columns left out):

| meta_cluster_idx | n | label | Mean_NucleiExpanded_Intensity… | meta_distinguishability_score | Head | Ig-fold | Pathogenic/Likely pathogenic | Synonymous |
|---|---|---|---|---|---|---|---|---|
| 0 | 1567 | 0 (n=1567) | 1.03 | 0.55 | 0.14 | 0.11 | 0.16 | 0.15 |
| 1 | 1238 | 1 (n=1238) | -0.26 | 0.50 | 0.14 | 0.13 | 0.02 | 0.28 |
| 2 | 1160 | 2 (n=1160) | 0.26 | 0.50 | 0.12 | 0.11 | 0.02 | 0.19 |
| 3 | 1119 | 3 (n=1119) | -0.90 | 0.58 | 0.06 | 0.17 | 0.11 | 0.00 |
| 4 | 1093 | 4 (n=1093) | -1.62 | 0.58 | 0.10 | 0.11 | 0.04 | 0.04 |

The result is a `ClusterSummary`, a `Dataset` that remembers its share blocks:

```python
summary.shares("domain")   # ['Head', 'Coil 1A', 'L1', 'Coil 1B', …, 'Unfolded']
summary.totals["class"]    # {'Pathogenic/Likely pathogenic': 90, 'Synonymous': 534, 'Frameshift': 161}
summary.group("domain", "Share of domain", palette="tab10")   # a ready-made FeatureGroup
```

`summary.group(key, name, **kw)` returns a `FeatureGroup` of that share block. Its features are
labelled `"<level> (n=<total>)"` and `vmin` defaults to 0. Any other `FeatureGroup` option can be
passed through.

You can still use the single-block form on a summary, but you have to pick the `features` yourself:
`label` is a string and `n` is a count, so the default "every non-`meta_` column" would fail.

```python
fb.ClusterMap(summary.df, features=landmark_cols, row_labels="label").save("vis/cm.png")
```

![Single-block clustermap of a summary](../images/cm-summary-features.png){ width="560" }

This works, but the features aren't labelled and nothing shows what the colors mean. Feature groups
fix both.

## Feature groups

Split the features into `FeatureGroup`s. Each group:

- is drawn as its own block, with its own colormap, colorbar and title;
- with `cluster=True` (the default), is used when clustering the rows;
- with `cluster=False`, is only displayed, in the row order the other groups produce.

Here the notebook's figure is built up one group at a time:

```python
landmark_group = FeatureGroup(
    "Landmark z-score (vs. synonymous)", landmark_cols,
    cmap="RdBu_r", center=0, clip=3, labels=landmarks, annot=True,
)
fb.ClusterMap(summary, groups=[landmark_group], row_labels="label", standardize=True,
              title="One group").save("vis/cm_groups_1.png")
```

![One feature group](../images/cm-groups-1.png){ width="600" }

```python
domain_group = summary.group("domain", "Share of domain", palette="tab10", annot=True)
fb.ClusterMap(summary, groups=[landmark_group, domain_group], row_labels="label",
              standardize=True, title="+ share of domain").save("vis/cm_groups_2.png")
```

![Two feature groups](../images/cm-groups-2.png)

```python
class_group = summary.group("class", "Share of class", palette=True, annot=True, cluster=False)
dist_group = FeatureGroup("Median distinguishability", "meta_distinguishability_score",
                          cluster=False, cmap="Reds", annot=True, labels=False)

(fb.ClusterMap(summary, groups=[landmark_group, domain_group, class_group, dist_group],
               row_labels="label", standardize=True, title="Cluster summary")
   .save("vis/clustermap_combined.png"))
```

![Full cluster summary](../images/cm-groups-full.png)

*From `notebooks/2026-09-30/pca` (drawn there vertically, see [Orientation](#orientation)).
Clusters 9, 7 and 10 split off together. They have low nuclear lamin intensity and circularity,
the highest distinguishability, and 57% of the pathogenic variants. Cluster 7 alone holds 63% of
the frameshifts and 48% of the NLS variants.*

!!! tip "Group rules"
    Group names must be unique, and a feature can belong to only one group. Both raise a
    `ValueError`, as do non-numeric features. `features` accepts a column name, a list or a polars
    selector.

## Which groups drive the clustering

`cluster=False` keeps a block out of the distance computation. It is still drawn, in the order the
clustering groups produce. That is how you show an *outcome* (class shares, a score) next to
clusters ordered by *phenotype* without the outcome shaping the order.

The figure above clusters on landmarks + domain shares. Below, the class shares and the
distinguishability also cluster (left). On the right, *only* the distinguishability clusters and
the landmark block is display-only:

<div class="grid" markdown>

![Everything clusters](../images/cm-cluster-all.png)

![Only distinguishability clusters](../images/cm-cluster-only-dist.png)

</div>

`row_order()` returns the order without drawing. These are the cluster ids top to bottom for the
three variants:

```python
[labels[i] for i in fb.ClusterMap(summary, groups=..., standardize=True).row_order()]
# landmarks + domain     11  0  3  4  1  2  6  5  8  9  7 10
# every group            11  0  1  2  6  5  8  3  4  9  7 10
# distinguishability     10  7  9 11  6  3  4  8  1  2  0  5
```

If no group clusters, there is no dendrogram and the rows keep their input order, which for a
`cluster_summary` is natural cluster-id order:

![No clustering groups](../images/cm-cluster-none.png){ width="640" }

## `standardize`

The distance between two rows is computed over the concatenated clustering features. Landmark
z-scores range over ±8, but domain shares stay below 0.5, so without standardizing, the landmarks
decide almost everything. `standardize=True` z-scores each clustering feature across rows before
the distances are computed, so every feature weighs in comparably. Zero-variance features are left
out with a warning. It only changes the ordering: the colors and annotations still show the raw
values.

<div class="grid" markdown>

![standardize=False](../images/cm-standardize-false.png)

![standardize=True](../images/cm-standardize-true.png)

</div>

With `standardize=False` (left), the order is set almost entirely by the landmark block:
`7 10 9 11 3 4 6 0 5 2 1 8`. With `standardize=True` (right), the domain shares count as well:
`11 0 3 4 1 2 6 5 8 9 7 10`. Cluster 11 now leads next to 0, the other cluster with high lamin
intensity and broad domain coverage.

## Linkage `method` and `metric`

`method` goes to `scipy.cluster.hierarchy.linkage` and `metric` to `scipy.spatial.distance.pdist`.
They are used for the rows and for `cluster_features`. The default is `method="average",
metric="euclidean"`. Here are four combinations on the landmark block alone (with
`standardize=True`):

```python
fb.ClusterMap(summary, groups=[landmark_group], row_labels="label", standardize=True,
              method="ward", metric="euclidean")
```

<div class="grid" markdown>

![average / euclidean](../images/cm-linkage-average-euclidean.png)

![ward / euclidean](../images/cm-linkage-ward-euclidean.png)

![complete / correlation](../images/cm-linkage-complete-correlation.png)

![single / cityblock](../images/cm-linkage-single-cityblock.png)

</div>

- `ward` (Euclidean only) makes compact, balanced merges. Here it gives the same top split as the
  default (7, 10, 11, 9 against the rest) but regroups the lower branch.
- `metric="correlation"` compares the **shape** of each row's profile and ignores its magnitude.
  Clusters 10, 7, 3 and 4 (granularity up, everything else down) sit together at the bottom,
  even though 7 and 10 are far more extreme than 3 and 4.
- `single` linkage chains rows together one at a time. That gives you a staircase dendrogram and is
  rarely what you want for summaries.

## Inside a block

### `cluster_features`

By default a group's features keep the order you gave. `cluster_features=True` reorders them by
clustering the features too, which puts features that rise and fall together side by side:

<div class="grid" markdown>

![cluster_features=False](../images/cm-cluster-features-false.png)

![cluster_features=True](../images/cm-cluster-features-true.png)

</div>

```python
summary.group("domain", "Share of domain", cmap="Greens", annot=True, cluster_features=True)
```

The domains no longer run N- to C-terminal. NLS, whose share is concentrated in clusters 7 and
10, is set apart at one end, and domains with similar profiles across clusters (Coil 1B and
Coil 2, Tail and Unfolded) end up close together.

!!! warning
    For a group that uses `palette=` (one hue per feature), the hues are assigned in the
    *plotted* order. With `cluster_features=True`, a named palette like `"tab10"` therefore doesn't
    keep a feature's color. Pass an explicit `{feature: color}` mapping if the hues must stay put.

### Color scale

A group without a per-feature palette has one shared scale:

- `cmap`: defaults to `RdBu_r` when `center` is set, else `viridis`.
- `vmin` / `vmax`: fixed limits. A missing limit comes from the data.
- `center`: makes the scale symmetric around this value.
- `clip`: caps that symmetric half-width. Colorbars get arrow ends when values fall outside.

<div class="grid" markdown>

![center=0, clip=3](../images/cm-scale-center-clip3.png)

![center=0, no clip](../images/cm-scale-center-noclip.png)

![magma, vmin=-1, vmax=2](../images/cm-scale-vmin-vmax.png)

![default viridis](../images/cm-scale-viridis.png)

</div>

```python
FeatureGroup("…", landmark_cols, labels=landmarks, annot=True, center=0, clip=3)  # top left
FeatureGroup("…", landmark_cols, labels=landmarks, annot=True, center=0)          # top right
FeatureGroup("…", landmark_cols, labels=landmarks, annot=True,
             cmap="magma", vmin=-1, vmax=2)                                      # bottom left
FeatureGroup("…", landmark_cols, labels=landmarks, annot=True)                    # bottom right
```

Without `clip`, cluster 7's granularity (8.4) stretches the scale to ±8.4 and washes out every
other cell. For z-scores, `center=0, clip=3` is usually the right choice.

### Labels and annotations

`labels` is `True` (column names), `False` (hidden) or a mapping to display names. The default
shows names when the block has ≤ 40 features. `annot=True` writes each value on its cell, in black
or white for contrast, formatted with `fmt`:

<div class="grid" markdown>

![labels=False](../images/cm-labels-false.png)

![labels=True](../images/cm-labels-true.png)

![labels mapping, fmt=.1f](../images/cm-labels-mapping.png)

![fmt=.0%](../images/cm-annot-percent.png)

</div>

```python
FeatureGroup("…", landmark_cols, center=0, clip=3, labels=False)
FeatureGroup("…", landmark_cols, center=0, clip=3, labels=True)
FeatureGroup("…", landmark_cols, center=0, clip=3, labels=landmarks, annot=True, fmt=".1f")
summary.group("class", "Share of class, fmt='.0%'", palette=True, annot=True, fmt=".0%")
```

## One color scale per feature

When a block's features are separate quantities, such as the share of each domain, give each one
its own hue and scale. A group becomes **per-feature** when `palette` is set, when `cmap` is a
mapping, or when any limit is a mapping. Each feature then gets a small colorbar of its own,
aligned with its column.

- `palette="tab10"` (or any seaborn palette name or list) gives white → color, one color per feature.
- `palette=True` uses the fisseq colors when every feature is a known level (Synonymous green,
  Frameshift purple, pathogenic red), else seaborn's default. The class block above uses it.
- `palette={feature: color}` sets the colors by hand.

<div class="grid" markdown>

![palette="tab10"](../images/cm-palette-tab10.png)

![palette={domain: color}](../images/cm-palette-dict.png)

</div>

```python
summary.group("domain", 'palette="tab10"', palette="tab10")
FeatureGroup("palette={domain: color}", ["Head", "Coil 1A", "L1", "Coil 1B"], vmin=0, annot=True,
             palette={"Head": "goldenrod", "Coil 1A": "teal", "L1": "slateblue",
                      "Coil 1B": "crimson"})
```

Mappings for `cmap`, `vmin`, `vmax`, `center` and `clip` set them feature by feature, so one block
can hold a diverging z-score, a differently-colored diverging z-score and a sequential score. A
scalar limit applies to every feature. A missing one comes from that feature's own values:

```python
FeatureGroup(
    "cmap={feature: cmap}, vmax={feature: v}",
    [landmark_cols[0], landmark_cols[3], "meta_distinguishability_score"],
    cmap={landmark_cols[0]: "RdBu_r", landmark_cols[3]: "PiYG",
          "meta_distinguishability_score": "Reds"},
    center={landmark_cols[0]: 0, landmark_cols[3]: 0},
    vmax={"meta_distinguishability_score": 1.0},
    labels={**landmarks, "meta_distinguishability_score": "Median distinguishability"},
    annot=True,
)
```

![Per-feature colormaps](../images/cm-cmap-mapping.png){ width="600" }

## Feature color strips: `col_colors`

`row_colors` tags the DataFrame rows. `col_colors` tags the **features**: a `{feature: category}`
mapping drawn as a strip above them, with a legend, and `col_palette` picks the colors. Features
left out of the mapping get a blank strip.

```python
channel = {c: "Granularity" if "Granularity" in c else
              "Shape" if "AreaShape" in c else "Intensity" for c in landmark_cols}
region = {"Head": "Head", "Tail": "Tail", "NLS": "Tail", "Ig-fold": "Tail", "Unfolded": "Tail"}
channel |= {d: f"{region.get(d, 'Rod')} domain" for d in fisseq.LMNA_DOMAIN_REGIONS}

fb.ClusterMap(
    summary, groups=[landmark_group, domain_group], row_labels="label", standardize=True,
    col_colors=channel,
    col_palette={"Intensity": "tab:green", "Granularity": "tab:olive", "Shape": "tab:gray",
                 "Head domain": "khaki", "Rod domain": "peru", "Tail domain": "sienna"},
)
```

![col_colors strip](../images/cm-col-colors.png)

## Orientation

By default (`orientation="horizontal"`) each DataFrame row is a heatmap row:

- the groups sit side by side;
- the dendrogram is on the left;
- `row_labels` are on the right.

`orientation="vertical"` transposes the figure, which suits summaries with many features and few
clusters:

- each DataFrame row is a heatmap column;
- the groups are stacked top to bottom, with their features as heatmap rows (labelled on the right);
- the dendrogram runs along the top;
- `row_labels` become the x tick labels under the bottom group;
- each group's title and colorbar sit on its left.

```python
fb.ClusterMap(summary, groups=[landmark_group, domain_group, class_group, dist_group],
              orientation="vertical", row_labels="label", standardize=True,
              title="Cluster summary")
```

![Vertical cluster summary](../images/cm-vertical.png)

`row_colors`, `row_labels`, `row_order` and the other `row_*` names always refer to DataFrame
rows, whichever way they are drawn.

The figure size follows from the number of rows and features. Group titles wrap between words to
fit their block, and blocks are made wide (horizontal) or tall (vertical) enough for their titles.
`figsize=` overrides the size, and the blocks then stretch to fill it:

![figsize=(12, 9)](../images/cm-vertical-figsize.png)

## Missing values in display-only groups

Groups with `cluster=False` keep their missing values, drawn in light grey, so you can show
quantities that don't exist for every cluster. Here the median distinguishability of the
**pathogenic** variants alone is joined onto the summary. Cluster 5 has no pathogenic variants,
so its cell is null:

```python
path_dist = (spatial.df.filter(is_pathogenic).group_by("meta_cluster_idx")
             .agg(pl.col("meta_distinguishability_score").median()
                  .alias("pathogenic_distinguishability")))
with_nulls = summary.join(path_dist, on="meta_cluster_idx", how="left")

fb.ClusterMap(with_nulls, row_labels="label", standardize=True, groups=[
    landmark_group,
    FeatureGroup("Median distinguishability",
                 ["meta_distinguishability_score", "pathogenic_distinguishability"],
                 cluster=False, cmap="Reds", annot=True,
                 labels={"meta_distinguishability_score": "all variants",
                         "pathogenic_distinguishability": "pathogenic only"}),
])
```

![Missing values drawn grey](../images/cm-missing.png){ width="600" }

The same column in a clustering group would be dropped (with a warning) under the default
`drop_nonfinite=True`. With `drop_nonfinite=False` it raises an error instead.

## Working with the result

`plot()` returns `(fig, ClusterMapAxes)`, which holds:

| Field | Holds |
|---|---|
| `heatmap_axes`, `colorbar_axes`, `title_axes` | dicts keyed by group name |
| `feature_colorbar_axes` | each per-feature group's colorbars, keyed by group then feature |
| `row_dendrogram_ax`, `row_colors_ax` | the dendrogram and row-color strip (or `None`) |
| `row_order` | the input row positions in plotting order (top to bottom, or left to right when vertical) |
| `feature_order` | each group's features in plotting order |
| `row_linkage` | the scipy linkage used to order the rows |

Each heatmap is an `imshow`, so in a horizontal figure the cell for plotted row `r` and feature `c`
is centered at `(c, r)`. That makes it easy to post-edit, for example to outline clusters of
interest across every block:

```python
from matplotlib.patches import Rectangle

cm = fb.ClusterMap(summary, groups=[landmark_group, domain_group, class_group, dist_group],
                   row_labels="label", standardize=True,
                   title="Cluster summary, clusters 7 and 10 outlined")
fig, axes = cm.plot()
ids = summary.df["meta_cluster_idx"].to_list()
for name, ax in axes.heatmap_axes.items():
    n_features = len(axes.feature_order[name])
    for cid in ["7", "10"]:
        r = axes.row_order.index(ids.index(cid))
        ax.add_patch(Rectangle((-0.5, r - 0.5), n_features, 1, fill=False,
                               edgecolor="black", linewidth=2, clip_on=False))
fig.savefig("vis/cluster_summary_outlined.png", bbox_inches="tight")
```

![Post-edited clustermap](../images/cm-post-edit.png)

You don't have to draw anything to get the clustering itself:

- `ClusterMap(...).row_order()` returns the row order, for example to sort a table the same way.
- `ClusterMap(...).row_linkage()` returns the linkage, or `None` when no group clusters, so you can
  cut the tree with scipy:

```python
from scipy.cluster.hierarchy import fcluster

fcluster(cm.row_linkage(), t=3, criterion="maxclust")
# cluster 9 on its own, 7 and 10 together, and the other nine clusters in a third group
```

## Before fisseqborn

The notebooks used to draw these figures with `sns.clustermap` and a hand-rolled helper. Here is
the synonymous-variant clustermap from `notebooks/2026-09-11/synonymous-clustering`, now
`ClusterMap(synonymous, row_colors="meta_cluster_idx", row_palette={...})`:

![Synonymous clustermap](../images/clustermap-synonymous.png){ width="560" }

And `fisseqstuff.vis.plot_cluster_summary` from `notebooks/2026-09-22/pca`, the layout that feature
groups and `orientation="vertical"` replace:

![Original cluster summary](../images/clustermap-summary-original.png)
