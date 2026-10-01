# Boxplots

`BoxPlot` draws a boxplot of `y` for each level of `x`. You can split each box further by `hue`,
overlay the individual points, and add significance brackets.

## Score by variant class, with significance stars

```python
import fisseqborn as fb
from fisseqborn import fisseq

(fb.BoxPlot(
    df,
    x="meta_clinvar_annotation",
    y="meta_impact_score_0.9",
    order=["Synonymous", "Single Missense", fisseq.PATHOGENIC],
    points="density",
    title="Impact score (variance ratio = 0.9)",
 )
 .annotate_pairs()
 .save("vis/impact_score_box_0.9.png"))
```

![Impact score boxplot with density-colored points](../images/box-density.png){ width="480" }

*From `notebooks/2026-09-22/pca`.*

The same plot at a lower variance ratio, and with `points="strip"` and a chosen pair of
comparisons drawn outside the axes (`annotate_pairs(pairs=[…], loc="outside")`):

<div class="grid" markdown>

![Impact score at 0.7](../images/box-impact-0.7.png)

![Strip points, outside brackets](../images/box-strip-outside.png)

</div>

*From `notebooks/2026-09-30/pca`. With `loc="outside"`, the brackets go above the axes where a
`title` would be, so leave the title off.*

- **`points`:**
    - `"density"` draws jittered points colored by how dense each box's distribution is at that value
      (viridis), as in this figure.
    - `"strip"` draws translucent black points instead.
    - Either way, fliers are hidden.
- **`annotate_pairs()`:** runs a Mann-Whitney test between every pair of boxes by default and draws
  the stars with statannotations. It always tests the plot's own `x`/`y` columns, so the stars can't
  come from a different column than the one plotted. Pass `pairs=[("Synonymous", fisseq.PATHOGENIC)]`
  to choose specific comparisons, or `test="t-test_welch"`, `loc="outside"`, and so on.
- **`show_counts=True`** (the default): appends `(n = …)` to each tick label.

## Per-experiment boxes

With `hue`, each `x` level gets one box per hue level. `fisseq.batch_palette` gives replicates of a
tile related colors:

```python
(fb.BoxPlot(
    df,
    x="meta_variant_type",
    y="test_auroc_corrected",
    hue="meta_experiment",
    palette=fisseq.batch_palette(df["meta_experiment"]),
    show_counts=False,
    figsize=(10, 5),
 )
 .refline(y=0.5)
 .set(ylim=(0, 1))
 .legend(outside=True)
 .save("vis/by-experiment-corrected.png"))
```

![Boxplot per experiment](../images/box-by-experiment.png)

*From `notebooks/2026-07-28/ovwt`.*

With `hue`, `annotate_pairs()` compares hue levels **within** each `x` level. For example,
`BoxPlot(df, x="meta_experiment", hue="meta_variant_type", …).annotate_pairs()` tests
Synonymous vs. Single Missense vs. Frameshift inside every experiment. Pairs that involve an empty group
are skipped.

## Variant classes within each experiment

`hue_order` sets the hue levels and their order. Levels left out of it aren't drawn. `.plot()` returns
`(fig, ax)` for any extra matplotlib touches, such as a y grid:

```python
by_type = (fb.BoxPlot(scores, x="meta_experiment", y="auroc_median_fold",
                      hue="meta_variant_type",
                      hue_order=["Synonymous", "Frameshift", "Single Missense"],
                      show_counts=False, figsize=(10, 5))
           .set(ylim=(0, None), yticks=[0.1 * i for i in range(11)])
           .legend(outside=True))
fig, ax = by_type.plot()
ax.grid(axis="y")
fig.savefig("vis/median_barcode_score_by_type.png", bbox_inches="tight")
```

![Variant classes per experiment](../images/box-by-type-per-experiment.png)

To compare two score **columns** side by side, unpivot them into one value column and put the
column name on `hue`:

```python
long = scores.filter(pl.col("meta_variant_type") == "Synonymous").pipe(
    pl.LazyFrame.unpivot, on=["auroc_median_fold", "auroc_pooled"],
    index=["meta_experiment", "meta_aa_changes"],
    value_name="distinguishability_score", variable_name="score_type",
)
(fb.BoxPlot(long, x="meta_experiment", y="distinguishability_score", hue="score_type",
            show_counts=False, figsize=(10, 5))
   .set(ylim=(0, 1), ylabel="Distinguishability Score (Synonymous Only)")
   .save("vis/distinguishability_score_strategies.png"))
```

![Two scoring strategies per experiment](../images/box-score-strategies.png)

*From `notebooks/2026-09-29/ovwtcv`. Synonymous variants should score 0.5. The pooled score sits
below that in every batch, while the median-fold score is centered on it.*

## A score across ClinVar classes

```python
(fb.BoxPlot(spatial, x="meta_clinvar_annotation", y="meta_distinguishability_score",
            order=["Synonymous", "Single Missense", "Frameshift", fisseq.PATHOGENIC],
            points="density", figsize=(8, 7))
   .annotate_pairs()
   .set(xlabel=None)
   .save("vis/global_distinguishability_box.png"))
```

![Distinguishability across classes](../images/box-distinguishability.png){ width="520" }

*From `notebooks/2026-09-30/pca`.*
