# Concepts

## Plots take a polars DataFrame and column names

Every plot is a class. You construct it with a `pl.DataFrame` and the names of the columns to use, the
same way you would call a seaborn function:

```python
plot = fb.BoxPlot(df, x="meta_variant_type", y="meta_distinguishability_score")
```

Columns are checked when the plot is constructed. A typo fails immediately and suggests the closest
matches:

```text
ValueError: Column(s) not found in DataFrame:
  'meta_varient_type' (did you mean 'meta_variant_type'?)
```

Only the columns the plot uses are converted to pandas, and only when the plot is drawn. You never
need `.to_pandas()` yourself.

## Layers are chained

Extra elements are added with chained methods, called *layers*. Every plot has three:

| Layer | Does |
|---|---|
| `.refline(x=, y=, diagonal=)` | Dashed black reference lines, e.g. `y=0.5` for chance-level AUROC |
| `.set(**kwargs)` | `ax.set(...)` after drawing, e.g. `.set(ylim=(0, 1), title="…")` |
| `.legend(outside=True, **kwargs)` | Restyle the legend, or move it to the right of the axes |

Some plots add their own layers, such as `BoxPlot.annotate_pairs()` for significance stars,
`EmbeddingPlot.highlight()` to overlay a subset of points and `VolcanoPlot.layer()` to add a group of
points selected by a polars expression.

Each layer call returns a **new** plot and leaves the original alone. You can build a base plot once
and branch off variants without layers piling up:

```python
base = fb.BoxPlot(df, x="meta_variant_type", y="meta_distinguishability_score")

base.annotate_pairs().save("vis/dist_stars.png")
base.refline(y=0.5).set(ylim=(0, 1)).save("vis/dist_chance.png")   # no stars here
base.save("vis/dist_plain.png")                                    # base itself is unchanged
```

## Drawing and saving

Nothing is drawn until you call one of:

- `plot(ax=None)`, which returns `(fig, ax)`. Pass an existing `ax` to draw into a subplot grid.
- `save(path)`, which draws if needed, creates parent directories and saves with `bbox_inches="tight"`.
- `show()`, which draws and displays the figure.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
fb.BoxPlot(df, x="meta_variant_type", y="meta_impact_score").plot(ax=axes[0])
fb.CorrelationPlot(df, x="meta_num_cells", y="meta_impact_score", stat="spearman").plot(ax=axes[1])
fig.savefig("vis/panel.png")
```

`ClusterMap` is *figure-level*: it lays out its own figure, so it doesn't accept `ax=`.

## Defaults

`title=`, `figsize=` and `dpi=` work on every plot. Package-wide defaults live in `fb.config`:

```python
fb.config.dpi = 300          # e.g. for publication figures
fb.config.figsize = (4, 4)
```

Each plot type also has a default size that matches the notebooks. For example, embeddings are 8×8 and
boxplots are 6×6.
