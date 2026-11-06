# Variant effect maps

`VariantEffectMap` draws one value per single substitution as a heatmap. Protein positions run
along x and the substituted amino acid down y, in `fisseq.AMINO_ACID_ORDER`
(`A V I L G F Y W C M P S T N Q D E H K R`). It follows the missense maps of the VIS-seq paper:

- Gray cells are variants with no value.
- A black dot marks each position's wild-type residue, which is the synonymous variant's cell.
- The color scale centers on 0 by default. Values beyond `vmin`/`vmax` get the end colors, and
  the colorbar shows an arrow.
- The bar above the heatmap names the protein regions (from `gene=`'s domains, or `regions=`;
  hidden with neither). Positions in several regions, such as the overlap of two library
  tiles, are striped in all of their colors.
- The line below is each position's mean over its non-synonymous substitutions, with points
  colored by region. Points in an overlap are split between their first two regions' colors.
- `marginal=True` adds a strip on the right with each amino acid's values, with a black line
  at the median. It is off by default.

Only single substitutions like `"A12V"` and `"A12A"` are drawn. Frameshifts, deletions,
nonsense, `|` multi-codon changes, `:<tag>` pseudo-variants and `"WT"` are ignored. When a variant
has several rows, for example one per batch, they are combined with `aggregate="median"`
(`"mean"`, `"min"`, `"max"` or `"first"` also work; `None` makes duplicates an error).

```python
import fisseqborn as fb
import polars as pl

feature = "Mean_NucleiExpanded_Intensity_MeanIntensity_CH1"
(fb.VariantEffectMap(profiles, feature, positions=(178, 273), vmin=-6, vmax=6,
                     title=fb.genes.LMNA.landmark_features[feature])
   .highlight(pl.col("meta_clinvar_annotation") == fb.fisseq.PATHOGENIC,
              label=fb.fisseq.PATHOGENIC, color="yellow", marker="^")
   .save("vis/variant_map_t3.png"))
```

![Variant effect map](../images/variant-map-synthetic.png)

*Synthetic data. Every value is random, except that the synonymous variants are 0. The
ClinVar annotations are random too.*

## Highlighting variants

`.highlight(where, ...)` puts a marker on the cell of every variant with a row that matches
`where`, as `EmbeddingPlot.highlight` does with points. Rows are matched before `aggregate`
combines them. A cell gets marked even when its variant has no value. The marks appear on
every wrapped row. Each call is one layer, and later calls draw on top:

- `color` and `marker` set the style (default red circles with a black edge). Other keyword
  arguments go to `ax.scatter`, for example `s=` for the size.
- `label=` adds a legend entry below the colorbar.
- `hue=` colors the marks by a categorical column instead, with `palette=` for the colors
  and `missing_color=` for levels that have none. When a variant's matching rows disagree,
  its first non-null level wins.

```python
(fb.VariantEffectMap(profiles, feature, positions=(178, 273))
   .highlight(pl.col("meta_clinvar_annotation").is_not_null(),
              hue="meta_clinvar_annotation", label="ClinVar")
   .save("vis/variant_map_clinvar.png"))
```

## One figure per tile

`positions=(first, last)` limits the map to an inclusive range. Without it, the map covers every
substituted position in the data. To make one figure per library tile:

```python
for tile, first, last in fb.genes.LMNA.tiles:
    fb.VariantEffectMap(profiles, feature, positions=(first, last), vmin=-6, vmax=6,
                        title=tile).save(f"vis/variant_map_{tile}.png")
```

To color the region bar by tile instead of by domain, pass the tiles as `regions=`. Each
overlap between neighboring tiles is striped in both tiles' colors:

```python
tiles = {tile: (first, last) for tile, first, last in fb.genes.LMNA.tiles}
fb.VariantEffectMap(profiles, feature, regions=tiles, positions_per_row=170,
                    vmin=-6, vmax=6).save("vis/variant_map_tiles.png")
```

## Several region bars

To show domains and tiles together, pass a list of `RegionTrack`s. The bars stack in order,
top to bottom (left to right in portrait), and each label is drawn at the start of its bar.
Each track has its own `palette=`. The default is the fisseq palette, else seaborn's, so two
default tracks can share colors. `mean_regions=` picks the bar that colors the position-mean
points, by label or index. The default is the first bar.

```python
fb.VariantEffectMap(profiles, feature, regions=[
    fb.RegionTrack(fb.genes.LMNA.domains, label="domain"),
    fb.RegionTrack({t: (a, b) for t, a, b in fb.genes.LMNA.tiles}, label="tile",
                   palette="Set2"),
], mean_regions="tile", positions_per_row=170).save("vis/variant_map_domains_tiles.png")
```

## The whole protein, wrapped

`positions_per_row=` splits the range into stacked rows that share one colorbar. Every row
keeps the same cell width, so the last, shorter row just ends early:

```python
fb.VariantEffectMap(profiles, feature, positions_per_row=170, vmin=-6, vmax=6).save(
    "vis/variant_map_full.png"
)
```

## Portrait

`orientation="portrait"` turns the map on its side. Positions run down the page and amino
acids across the top. The region bar moves to the left of the heatmap and the position means
to the right (below for the `marginal` strip). With `positions_per_row=`, the wrapped
ranges become side-by-side columns:

```python
fb.VariantEffectMap(profiles, feature, positions=(178, 273), positions_per_row=48,
                    orientation="portrait", vmin=-6, vmax=6).save("vis/variant_map_tall.png")
```

![Portrait variant effect map](../images/variant-map-portrait-synthetic.png)

## Panels and colors

- `regions=None` hides the region bar. A different mapping (name -> inclusive range) labels
  another protein. `region_palette=` sets its colors. For several bars, use `RegionTrack`s
  (see above).
- `position_mean=False` hides the line below.
- `marginal=True` adds the per-amino-acid strip.
- `wild_type_marker=False` drops the dots.
- `cmap`, `vmin`, `vmax`, `center` and `clip` set the color scale. Without limits, the scale is
  symmetric around `center` and spans the largest deviation, capped at `clip`.
- `missing_color` sets the color of missing variants, and `cbar_label` the colorbar label
  (default: the value column).

The numbers behind the plot are `.matrix()` (amino acids by positions, NaN where a variant is
missing), `.position_means()`, `.wild_type()` (position -> residue) and `.cells` (one row per
drawn variant).

The plot is figure-level. `.plot()` returns the figure and a `VariantEffectMapAxes`, which has
one entry per row (per column in portrait) in `heatmaps`, `regions`, `means` and `marginals`.
Each entry of `regions` is a list of that row's bars, in track order.
Layers such as `.set(...)` apply to the first heatmap. `.highlight()` is the exception: it
marks every heatmap.
