# Variant effect maps

`VariantEffectMap` draws one value per single substitution as a heatmap. Protein positions run
along x and the substituted amino acid down y, in `fisseq.AMINO_ACID_ORDER`
(`A V I L G F Y W C M P S T N Q D E H K R`). It follows the missense maps of the VIS-seq paper:

- Gray cells are variants with no value.
- A black dot marks each position's wild-type residue, which is the synonymous variant's cell.
- The color scale centers on 0 by default. Values beyond `vmin`/`vmax` get the end colors, and
  the colorbar shows an arrow.
- The bar above the heatmap names the protein regions (`fisseq.LMNA_DOMAIN_REGIONS` by
  default).
- The line below is each position's mean over its non-synonymous substitutions, with points
  colored by region.
- The strip on the right shows each amino acid's values, with a black line at the median.

Only single substitutions like `"A12V"` and `"A12A"` are drawn. Frameshifts, deletions,
nonsense, `|` multi-codon changes, `:<tag>` pseudo-variants and `"WT"` are ignored. When a variant
has several rows, for example one per batch, they are combined with `aggregate="median"`
(`"mean"`, `"min"`, `"max"` or `"first"` also work; `None` makes duplicates an error).

```python
import fisseqborn as fb

feature = "Mean_NucleiExpanded_Intensity_MeanIntensity_CH1"
(fb.VariantEffectMap(profiles, feature, positions=(178, 273), vmin=-6, vmax=6,
                     title=fb.fisseq.LMNA_LANDMARK_FEATURES[feature])
   .save("vis/variant_map_t3.png"))
```

![Variant effect map](../images/variant-map-synthetic.png)

*Synthetic data. Every value is random, except that the synonymous variants are 0.*

## One figure per tile

`positions=(first, last)` limits the map to an inclusive range. Without it, the map covers every
substituted position in the data. To make one figure per library tile:

```python
for tile, first, last in fb.fisseq.LMNA_TILES:
    fb.VariantEffectMap(profiles, feature, positions=(first, last), vmin=-6, vmax=6,
                        title=tile).save(f"vis/variant_map_{tile}.png")
```

## The whole protein, wrapped

`positions_per_row=` splits the range into stacked rows that share one colorbar. Every row
keeps the same cell width, so the last, shorter row just ends early:

```python
fb.VariantEffectMap(profiles, feature, positions_per_row=170, vmin=-6, vmax=6).save(
    "vis/variant_map_full.png"
)
```

## Panels and colors

- `regions=None` hides the region bar. A different mapping (name -> inclusive range) labels
  another protein. `region_palette=` sets its colors.
- `position_mean=False` hides the line below.
- `marginal=False` hides the strip on the right.
- `wild_type_marker=False` drops the dots.
- `cmap`, `vmin`, `vmax`, `center` and `clip` set the color scale. Without limits, the scale is
  symmetric around `center` and spans the largest deviation, capped at `clip`.
- `missing_color` sets the color of missing variants, and `cbar_label` the colorbar label
  (default: the value column).

The numbers behind the plot are `.matrix()` (amino acids by positions, NaN where a variant is
missing), `.position_means()`, `.wild_type()` (position -> residue) and `.cells` (one row per
drawn variant).

The plot is figure-level. `.plot()` returns the figure and a `VariantEffectMapAxes`, which has
one entry per row in `heatmaps`, `regions`, `means` and `marginals`. Layers such as
`.set(...)` apply to the first heatmap.
