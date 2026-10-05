# The fisseq theme

`fisseqborn.fisseq` collects the project's shared constants. Plots apply them automatically.

## Automatic palettes and orders

Whenever a categorical column is used for color (`hue`, or `x` in a boxplot), fisseqborn:

1. uses the fisseq palette **if every level of the column appears in it**. Otherwise it uses seaborn's
   default palette for up to 10 levels, `tab20` for 11–20 and `husl` beyond that, so colors never
   repeat;
2. puts known levels in canonical order (Synonymous, Single Missense, Frameshift, …; LMNA domains
   N- to C-terminal) and sorts
   everything else *naturally*, so cluster `"2"` comes before `"10"`.

An explicit `palette=`, `order=` or `hue_order=` always wins.

| Level | Color |
|---|---|
| Synonymous | `darkgreen` |
| Single Missense, WT, 3nt Deletion, Other | `grey` |
| Frameshift, Nonsense | `purple` |
| `fisseq.PATHOGENIC` ("Pathogenic/Likely pathogenic"), Pathogenic | `red` |
| Uncertain significance | `gold` |

## Constants

```python
from fisseqborn import fisseq

fisseq.PATHOGENIC               # "Pathogenic/Likely pathogenic"
fisseq.VARIANT_TYPE_PALETTE     # variant type -> color
fisseq.CLINVAR_PALETTE          # ClinVar significance -> color
fisseq.PALETTE                  # both merged
fisseq.VARIANT_TYPE_ORDER, fisseq.CLINVAR_ORDER
fisseq.LMNA_DOMAIN_REGIONS      # domain -> (start, end) amino-acid positions
fisseq.LMNA_TILES               # [(tile, start, end)] library tiles (neighbors overlap)
fisseq.LMNA_LANDMARK_FEATURES   # CellProfiler feature -> readable name
```

## Experiment palettes

`fisseq.batch_palette(experiments)` colors `T{tile}_R{replicate}` experiment names. Replicates of the
same tile share a hue, with lower replicates lighter. Tiles are spread around the color wheel so
neighbors are easy to tell apart. This replaces the `build_batch_palette` helper that was copied between
notebooks.

```python
fb.BoxPlot(df, x="meta_variant_type", y="test_auroc_corrected", hue="meta_experiment",
           palette=fisseq.batch_palette(df["meta_experiment"]))
```
