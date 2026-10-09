# Calibration plots

`CalibrationPlot` draws a `Calibration` (see [Evidence calibration](../calibration.md)). It shows
the score distributions of its control groups (P/LP, B/LB, gnomAD, synonymous) over shaded
bands, one per evidence point's score range, red for pathogenic and blue for benign, each
labelled with its points. This is the view of Fig. 4a in Tejura et al. Dashed lines are each
group's fitted mixture density, the median over bootstrap fits.

```python
import fisseqborn as fb

cal = profiles.calibrate("meta_distinguishability_score", gnomad=GNOMAD_CSV, clinvar=CLINVAR_PATH)
(fb.CalibrationPlot(cal, title="LMNA distinguishability")
   .refline(x=0.5)
   .save("vis/calibration.png"))
```

![LMNA calibration](../images/calibration-lmna-table.png)

*LMNA LOBO distinguishability (`notebooks/2026-08-28/ovwtlobo`) against the gnomAD v4.1.2
export and the converted ClinVar table. The table has one benign call, so B/LB is dropped
and not drawn.*

- **Data.** Pass the dataset `calibrate` returned: it carries the `Calibration`, and its
  `meta_excalibr_groups` column says which group each variant is in. For a plain DataFrame,
  pass `calibration=`.
- **Score column.** `score` defaults to the column the calibration was fit on.
- **Groups.** `groups=["P/LP", "B/LB"]` draws a subset.
- **Distributions.** `kind="kde"` draws KDEs instead of histograms. `fit=False` hides the
  fitted densities and `bands=False` the point bands.
- **Layers.** `.refline`, `.set` and `.legend` work as on every plot.
