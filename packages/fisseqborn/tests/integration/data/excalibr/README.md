# Upstream ExCALIBR examples

Copied unchanged from [rosstewart/exCALIBR](https://github.com/rosstewart/exCALIBR)
(`example/` at commit 58bdb7c839e84d00f4a04f5028325f0bfc99cdf8), under its MIT license (`LICENSE`, here).

- `brca_findlay_example.csv`: BRCA1 saturation genome editing scores (Findlay et al. 2018)
  with `sample_assignments` codes 0 = P/LP, 1 = B/LB, 2 = gnomAD, 3 = synonymous
  (comma-separated when a variant is in several groups).
- `brca_findlay_PU_example.csv`, `brca_findlay_NU_example.csv`: the same data with only
  P/LP (positive-unlabeled) or only B/LB and synonymous (negative-unlabeled) controls.
- `BRCA1_Findlay_2018.json`: upstream's published calibration of that data set.
