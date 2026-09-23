# ROC curves

`RocPlot` measures how well a score separates a **positive** class from a **negative** (control)
class of `label`. The negative class defaults to Synonymous. Rows with any other label are dropped.

## One curve per score column

```python
import fisseqborn as fb
from fisseqborn import fisseq

ratios = [0.7, 0.8, 0.9, 1.0]
(fb.RocPlot(df, label="meta_clinvar_annotation", positive=fisseq.PATHOGENIC,
            score=[f"meta_impact_score_{r}" for r in ratios],
            names={f"meta_impact_score_{r}": str(r) for r in ratios},
            title="ROC: meta_impact_score vs pathogenic (ClinVar)")
   .save("vis/roc_pathogenic.png"))
```

![ROC curves by variance ratio](../images/roc-pathogenic.png){ width="420" }

*From `notebooks/2026-09-22/pca`.*

Each legend entry reads `name (AUC = …)`. `names` maps score columns (or group levels) to shorter
display names.

## One curve per group

Pass `group=` to draw one curve per level of a column, such as each experiment. `combined=True` adds a
black curve over all groups pooled:

```python
(fb.RocPlot(df, label="meta_variant_type", positive="Single Missense",
            score="meta_distinguishability_score", group="meta_experiment", combined=True,
            palette=fisseq.batch_palette(df["meta_experiment"]))
   .legend(outside=True, fontsize=7)
   .save("vis/roc_by_experiment.png"))
```

`positive` and `negative` also accept lists of levels. For example,
`positive=["Single Missense", "Frameshift"]`.

## The AUC table

`RocPlot(...).aucs()` returns the numbers as a polars DataFrame, one row per curve, with columns
`curve`, `auc`, `n_positive` and `n_negative`.
