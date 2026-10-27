# Element data pipeline

`fisseq-element-data-pipeline` runs the [data pipeline](../data-pipeline/index.md)'s workflow on
Element Biosciences cell tables. Only INPUT differs; every later stage, parameter, process name
and the publish layout are the data pipeline's, so fisseqborn reads its output directory as a
data-pipeline run.

INPUT builds the identity columns the shared stages expect from Element's:

| Output | From |
|---|---|
| `upBarcode` | the whole `MostAbundantTarget` (`target_col`), e.g. `W498A_0007`. A multi-variant target such as `A502L_R644H` has no barcode id and is kept as its own barcode. |
| `aaChanges` | `variant` (`variant_col`), with `wt_label` relabelled `WT` |
| `editDistance` | 0 for every cell: Element reports none, so QC_FILTER's edit-distance filter passes every cell |

Element's per-cell bookkeeping columns (`cell_id_global`, `tile`, `ImageNumber`,
`ObjectNumber`, `origin_x`, `origin_y`, `MostAbundantTargetCount`, `TotalCount`, `Confidence`)
and any other non-numeric column become `meta_*`; every other column is a feature.

## Running

```bash
cd packages/fisseq-element-data-pipeline
nextflow run . -params-file params.yaml --pipeline_dir /path/to/experiment
```

`params.yaml` is the data pipeline's, except that an `experiments:` entry (or the pipeline-wide
default) sets `target_col` and `variant_col` instead of `barcode_col`, `aa_changes_col` and
`edit_distance_col`. The image is `ghcr.io/lilferrit/fisseqmono/fisseq-element-data-pipeline`.
See the data pipeline's [configuration](../data-pipeline/configuration.md) and
[Nextflow workflow](../data-pipeline/nextflow.md) pages for everything else.
