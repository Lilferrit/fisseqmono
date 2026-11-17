#!/usr/bin/env nextflow
// Single pipeline mode -- no --pipeline_mode dispatch needed unless/until a
// second mode is added.
//
// FISSEQ Data Pipeline -- Nextflow DSL2
//
// DAG:
//
//   params.yaml (experiments: [...]) ──► INPUT ──► input/<batch_stem>.parquet
//        │
//        ▼
//   QC_FILTER   (per experiment)
//        │
//        ▼
//   NORMALIZE   (per experiment)   ← z-score fit on WT control cells
//        │
//        ├──► OVWT_BATCHWISE  (per experiment; optional: params.run_ovwt)
//        │
//        ├──► Feature selection, batchwise (optional: params.run_feature_selection)
//        │
//        └──► ComBat across experiments (optional: params.run_combat):
//               aggregates -> COMBAT_FEATURES, OVWT_BATCHWISE -> COMBAT_OVWT
//
// Every output but ComBat's is per experiment. Other cross-experiment aggregation is done
// downstream by the fisseqborn package, which reads the layout below.
//
// Output layout:
//   {pipeline_dir}/
//     input/{batch_stem}.parquet                     INPUT output
//     qc_filter/{batch_stem}/                        filtered_cells, barcode_counts, variants_per_barcode
//     normalization/{batch_stem}/                    filtered_keys, normalizer
//     ovwt_batchwise/{batch_stem}/                   results, cell_scores, models.pkl
//     feature_select_batchwise/{batch_stem}/         aggregates, passthrough_aggregates, splits,
//                                                    correlations, blocklists, blocklist, output
//     combat/                                        features, ovwt_scores (every experiment, stacked)
nextflow.enable.dsl = 2

include { FisseqPipeline } from './workflows/fisseq'

workflow {
    FisseqPipeline()
}
