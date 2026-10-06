// OVWT_BATCHWISE: k-fold (or barcode-holdout) one-vs-wildtype XGBoost scoring of one
// experiment's normalized cells (python -m fisseq_common.stages.ovwt). Every cell gets one
// out-of-fold score; each variant gets auroc_pooled, auroc_median_barcode and per-fold
// AUROCs.
//
// ovwt_cv_mode is "kfold" (ovwt_n_folds folds, every barcode in every fold's training set)
// or "barcode_holdout" (whole barcodes held out, one barcode group per fold, so a
// single-barcode variant is skipped). Under "barcode_holdout" ovwt_n_folds caps the fold
// count, and null -- rendered as the literal `null` that Hydra parses back into None --
// means one fold per barcode. Both pipelines validate these before any task runs.
//
// `cells` is the cell feature table the filter stage's keys and normalizer apply to. Shared by
// both pipelines (the embeddings pipeline also runs it as OVWT_BATCHWISE_CP_FEATURES).
// conf/modules.config sets ext.args (join_keys, feature_selector) and publishDir.

include { threadEnv } from '../functions'

process OVWT_BATCHWISE {
    tag "${batch_stem}"
    errorStrategy 'ignore'
    label 'process_high'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), path(cells), path(filtered_keys), path(normalizer)

    output:
    tuple val(batch_stem), path("results.parquet"), path("cell_scores.parquet"), path("models.pkl"), emit: ovwt

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_common.stages.ovwt \\
        output_dir=. \\
        cells_file=${cells} \\
        filtered_keys_file=${filtered_keys} \\
        normalizer_file=${normalizer} \\
        label_column=${params.filter_label_column} \\
        wt_label=${params.ovwt_wt_label} \\
        cv_mode=${params.ovwt_cv_mode} \\
        n_folds=${params.ovwt_n_folds} \\
        calibrate=${params.ovwt_calibrate} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
