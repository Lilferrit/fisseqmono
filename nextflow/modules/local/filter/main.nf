// FILTER: keep the QC-passed cells and fit the control-row normalizer
// (fisseq_common.stages.filter). Publishes only filtered_keys.parquet and
// normalizer.parquet; every consumer rebuilds the normalized table by joining back.
//
// Shared by both pipelines, under each pipeline's own process names:
//   data pipeline:       NORMALIZE (cells and qc_passed are both QC_FILTER's filtered_cells)
//   embeddings pipeline: FILTER_EMBEDDINGS, FILTER_CP_FEATURES
// conf/modules.config sets ext.entry, ext.args (which binds `cells` and `qc_passed` to the
// entry point's own config keys) and publishDir.

include { threadEnv } from '../functions'

process FILTER {
    tag "${batch_stem}"
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), path(cells, stageAs: 'cells/*'), path(qc_passed, stageAs: 'qc_passed/*')

    output:
    tuple val(batch_stem), path("filtered_keys.parquet"), path("normalizer.parquet"), emit: filtered

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m ${task.ext.entry} \\
        output_dir=. \\
        label_column=${params.filter_label_column} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
