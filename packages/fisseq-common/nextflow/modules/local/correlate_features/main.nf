// CORRELATE_FEATURES: per-feature Pearson r between one bootstrap replicate's two half
// aggregates, for one method (python -m fisseq_common.stages.correlatefeatures).
//
// Shared by both pipelines. conf/modules.config sets publishDir.

include { threadEnv } from '../functions'

process CORRELATE_FEATURES {
    tag "${batch_stem} ${method} rep${rep}"
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), val(rep), val(method), path(half1, stageAs: 'half1/*'), path(half2, stageAs: 'half2/*')

    output:
    tuple val(batch_stem), val(method), path("${method}.parquet"), emit: correlations

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_common.stages.correlatefeatures \\
        output_dir=. \\
        half1_file=${half1} \\
        half2_file=${half2} \\
        label_column=${params.filter_label_column} \\
        output_name=${method} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
