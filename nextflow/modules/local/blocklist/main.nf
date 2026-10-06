// BLOCKLIST: one method's verdict per feature -- the median correlation across every
// bootstrap replicate against a minimum (fisseq_common.stages.blocklist). The one
// synchronization point across replicates: every other task in the chain fans out per
// replicate, this one gathers them.
//
// Shared by both pipelines. conf/modules.config sets ext.entry, ext.args (each pipeline's
// minimum_correlation param) and publishDir.

include { threadEnv } from '../functions'

process BLOCKLIST {
    tag "${batch_stem} ${method}"
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), val(method), path(correlations, stageAs: 'correlations/rep*.parquet')

    output:
    tuple val(batch_stem), path("${method}.parquet"), emit: blocklist

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m ${task.ext.entry} \\
        output_dir=. \\
        'correlation_files=correlations/*.parquet' \\
        output_name=${method} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
