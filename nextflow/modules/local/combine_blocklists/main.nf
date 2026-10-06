// COMBINE_BLOCKLISTS: one experiment's per-method blocklists, concatenated
// (fisseq_common.stages.combineblocklists). Stat suffixes (f_median vs f_KS) keep each
// method's feature names disjoint.
//
// Shared by both pipelines. conf/modules.config sets ext.entry and publishDir.

include { threadEnv } from '../functions'

process COMBINE_BLOCKLISTS {
    tag "${batch_stem}"
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), path(blocklists, stageAs: 'blocklists/*')

    output:
    tuple val(batch_stem), path("blocklist.parquet"), emit: blocklist

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m ${task.ext.entry} \\
        output_dir=. \\
        'blocklist_files=blocklists/*.parquet' \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
