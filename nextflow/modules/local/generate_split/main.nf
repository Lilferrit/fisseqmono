// GENERATE_SPLIT: one bootstrap replicate's stratified 50/50 pseudo-replicate split of an
// experiment's QC-passed cells (fisseq_common.stages.generatesplit), seeded with
// random_seed + rep. Reads filtered_keys.parquet alone; each half is written as cell keys,
// which AGGREGATE_HALF applies with a semi-join.
//
// Shared by both pipelines. conf/modules.config sets ext.entry and publishDir.

include { threadEnv } from '../functions'

process GENERATE_SPLIT {
    tag "${batch_stem} rep${rep}"
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), path(filtered_keys), val(rep)

    output:
    tuple val(batch_stem), val(rep), path("half1.parquet"), path("half2.parquet"), emit: split

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m ${task.ext.entry} \\
        output_dir=. \\
        filtered_keys_file=${filtered_keys} \\
        label_column=${params.filter_label_column} \\
        bootstrap_idx=${rep} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
