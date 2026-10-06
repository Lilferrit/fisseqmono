// FILTER: keep the QC-passed cells and fit the normalizer on the wildtype cells
// (python -m fisseq_common.stages.filter). Publishes only filtered_keys.parquet and
// normalizer.parquet; every consumer rebuilds the normalized table by joining back.
//
// `cells` is the cell feature table, `qc_passed` QC_FILTER's filtered_cells (the data
// pipeline passes filtered_cells as both). Shared by both pipelines, under each pipeline's own
// process names (data: NORMALIZE; embeddings: NORMALIZE, NORMALIZE_CP_FEATURES).
// conf/modules.config sets ext.args (join_keys; the data pipeline's batch_name) and publishDir.

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
    python -m fisseq_common.stages.filter \\
        output_dir=. \\
        cells_file=${cells} \\
        qc_passed_file=${qc_passed} \\
        label_column=${params.filter_label_column} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
