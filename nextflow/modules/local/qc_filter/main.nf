// QC_FILTER: edit-distance, barcode-count and variant-barcode-count filters on one
// experiment's cells (fisseq_common.stages.qcfilter).
//
// Shared by both pipelines. Each pipeline's conf/modules.config sets:
//   ext.entry   the pipeline's `python -m` module (its config names the input columns);
//   ext.args    pipeline-specific overrides (the data pipeline's variant and
//               pseudo-variant downsampling);
//   publishDir  <pipeline_dir>/qc_filter/<batch_stem>.

include { threadEnv } from '../functions'

process QC_FILTER {
    tag "${batch_stem}"
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), path(cells)

    output:
    tuple val(batch_stem), path("filtered_cells.parquet"), path("barcode_counts.parquet"), path("variants_per_barcode.parquet"), emit: qc

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m ${task.ext.entry} \\
        output_dir=. \\
        'cell_files=[${cells}]' \\
        bc_threshold=${params.barcode_count_threshold} \\
        variant_bc_threshold=${params.variant_barcode_count_threshold} \\
        edit_distance_threshold=${params.edit_distance_threshold} \\
        label_column=${params.filter_label_column} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
