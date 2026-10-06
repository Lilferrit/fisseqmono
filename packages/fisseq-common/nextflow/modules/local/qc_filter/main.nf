// QC_FILTER: edit-distance, barcode-count and variant-barcode-count filters on one
// experiment's cells, plus the optional variant and pseudo-variant downsampling
// (python -m fisseq_common.stages.qcfilter).
//
// Shared by both pipelines. Each pipeline's conf/modules.config sets:
//   ext.args    its input column names, sort_output_by and assign_cell_index;
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
    // qc_downsample_amounts: null -> Hydra null; a scalar interpolates unquoted; a Groovy
    // List becomes a single-quoted Hydra bracket-list. The two class lists are always
    // non-empty List<String>, each element quoted ("Single Missense" has a space).
    def amounts = (params.qc_downsample_amounts == null)
        ? 'null'
        : (params.qc_downsample_amounts instanceof List)
            ? "'[${params.qc_downsample_amounts.join(',')}]'"
            : "${params.qc_downsample_amounts}"
    def variantClasses = params.qc_variant_downsample_classes.collect { c -> "\"${c}\"" }.join(',')
    def downsampleClasses = params.qc_downsample_classes.collect { c -> "\"${c}\"" }.join(',')
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_common.stages.qcfilter \\
        output_dir=. \\
        'cell_files=[${cells}]' \\
        bc_threshold=${params.barcode_count_threshold} \\
        variant_bc_threshold=${params.variant_barcode_count_threshold} \\
        edit_distance_threshold=${params.edit_distance_threshold} \\
        label_column=${params.filter_label_column} \\
        n_variants=${params.qc_n_variants} \\
        variant_downsample_classes='[${variantClasses}]' \\
        variant_downsample_mode=${params.qc_variant_downsample_mode} \\
        downsample_amounts=${amounts} \\
        downsample_classes='[${downsampleClasses}]' \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
