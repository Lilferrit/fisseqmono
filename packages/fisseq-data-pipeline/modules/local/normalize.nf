nextflow.enable.dsl = 2

// NORMALIZE: publishes only the QC-passed cells' keys and the wildtype-fitted
// normalizer to normalization/<batch_stem>/. No normalized copy of the cells is
// written; every consumer rebuilds it from QC_FILTER's filtered_cells.parquet and
// these two files (fisseq_data_pipeline.cells).
process NORMALIZE {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/normalization/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(filtered_cells)

    output:
    tuple val(batch_stem), path("filtered_keys.parquet"), path("normalizer.parquet"), emit: normalized

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "Starting NORMALIZE for ${batch_stem}"
    python -m fisseq_data_pipeline.normalize \\
        output_dir=. \\
        input_file=${filtered_cells} \\
        batch_name=${batch_stem}
    """
}
