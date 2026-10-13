// BUILD_CELL_METADATA. Projects BUILD_CELL_IMAGES' cell_table.parquet down
// to the seven meta_* columns QC_FILTER reads (metadata.parquet), so
// QC_FILTER is the shared fan-out point rather than the image-reading
// cellDINO track. EMBED_CELLS builds the same meta_* columns from each
// shard's meta.json, so both tracks see identical values.
//
// QC can't just read cell_table.parquet directly: qcfilter's filter_columns
// drops the unprefixed well/tile/tile_cell_index columns that become
// filter.py's JOIN_KEYS. See cell_metadata.py and docs/architecture.md.

include { threadEnv } from '../../../../fisseq-common/nextflow/modules/local/functions'

process BUILD_CELL_METADATA {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cell_metadata/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(cell_table_args), path(cell_table)

    output:
    tuple val(batch_stem), path("metadata.parquet"), emit: metadata

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.cell_metadata \\
        output_dir=. \\
        cell_table=${cell_table} \\
        batch_stem=${batch_stem} \\
        ${cell_table_args} \\
        random_seed=${params.random_seed}
    """
}
