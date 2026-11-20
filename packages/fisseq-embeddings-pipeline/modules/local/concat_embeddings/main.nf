// CONCAT_EMBEDDINGS. Stacks one experiment's per-well EMBED_CELLS tables
// into its embeddings.parquet, the cellDINO track's cell table. The workflow
// hands it an experiment's wells only once every one of them has embedded:
// an experiment with a failed well never gets here, so it never publishes a
// partial table.

include { threadEnv } from '../../../../fisseq-common/nextflow/modules/local/functions'

process CONCAT_EMBEDDINGS {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/embeddings/${batch_stem}" }, mode: params.publish_mode
    tag "${batch_stem}"

    input:
    tuple val(batch_stem), path(well_embeddings, stageAs: 'wells/*')

    output:
    tuple val(batch_stem), path("embeddings.parquet"), emit: embeddings

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.concat_embeddings \\
        output_dir=. \\
        inputs_dir=wells \\
        random_seed=${params.random_seed}
    """
}
