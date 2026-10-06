// FILTER_AGGREGATE. Aggregates -> filtered aggregates, plus the terminal
// passthrough view. Two outputs on purpose: filtered_aggregate.parquet
// carries only reproducible columns; aggregate_with_passthrough.parquet
// adds the passthrough methods. The split keeps passthrough columns out of
// anything that picks features with FEATURE_SELECTOR (which would match
// emb_0000_KSnegLogP) from filtered_aggregate.parquet.

include { threadEnv } from '../../../../../nextflow/modules/local/functions'

process FILTER_AGGREGATE {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(aggregate_parquet), path(blocklist_parquet), path(passthrough_parquets, stageAs: 'passthrough/*')

    output:
    tuple val(batch_stem), path("filtered_aggregate.parquet"), path("aggregate_with_passthrough.parquet"), emit: filtered

    script:
    // An empty pattern means "no passthrough columns"; the default.
    def pattern = passthrough_parquets ? "'passthrough_files=passthrough/*.parquet'" : 'passthrough_files=null'
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.filter_aggregate \\
        output_dir=. \\
        aggregate_file=${aggregate_parquet} \\
        blocklist_file=${blocklist_parquet} \\
        ${pattern} \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
