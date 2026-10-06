// AGGREGATE_HALF: one aggregation method over one cell subset, written leanly as
// [label] + that method's stat columns (fisseq_common.stages.aggregate.aggregate_cells).
// One method per task, together with params.aggregate_feature_chunk_size, keeps the
// reference-based aggregators' peak memory bounded.
//
// `split` is one GENERATE_SPLIT half, or [] to aggregate every QC-passed cell; rep and half
// are 0 then. Shared by both pipelines, under each pipeline's own process names:
//   data pipeline:       AGGREGATE_HALF_BATCHWISE, AGGREGATE_FEATURE_TYPE_BATCHWISE,
//                        AGGREGATE_FEATURE_TYPE_PASSTHROUGH
//   embeddings pipeline: AGGREGATE_HALF, AGGREGATE_PASSTHROUGH
// conf/modules.config sets ext.entry, ext.cells_key / ext.split_key (the entry point's keys
// for `cells` and `split`), ext.args, ext.seed (default params.random_seed) and publishDir.

include { threadEnv } from '../functions'

process AGGREGATE_HALF {
    tag "${batch_stem} ${method}${split ? " rep${rep} half${half}" : ''}"
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), path(cells), path(filtered_keys), path(normalizer), val(rep), val(half), path(split, stageAs: 'split/*'), val(method)

    output:
    tuple val(batch_stem), val(rep), val(method), val(half), path("${method}.parquet"), emit: aggregate

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def split_arg = split ? "${task.ext.split_key ?: 'split_file'}=${split}" : ''
    def chunk_size = params.aggregate_feature_chunk_size == null ? 'null' : params.aggregate_feature_chunk_size
    def seed = task.ext.seed != null ? task.ext.seed : params.random_seed
    """
    ${threadEnv(task.cpus)}
    python -m ${task.ext.entry} \\
        output_dir=. \\
        ${task.ext.cells_key ?: 'cells_file'}=${cells} \\
        filtered_keys_file=${filtered_keys} \\
        normalizer_file=${normalizer} \\
        aggregator=${method} \\
        ${split_arg} \\
        label_column=${params.filter_label_column} \\
        feature_chunk_size=${chunk_size} \\
        output_name=${method} \\
        ${args} \\
        random_seed=${seed}
    """
}
