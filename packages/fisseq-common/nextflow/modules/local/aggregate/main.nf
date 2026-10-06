// AGGREGATE: one aggregation method over one cell subset, written leanly as
// [label] + that method's stat columns (python -m fisseq_common.stages.aggregate). One method
// per task, together with params.aggregate_feature_chunk_size, keeps the reference-based
// aggregators' peak memory bounded.
//
// `split` is one GENERATE_SPLIT half, or [] to aggregate every QC-passed cell; rep and half
// are 0 then. Each pipeline runs it under three names:
//   AGGREGATE_FEATURE_TYPE_BATCHWISE    every cell, z-scored against the synonymous variants
//   AGGREGATE_FEATURE_TYPE_PASSTHROUGH  every cell, raw (passthrough methods)
//   AGGREGATE_HALF_BATCHWISE            one bootstrap half, raw
// conf/modules.config sets ext.args (join_keys, feature_selector, downsample_wt,
// normalize_to_synonymous), ext.seed (default params.random_seed) and publishDir.

include { threadEnv } from '../functions'

process AGGREGATE {
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
    def split_arg = split ? "split_file=${split}" : ''
    def chunk_size = params.aggregate_feature_chunk_size == null ? 'null' : params.aggregate_feature_chunk_size
    def seed = task.ext.seed != null ? task.ext.seed : params.random_seed
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_common.stages.aggregate \\
        output_dir=. \\
        cells_file=${cells} \\
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
