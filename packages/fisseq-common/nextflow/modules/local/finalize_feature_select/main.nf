// FINALIZE_FEATURE_SELECT: one experiment's per-variant table (python -m
// fisseq_common.stages.finalize). Joins the per-method aggregates, drops the columns the
// combined blocklist marks as not reproducible, z-scores to the synonymous variants, adds the
// impact score and per-variant metadata (from filtered_keys), and joins the passthrough
// aggregates raw, last.
//
// Shared by both pipelines. conf/modules.config sets publishDir.

include { threadEnv } from '../functions'

process FINALIZE_FEATURE_SELECT {
    tag "${batch_stem}"
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"

    input:
    tuple val(batch_stem), path(feature_type_files, stageAs: 'ft/*'), path(passthrough_files, stageAs: 'pt/*'), path(filtered_keys), path(block_list_file)

    output:
    tuple val(batch_stem), path("output.parquet"), emit: selected

    when:
    task.ext.when == null || task.ext.when

    script:
    // passthrough_files is empty whenever params.feature_select_passthrough_types is, so pt/
    // may not exist: mkdir -p keeps the glob well-formed, and the stage treats a glob
    // matching nothing as "no passthrough columns".
    """
    ${threadEnv(task.cpus)}
    mkdir -p pt
    python -m fisseq_common.stages.finalize \\
        output_dir=. \\
        'feature_type_files=ft/*.parquet' \\
        'passthrough_feature_type_files=pt/*.parquet' \\
        filtered_keys_file=${filtered_keys} \\
        block_list_file=${block_list_file} \\
        label_column=${params.filter_label_column} \\
        compute_impact_score=true \\
        random_seed=${params.random_seed}
    """
}
