nextflow.enable.dsl = 2

// FINALIZE_FEATURE_SELECT: wraps python -m fisseq_data_pipeline.featureselect.
// Feature-selection stage 4 -- joins one batch's per-type aggregates, applies
// the combined blocklist, and z-scores to the synonymous baseline; passthrough aggregates are joined in raw afterwards.
process FINALIZE_FEATURE_SELECT {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/${publish_subdir}" }, mode: 'copy'

    input:
    tuple val(batch_key), path(feature_type_files), path(passthrough_files, stageAs: 'pt/*'), path(cells), path(filtered_keys), path(normalizer), path(block_list_file), val(publish_subdir)

    output:
    tuple val(batch_key), path("output.parquet")

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "Starting FINALIZE_FEATURE_SELECT for ${batch_key}"
    mkdir -p ft
    mv ${feature_type_files} ft/
    # passthrough_files is staged straight into pt/ (stageAs above) rather than
    # moved, because the list is empty whenever params.feature_select_passthrough_types
    # is -- an `mv` with no arguments would fail the task. mkdir -p keeps the
    # glob well-formed in that case; featureselect.py treats a zero-match
    # passthrough glob as a warning, not an error.
    mkdir -p pt
    python -m fisseq_data_pipeline.featureselect \\
        output_dir=. \\
        output_root=out \\
        cells_file=${cells} \\
        filtered_keys_file=${filtered_keys} \\
        normalizer_file=${normalizer} \\
        "feature_type_files=ft/*.parquet" \\
        "passthrough_feature_type_files=pt/*.parquet" \\
        block_list_file=${block_list_file} \\
        compute_impact_score=true \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    mv out.*.parquet output.parquet
    """
}
