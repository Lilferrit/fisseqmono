// COMBAT_OVWT: every experiment's OvWT results.parquet, stacked and ComBat-corrected with the
// variant as a covariate (python -m fisseq_common.stages.combatovwt). The three AUROC scores
// are corrected jointly; auroc_folds is dropped. One task per run (per track), after every
// experiment's OVWT_BATCHWISE; writes one table, one row per (variant, experiment).
//
// batch_stems and results are in the same order (gatherBatches in ../functions.nf). Shared by
// both pipelines; conf/modules.config sets publishDir.

include { threadEnv; hydraList } from '../functions'

process COMBAT_OVWT {
    tag "${batch_stems.size()} experiments"
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"

    input:
    tuple val(batch_stems), path(results, stageAs: 'inputs/?/*')

    output:
    path("ovwt_scores.parquet"), emit: scores

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_common.stages.combatovwt \\
        output_dir=. \\
        ${hydraList('input_files', results)} \\
        ${hydraList('batch_names', batch_stems)} \\
        label_column=${params.filter_label_column} \\
        mean_only=${params.combat_mean_only} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
