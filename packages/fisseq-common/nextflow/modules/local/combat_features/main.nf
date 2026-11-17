// COMBAT_FEATURES: every experiment's per-variant aggregates (the aggregate stage's
// aggregates/<method>.parquet, every method, every feature), joined per experiment, stacked and
// ComBat-corrected with the variant as a covariate (python -m fisseq_common.stages.combatfeatures).
// One task per run (per track), after every experiment's aggregates; writes one table, one row
// per (variant, experiment). Variants in a single experiment, rows with no value and features
// holding a null are dropped first (the task log names them).
//
// aggregates and batch_stems (one per file, so an experiment repeats once per method) are in
// the same order (gatherBatches in ../functions.nf). Shared by both pipelines;
// conf/modules.config sets publishDir.

include { threadEnv; hydraList } from '../functions'

process COMBAT_FEATURES {
    tag "${batch_stems.size()} experiments"
    errorStrategy 'ignore'
    label 'process_high'
    container "${params.container_image}"

    input:
    tuple val(batch_stems), path(aggregates, stageAs: 'inputs/?/*')

    output:
    path("features.parquet"), emit: features

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_common.stages.combatfeatures \\
        output_dir=. \\
        ${hydraList('input_files', aggregates)} \\
        ${hydraList('batch_names', batch_stems)} \\
        label_column=${params.filter_label_column} \\
        mean_only=${params.combat_mean_only} \\
        ${args} \\
        random_seed=${params.random_seed}
    """
}
