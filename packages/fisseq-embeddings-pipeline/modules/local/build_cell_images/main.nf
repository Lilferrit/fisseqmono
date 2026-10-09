// BUILD_CELL_IMAGES -- the ONLY task that touches starcall-workflow's tree
// or runs a nested snakemake. Three phases:
//
//   1. build_cell_images_prepare resolves phenotyping_dir/segmentation_dir/
//      sequencing_dir (resolved_dirs.env) and writes the nested snakemake's
//      --configfile (snakemake_config.yaml) -- plus, in cluster mode, the
//      jobscript every starcall child job re-enters the image through.
//   2. Snakemake against the REAL data dirs (so its own mtime caching
//      reuses whatever is already computed), of starcall-workflow's own
//      Snakefile -- cloned into the image at a pinned commit, unmodified --
//      plus this repo's rules (snakemake/Snakefile,
//      task.ext.fisseq_snakefile), in two passes: fisseq_shards (every
//      well's WebDataset shards, alone, so use_corrected regenerating its
//      temp() corrected_tiles.tif doesn't rerun CellProfiler), then
//      tiles_manifest.csv, whose rule
//      lists every tile's cell and reads tables and every well's shards as
//      inputs, so snakemake builds the whole DAG from the grid.
//      See docs/architecture.md decision 17.
//   3. build_cell_images_table joins the per-tile CSVs into
//      cell_table.parquet, plus shards.parquet naming every shard.
//
// Local mode (no params.starcall_profile): every starcall rule runs inside
// this one task, `--cores params.snakemake_cores`. Cluster mode: the nested
// snakemake gets `--profile params.starcall_profile`, which says how to
// submit jobs (it lives outside this repo -- nothing scheduler-specific is
// here), plus our --jobscript, which re-enters the image on each node.
// See docs/nextflow.md.
//
// One starcall_workflow_dir per experiment, and never two runs against the
// same tree at once: the --unlock below clears a stale lock from a killed
// run unconditionally, which is only safe under that rule.

include { threadEnv; hydraList } from '../../../../fisseq-common/nextflow/modules/local/functions'

process BUILD_CELL_IMAGES {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cell_images/${plan.batch_stem}" }, mode: 'copy', pattern: '{cell_table,shards}.parquet'

    input:
    val(plan)

    output:
    tuple val(plan.batch_stem), path("cell_table.parquet"), path("shards.parquet"), env("phenotyping_dir"), emit: cell_images

    script:
    def starcall_dir = plan.starcall_workflow_dir
    def cache_dir = params.snakemake_cache_dir ?: "${params.pipeline_dir}/.snakemake_cache"
    def snakemake = task.ext.snakemake_bin
    def cluster_mode = params.starcall_profile as boolean
    // A list from params.yaml, or a comma-separated string from the command
    // line; absent from an older params.yaml.
    def site_binds = params.starcall_job_binds ?: []
    if (!(site_binds instanceof Collection)) {
        site_binds = site_binds.toString().tokenize(',')*.trim()
    }
    def jobscript_args = !cluster_mode ? '' : [
        "starcall_job_image=${params.starcall_job_image}",
        "starcall_container_bin=${params.starcall_container_bin}",
        "starcall_job_gpu=${params.starcall_gpu.toString().toBoolean()}",
        hydraList('jobscript_binds', [starcall_dir, cache_dir] + site_binds),
    ].join(' ')
    // In cluster mode --cores is left to the profile: there it is the budget
    // across all submitted jobs, and a local-sized value would silently cap
    // every rule's own threads.
    def submission = cluster_mode
        ? "--profile '${params.starcall_profile}' --jobscript \"\$PWD/starcall_jobscript.sh\""
        : "--cores ${params.snakemake_cores}"
    // --retries reruns a failed starcall job with attempt + 1, and the
    // wrapper Snakefile doubles every rule's mem_mb per attempt. null leaves it
    // to the profile's own `retries:` (or snakemake's default, none).
    def retries = params.starcall_retries == null ? '' : "--retries ${params.starcall_retries}"
    """
    set -euo pipefail
    ${threadEnv(task.cpus)}

    # Snakemake mkdir's \$XDG_CACHE_HOME/snakemake (falling back to
    # \$HOME/.cache) before it parses a single rule, with no flag to move it,
    # and under Apptainer \$HOME is often a read-only cluster home. \$HOME
    # moves too because --use-conda's bare `conda` reads ~/.condarc.
    export XDG_CACHE_HOME="${cache_dir}"
    export HOME="${cache_dir}/home"
    mkdir -p "\$XDG_CACHE_HOME" "\$HOME"

    python -m fisseq_embeddings_pipeline.build_cell_images_prepare \\
        output_dir=. \\
        ${plan.cell_images_args} \\
        ${jobscript_args} \\
        random_seed=${params.random_seed}

    # Resolved by phase 1, not recomputed here. Exported so the
    # env("phenotyping_dir") output (EMBED_CELLS' bind path) sees it.
    source resolved_dirs.env
    export phenotyping_dir

    # A killed run leaves <starcall_workflow_dir>/.snakemake/locks behind and
    # the next one dies with "Directory cannot be locked".
    ${snakemake} \\
        --snakefile "${task.ext.fisseq_snakefile}" \\
        --directory "${starcall_dir}" \\
        --unlock \\
        --configfile "\$PWD/snakemake_config.yaml" \\
        || true

    # '--' stops --configfile (and --rerun-triggers) from swallowing the
    # target.
    nested_snakemake() {
        ${snakemake} \\
            --snakefile "${task.ext.fisseq_snakefile}" \\
            --directory "${starcall_dir}" \\
            ${submission} \\
            ${retries} \\
            --use-conda --conda-frontend conda \\
            --rerun-triggers mtime \\
            --rerun-incomplete \\
            --configfile "\$PWD/snakemake_config.yaml" \\
            -- \\
            "\$1"
    }

    # Two passes, shards first. make_cell_shard stitches each tile itself
    # from files starcall keeps, except with use_corrected: its corrected
    # images come from starcall's temp() corrected_tiles.tif, which cutting
    # missing shards regenerates. In the same DAG, snakemake would then
    # rerun every job downstream of it too -- CellProfiler included --
    # however complete their outputs ("Input files updated by another
    # job"). The fisseq_shards pass has nothing else in its DAG, and by the
    # manifest pass the temp file is gone again, so only tables that are
    # really missing get built. See docs/architecture.md decision 17.
    nested_snakemake fisseq_shards
    nested_snakemake "\$PWD/tiles_manifest.csv"

    python -m fisseq_embeddings_pipeline.build_cell_images_table \\
        output_dir=. \\
        random_seed=${params.random_seed}

    test -s cell_table.parquet
    """
}
