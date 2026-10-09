// EmbeddingsPipeline -- the whole DAG. See docs/embeddings-pipeline/architecture.md for the
// picture.
//
//   PLAN_EXPERIMENTS -> BUILD_CELL_IMAGES -> BUILD_CELL_METADATA -> QC_FILTER
//
// QC_FILTER is where the two tracks fan out. It hangs off
// BUILD_CELL_METADATA (a flat projection of cell_table.parquet), so an
// EMBED_CELLS failure can't take the CellProfiler track down with it. The
// cellDINO track's per-tile shards are cut inside BUILD_CELL_IMAGES' own
// nested snakemake (make_cell_shard), not by a stage here.
//
//   cellDINO:  EMBED_CELLS -> NORMALIZE ->
//              {OVWT_BATCHWISE, feature selection}
//   CP track:  BUILD_CP_FEATURES -> NORMALIZE_CP_FEATURES ->
//              {AGGREGATE_FEATURE_TYPE_CP_FEATURES, OVWT_BATCHWISE_CP_FEATURES}
//
// Everything downstream of EMBED_CELLS is fisseq-data-pipeline's: the same shared modules
// (QC_FILTER, NORMALIZE on the wildtype cells, OVWT_BATCHWISE, the bootstrap feature
// selection ending in FINALIZE_FEATURE_SELECT), the same parameters and the same publish
// layout. Only the cell table differs: embedding dimensions instead of CellProfiler features.
//
// Every output is per experiment: aggregating across experiments is the
// fisseqborn package's job (`fisseqborn-global`). Every task carries
// errorStrategy 'ignore', so independent branches keep running when one
// fails -- check the run report for failures.

// Shared modules (one copy for both pipelines) live in fisseq-common's nextflow/ directory;
// this pipeline's per-process settings for them are in conf/modules.config.
include { PLAN_EXPERIMENTS } from '../modules/local/plan_experiments/main.nf'
include { BUILD_CELL_IMAGES } from '../modules/local/build_cell_images/main.nf'
include { BUILD_CELL_METADATA } from '../modules/local/build_cell_metadata/main.nf'
include { EMBED_CELLS } from '../modules/local/embed_cells/main.nf'
include { BUILD_CP_FEATURES } from '../modules/local/build_cp_features/main.nf'
include { QC_FILTER } from '../../fisseq-common/nextflow/modules/local/qc_filter/main.nf'
include { FILTER as NORMALIZE } from '../../fisseq-common/nextflow/modules/local/filter/main.nf'
include { FILTER as NORMALIZE_CP_FEATURES } from '../../fisseq-common/nextflow/modules/local/filter/main.nf'
include { OVWT_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/ovwt_batchwise/main.nf'
include { OVWT_BATCHWISE as OVWT_BATCHWISE_CP_FEATURES } from '../../fisseq-common/nextflow/modules/local/ovwt_batchwise/main.nf'
include { AGGREGATE as AGGREGATE_FEATURE_TYPE_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/aggregate/main.nf'
include { AGGREGATE as AGGREGATE_FEATURE_TYPE_PASSTHROUGH } from '../../fisseq-common/nextflow/modules/local/aggregate/main.nf'
include { AGGREGATE as AGGREGATE_HALF_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/aggregate/main.nf'
include { AGGREGATE as AGGREGATE_FEATURE_TYPE_CP_FEATURES } from '../../fisseq-common/nextflow/modules/local/aggregate/main.nf'
include { GENERATE_SPLIT as GENERATE_SPLIT_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/generate_split/main.nf'
include { CORRELATE_FEATURES as CORRELATE_FEATURES_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/correlate_features/main.nf'
include { BLOCKLIST as BLOCKLIST_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/blocklist/main.nf'
include { COMBINE_BLOCKLISTS as COMBINE_BLOCKLISTS_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/combine_blocklists/main.nf'
include { FINALIZE_FEATURE_SELECT as FINALIZE_FEATURE_SELECT_BATCHWISE } from '../../fisseq-common/nextflow/modules/local/finalize_feature_select/main.nf'

// Nextflow CLI overrides (--run_ovwt false) arrive as the Groovy-truthy String "false", so
// every gate is coerced before use.
def asBool(v) {
    v == null ? false : v.toString().toBoolean()
}

// The profile is read only by BUILD_CELL_IMAGES' nested snakemake, deep inside an
// errorStrategy 'ignore' task, so a malformed config.yaml would otherwise surface as both
// experiments silently ignored. Checked here, on the head node, where the path is readable
// without a container bind.
def checkStarcallProfile(String dir) {
    def config = file("${dir}/config.yaml")
    if (!config.exists()) {
        error("starcall_profile ${dir} has no config.yaml")
    }
    def parsed = null
    try {
        parsed = new org.yaml.snakeyaml.Yaml().load(config.text)
    }
    catch (Exception e) {
        error("starcall_profile ${config} is not valid YAML: ${e.message}")
    }
    if (!(parsed instanceof Map)) {
        error("starcall_profile ${config} must be a YAML mapping of snakemake flags")
    }
}

workflow EmbeddingsPipeline {
    main:
    if (params.starcall_profile) {
        checkStarcallProfile(params.starcall_profile.toString())
    }
    // Validation and per-stage routing happen in Python
    // (fisseq_embeddings_pipeline.config.experiments), over the run's params
    // serialized to JSON.
    def params_json = file("${workflow.workDir}/fisseq_params.json")
    params_json.text = groovy.json.JsonOutput.toJson(params)
    plans = PLAN_EXPERIMENTS(params_json).plans
        .flatMap { f -> new groovy.json.JsonSlurperClassic().parseText(f.text) }

    cell_images = BUILD_CELL_IMAGES(plans)  // (batch_stem, cell_table, shards, phenotyping_dir)
    cell_tables = cell_images.map { stem, cell_table, _shards, _pheno_dir -> tuple(stem, cell_table) }

    metadata = BUILD_CELL_METADATA(
        plans.map { p -> tuple(p.batch_stem, p.cell_table_args) }.join(cell_tables)
    )
    qc = QC_FILTER(metadata)
    // Only the join key; the other two QC outputs are report files.
    qc_passed = qc.map { stem, filtered, _barcode_counts, _variants -> tuple(stem, filtered) }

    // ── cellDINO track ───────────────────────────────────────────────────
    embeddings = EMBED_CELLS(
        cell_images.map { stem, _cell_table, shards, pheno_dir -> tuple(stem, shards, pheno_dir) }
    )

    // embeddings_only stops the cellDINO track here and skips the CP track:
    // for when all you want is the embeddings (and what the containerized
    // real-starcall integration test uses).
    if (!asBool(params.embeddings_only)) {
        // The shared FILTER module: the cells are EMBED_CELLS' embeddings, the QC-passed
        // table QC_FILTER's; the normalizer is fit on the wildtype cells.
        normalized = NORMALIZE(embeddings.join(qc_passed))  // (stem, filtered_keys, normalizer)
        // tuple(batch_stem, embeddings, filtered_keys, normalizer)
        norm_ch = embeddings.join(normalized)

        if (asBool(params.run_ovwt)) {
            OVWT_BATCHWISE(norm_ch)
        }

        // Feature selection, exactly as fisseq-data-pipeline runs it -- decomposed
        // bootstrap + per-feature-type pipeline.
        //   Stage 1:    per-feature-type full aggregation.
        //   Stage 2a-d: per-bootstrap split -> per-half aggregation -> correlation
        //               -> per-feature-type blocklist (gathered over bootstraps).
        //   Stage 3:    combine per-feature-type blocklists.
        //   Stage 4:    join stage-1 aggregates, apply combined blocklist,
        //               z-score to the synonymous variants.
        if (asBool(params.run_feature_selection)) {
            feature_types_ch = channel.fromList(params.feature_select_types)
            // Explicit cast: CLI overrides (--feature_select_bootstrap_reps 3)
            // arrive as Strings and silently produce a bogus range if left
            // uncoerced in a Groovy IntRange.
            bootstrap_ch = channel.of(1..(params.feature_select_bootstrap_reps as int))

            // Stage 1: full per-feature-type aggregation, one task per
            // (experiment, feature_type). The published aggregates are z-scored
            // against the experiment's synonymous variants (conf/modules.config).
            // AGGREGATE's input: (batch_stem, cells, keys, normalizer, rep, half, split,
            // method); rep = half = 0 and no split file aggregate every cell.
            agg_input_ch = norm_ch
                .combine(feature_types_ch)
                .map { batch_stem, cells, keys, normalizer, feature_type ->
                    tuple(batch_stem, cells, keys, normalizer, 0, 0, [], feature_type)
                }
            AGGREGATE_FEATURE_TYPE_BATCHWISE(agg_input_ch)
            agg_ch = AGGREGATE_FEATURE_TYPE_BATCHWISE.out.aggregate
                .map { batch_stem, _rep, feature_type, _half, agg_file -> tuple(batch_stem, feature_type, agg_file) }

            // Stage 1b: passthrough aggregation. Same process, and deliberately
            // nothing downstream of it but the stage-4 join -- passthrough types
            // never reach the bootstrap halves, the correlation, or the
            // blocklist, which is the whole point of the second list. They are
            // not z-scored (p-values must keep their own scale) and publish to
            // their own directory, so no aggregates/ glob ever mixes the two.
            passthrough_types_ch = channel.fromList(params.feature_select_passthrough_types)
            pt_agg_input_ch = norm_ch
                .combine(passthrough_types_ch)
                .map { batch_stem, cells, keys, normalizer, feature_type ->
                    tuple(batch_stem, cells, keys, normalizer, 0, 0, [], feature_type)
                }
            AGGREGATE_FEATURE_TYPE_PASSTHROUGH(pt_agg_input_ch)
            pt_agg_ch = AGGREGATE_FEATURE_TYPE_PASSTHROUGH.out.aggregate
                .map { batch_stem, _rep, feature_type, _half, agg_file -> tuple(batch_stem, feature_type, agg_file) }

            // Stage 2a: one 50/50 split per (experiment, bootstrap replicate).
            split_input_ch = norm_ch
                .combine(bootstrap_ch)
                .map { batch_stem, _cells, keys, _normalizer, bootstrap_idx ->
                    tuple(batch_stem, keys, bootstrap_idx)
                }
            GENERATE_SPLIT_BATCHWISE(split_input_ch)
            split_ch = GENERATE_SPLIT_BATCHWISE.out.split  // (batch_stem, bootstrap_idx, half1, half2)

            // Stage 2b: expand each split into two per-half tuples, cross with
            // feature types, and re-attach the experiment's cells, keys and normalizer
            // via .combine(norm_ch, by: 0) (keyed on batch_stem -- norm_ch has exactly
            // one entry per experiment, so this is a per-experiment broadcast,
            // not a fan-out).
            // NOTE: .join() is NOT a broadcast operator -- for a many-to-one key
            // relationship like this one it silently keeps only one match per key
            // and drops the rest, starving every downstream stage. Only use
            // .join() where both sides are already collapsed to exactly one item
            // per key (see the finalize-stage joins below).
            half_ch = split_ch.flatMap { batch_stem, bootstrap_idx, half1, half2 ->
                [
                    tuple(batch_stem, bootstrap_idx, 1, half1),
                    tuple(batch_stem, bootstrap_idx, 2, half2),
                ]
            }
            agg_half_input_ch = half_ch
                .combine(feature_types_ch)
                // (batch_stem, bootstrap_idx, half_num, index_file, feature_type)
                .combine(norm_ch, by: 0)
                // (batch_stem, bootstrap_idx, half_num, index_file, feature_type, cells, keys, normalizer)
                .map { batch_stem, bootstrap_idx, half_num, index_file, feature_type, cells, keys, normalizer ->
                    tuple(batch_stem, cells, keys, normalizer, bootstrap_idx, half_num, index_file, feature_type)
                }
            AGGREGATE_HALF_BATCHWISE(agg_half_input_ch)
            half_agg_ch = AGGREGATE_HALF_BATCHWISE.out.aggregate
            // (batch_stem, bootstrap_idx, feature_type, half_num, half_agg_file)

            // Stage 2c: group by (batch_stem, bootstrap_idx, feature_type) --
            // exactly 2 per group -- pair by half_num (not arrival order) before
            // correlating.
            corr_input_ch = half_agg_ch
                .groupTuple(by: [0, 1, 2])
                .map { batch_stem, bootstrap_idx, feature_type, half_nums, half_files ->
                    def pairs = [half_nums, half_files].transpose().sort { pair -> pair[0] }
                    tuple(batch_stem, bootstrap_idx, feature_type, pairs[0][1], pairs[1][1])
                }
            CORRELATE_FEATURES_BATCHWISE(corr_input_ch)
            corr_ch = CORRELATE_FEATURES_BATCHWISE.out.correlations  // (batch_stem, feature_type, corr_file)

            // Stage 2d: group by (batch_stem, feature_type) -- gathers all
            // bootstrap replicates. THE one intentional synchronization point,
            // scoped to this stage only.
            blocklist_input_ch = corr_ch.groupTuple(by: [0, 1])
            BLOCKLIST_BATCHWISE(blocklist_input_ch)
            bl_ch = BLOCKLIST_BATCHWISE.out.blocklist  // (batch_stem, blocklist_file)

            // Stage 3: group by batch_stem -- gathers all feature types.
            combine_bl_input_ch = bl_ch.groupTuple(by: 0)
            COMBINE_BLOCKLISTS_BATCHWISE(combine_bl_input_ch)
            combined_bl_ch = COMBINE_BLOCKLISTS_BATCHWISE.out.blocklist  // (batch_stem, combined_blocklist_file)

            // Stage 4: group stage-1 output by batch_stem (all feature types'
            // full aggregates), join NORMALIZE's keys (for the per-variant metadata) and
            // stage-3's combined blocklist.
            // groupTuple() on an empty channel emits nothing, so with an empty
            // params.feature_select_passthrough_types this side of the join has no
            // entry for any batch. `remainder: true` is what keeps that from
            // starving FINALIZE_FEATURE_SELECT entirely (the stage would silently
            // never run); the null it yields instead becomes an empty file list.
            pt_files_ch = pt_agg_ch
                .map { batch_stem, _feature_type, agg_file -> tuple(batch_stem, agg_file) }
                .groupTuple(by: 0)
            finalize_input_ch = agg_ch
                .map { batch_stem, _feature_type, agg_file -> tuple(batch_stem, agg_file) }
                .groupTuple(by: 0)
                .join(norm_ch.map { batch_stem, _cells, keys, _normalizer -> tuple(batch_stem, keys) })
                .join(combined_bl_ch)
                .join(pt_files_ch, remainder: true)
                .map { batch_stem, agg_files, keys, combined_bl_file, pt_files ->
                    tuple(batch_stem, agg_files, pt_files ?: [], keys, combined_bl_file)
                }
            FINALIZE_FEATURE_SELECT_BATCHWISE(finalize_input_ch)
        }

        // The CellProfiler-feature track: the same normalization, aggregation and OvWT on
        // the CellProfiler features of the experiments that set cp_features, without the
        // bootstrap feature selection.
        cp_features = BUILD_CP_FEATURES(
            plans.filter { p -> p.cp_features }
                .map { p -> tuple(p.batch_stem, p.cell_table_args) }
                .join(cell_tables)
        )
        cp_normalized = NORMALIZE_CP_FEATURES(cp_features.join(qc_passed))
        cp_norm_ch = cp_features.join(cp_normalized)
        AGGREGATE_FEATURE_TYPE_CP_FEATURES(
            cp_norm_ch
                .combine(channel.fromList(params.feature_select_types_cp_features))
                .map { batch_stem, cells, keys, normalizer, feature_type ->
                    tuple(batch_stem, cells, keys, normalizer, 0, 0, [], feature_type)
                }
        )
        if (asBool(params.run_ovwt)) {
            OVWT_BATCHWISE_CP_FEATURES(cp_norm_ch)
        }
    }
}
