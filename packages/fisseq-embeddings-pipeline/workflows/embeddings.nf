// EmbeddingsPipeline -- the whole DAG. See docs/architecture.md for the
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
//   cellDINO:  EMBED_CELLS -> FILTER_EMBEDDINGS ->
//              {AGGREGATE_EMBEDDINGS, OVWT_BATCHWISE, reproducibility chain}
//   CP track:  BUILD_CP_FEATURES -> FILTER_CP_FEATURES ->
//              {AGGREGATE_CP_FEATURES, OVWT_BATCHWISE_CP_FEATURES}
//
// Every output is per experiment: aggregating across experiments is the
// fisseqborn package's job (`fisseqborn-global`). Every task carries
// errorStrategy 'ignore', so independent branches keep running when one
// fails -- check the run report for failures.

// Shared modules (one copy for both pipelines) live in the repo's nextflow/ directory; this
// pipeline's per-process settings for them are in conf/modules.config.
include { PLAN_EXPERIMENTS } from '../modules/local/plan_experiments/main.nf'
include { BUILD_CELL_IMAGES } from '../modules/local/build_cell_images/main.nf'
include { BUILD_CELL_METADATA } from '../modules/local/build_cell_metadata/main.nf'
include { QC_FILTER } from '../../../nextflow/modules/local/qc_filter/main.nf'
include { EMBED_CELLS } from '../modules/local/embed_cells/main.nf'
include { FILTER as FILTER_EMBEDDINGS } from '../../../nextflow/modules/local/filter/main.nf'
include { AGGREGATE_EMBEDDINGS } from '../modules/local/aggregate_embeddings/main.nf'
include { OVWT_BATCHWISE } from '../../../nextflow/modules/local/ovwt_batchwise/main.nf'
include { GENERATE_SPLIT } from '../../../nextflow/modules/local/generate_split/main.nf'
include { AGGREGATE_HALF } from '../../../nextflow/modules/local/aggregate_half/main.nf'
include { AGGREGATE_HALF as AGGREGATE_PASSTHROUGH } from '../../../nextflow/modules/local/aggregate_half/main.nf'
include { CORRELATE_FEATURES } from '../../../nextflow/modules/local/correlate_features/main.nf'
include { BLOCKLIST } from '../../../nextflow/modules/local/blocklist/main.nf'
include { COMBINE_BLOCKLISTS } from '../../../nextflow/modules/local/combine_blocklists/main.nf'
include { FILTER_AGGREGATE } from '../modules/local/filter_aggregate/main.nf'
include { BUILD_CP_FEATURES } from '../modules/local/build_cp_features/main.nf'
include { FILTER as FILTER_CP_FEATURES } from '../../../nextflow/modules/local/filter/main.nf'
include { AGGREGATE_CP_FEATURES } from '../modules/local/aggregate_cp_features/main.nf'
include { OVWT_BATCHWISE as OVWT_BATCHWISE_CP_FEATURES } from '../../../nextflow/modules/local/ovwt_batchwise/main.nf'

workflow EmbeddingsPipeline {
    main:
    // Validation and per-stage routing happen in Python
    // (fisseq_embeddings_pipeline.config.experiments), over the run's params
    // serialized to JSON.
    def params_json = file("${workflow.workDir}/fisseq_params.json")
    params_json.text = groovy.json.JsonOutput.toJson(params)
    plans = PLAN_EXPERIMENTS(params_json).plans
        .flatMap { f -> new groovy.json.JsonSlurperClassic().parseText(f.text) }

    cell_images = BUILD_CELL_IMAGES(plans)  // (batch_stem, cell_table, tiles, phenotyping_dir)
    cell_tables = cell_images.map { stem, cell_table, _tiles, _pheno_dir -> tuple(stem, cell_table) }

    metadata = BUILD_CELL_METADATA(
        plans.map { p -> tuple(p.batch_stem, p.cell_table_args) }.join(cell_tables)
    )
    qc = QC_FILTER(metadata)
    // Only the join key; the other two QC outputs are report files.
    qc_passed = qc.map { stem, filtered, _barcode_counts, _variants -> tuple(stem, filtered) }

    // ── cellDINO track ───────────────────────────────────────────────────
    embeddings = EMBED_CELLS(
        cell_images
            .map { stem, _cell_table, tiles, pheno_dir -> tuple(stem, tiles, pheno_dir) }
            .join(metadata)
    )

    // embeddings_only stops the cellDINO track here and skips the CP track:
    // for when all you want is the embeddings (and what the containerized
    // real-starcall integration test uses).
    // (.toString().toBoolean(): `--embeddings_only false` can arrive as the
    // string "false", which Groovy treats as true.)
    if (!params.embeddings_only.toString().toBoolean()) {
        filtered = FILTER_EMBEDDINGS(embeddings.join(qc_passed))  // (stem, filtered_keys, normalizer)
        // Consumers reconstruct the QC-passed, synonymous-corrected table
        // themselves from these three; none reads a pre-normalized copy.
        embed_and_filtered = embeddings.join(filtered)            // (stem, embeddings, filtered_keys, normalizer)
        aggregates = AGGREGATE_EMBEDDINGS(embed_and_filtered)
        OVWT_BATCHWISE(embed_and_filtered)

        // ── Reproducibility filtering ───────────────────────────────────
        // One split per bootstrap replicate, two halves per split, one
        // AGGREGATE_HALF task per (replicate, half, method). Bare emb_*
        // column names only when aggregate_methods is exactly ["median"],
        // mirroring AGGREGATE_EMBEDDINGS (conf/modules.config's bare_columns).
        def reps = params.reproducibility_bootstrap_reps as int
        def methods = params.aggregate_methods as List
        def passthrough_methods = (params.aggregate_methods_passthrough ?: []) as List

        splits = GENERATE_SPLIT(
            filtered.map { stem, keys, _normalizer -> tuple(stem, keys) }.combine(channel.of(1..reps))
        )
        halves = splits.flatMap { stem, rep, half1, half2 ->
            [tuple(stem, rep, 1, half1), tuple(stem, rep, 2, half2)]
        }
        // (stem, embeddings, keys, normalizer, rep, half, split, method)
        half_aggregates = AGGREGATE_HALF(
            embed_and_filtered.combine(halves, by: 0).combine(channel.fromList(methods))
        )
        // size: stops a group from waiting on a half whose task failed; that
        // replicate (and so that method's blocklist) is then simply missing.
        half_pairs = half_aggregates
            .groupTuple(by: [0, 1, 2], size: 2)
            .map { stem, rep, method, half_ids, files ->
                def by_half = [half_ids, files].transpose().sort { h -> h[0] }
                tuple(stem, rep, method, by_half[0][1], by_half[1][1])
            }
        correlations = CORRELATE_FEATURES(half_pairs)
        method_blocklists = BLOCKLIST(correlations.groupTuple(by: [0, 1], size: reps))
        blocklists = COMBINE_BLOCKLISTS(method_blocklists.groupTuple(size: methods.size()))

        passthrough = passthrough_methods
            // rep = half = 0 and no split file: every QC-passed cell.
            ? AGGREGATE_PASSTHROUGH(
                embed_and_filtered
                    .combine(channel.fromList(passthrough_methods))
                    .map { stem, emb, keys, norm, method -> tuple(stem, emb, keys, norm, 0, 0, [], method) }
            )
                .map { stem, _rep, _method, _half, file -> tuple(stem, file) }
                .groupTuple(size: passthrough_methods.size())
            : aggregates.map { stem, _agg -> tuple(stem, []) }
        FILTER_AGGREGATE(aggregates.join(blocklists).join(passthrough))

        // ── CellProfiler-feature track (experiments with cp_features: true) ─
        // Reuses the SAME QC_FILTER output -- no second QC pass -- and gets
        // no reproducibility filtering: its columns are hand-engineered and
        // meant to stay comparable to the published CellProfiler analysis.
        cp_features = BUILD_CP_FEATURES(
            plans.filter { p -> p.cp_features }
                .map { p -> tuple(p.batch_stem, p.cell_table_args) }
                .join(cell_tables)
        )
        cp_filtered = FILTER_CP_FEATURES(cp_features.join(qc_passed))
        cp_and_filtered = cp_features.join(cp_filtered)
        AGGREGATE_CP_FEATURES(cp_and_filtered)
        OVWT_BATCHWISE_CP_FEATURES(cp_and_filtered)
    }
}
