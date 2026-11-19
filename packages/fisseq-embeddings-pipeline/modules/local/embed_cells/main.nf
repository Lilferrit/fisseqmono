// EMBED_CELLS, the pipeline's only GPU-bound stage. container
// "${params.container_image}" and a trailing random_seed=${params.random_seed}
// are the two additions every module picks up versus fisseq-data-pipeline's
// modules. The `channels=[...]` list interpolation below mirrors
// aggregate_embeddings/main.nf's `aggregators=[...]` precedent;
// `apply_mask` is a plain scalar (one shared mask per cell, so one flag
// covering every selected channel -- see EmbedCellsConfig).
//
// One task per well: a task killed partway (walltime, preemption) loses one
// well, and -resume reruns just that one. The well's shards are staged under
// shards/ and EMBED_CELLS reads every one there. Each sample's meta.json
// carries the cell's meta_* columns; batch_stem adds meta_batch. The per-well
// table isn't published: CONCAT_EMBEDDINGS stacks an experiment's wells into
// its embeddings.parquet. n_wells only rides along, for that grouping.

include { threadEnv } from '../../../../fisseq-common/nextflow/modules/local/functions'

process EMBED_CELLS {
    errorStrategy 'ignore'
    label 'process_gpu'
    label 'process_high'
    container "${params.container_image}"
    tag "${batch_stem} ${well}"

    input:
    tuple val(batch_stem), val(n_wells), val(well), path(shard_files, stageAs: 'shards/*')

    output:
    tuple val(batch_stem), val(n_wells), val(well), path("embeddings_${well}.parquet"), emit: embeddings

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.embed \\
        output_dir=. \\
        shards_dir=shards \\
        batch_stem=${batch_stem} \\
        output=embeddings_${well}.parquet \\
        checkpoint_path=${params.cell_dino_checkpoint} \\
        arch=${params.cell_dino_arch} \\
        patch_size=${params.cell_dino_patch_size} \\
        crop_size=${params.cell_dino_crop_size} \\
        'channels=[${params.cell_dino_channels.join(",")}]' \\
        apply_mask=${params.cell_dino_apply_mask} \\
        channel_pool=${params.cell_dino_channel_pool} \\
        device=${params.cell_dino_device} \\
        batch_size=${params.cell_dino_batch_size} \\
        num_workers=${params.cell_dino_num_workers} \\
        random_seed=${params.random_seed}
    """
}
