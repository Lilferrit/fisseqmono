// EMBED_CELLS, the pipeline's only GPU-bound stage. container
// "${params.container_image}" and a trailing random_seed=${params.random_seed}
// are the two additions every module picks up versus fisseq-data-pipeline's
// modules. The `channels=[...]` list interpolation below mirrors
// aggregate_embeddings/main.nf's `aggregators=[...]` precedent;
// `apply_mask` is a plain scalar (one shared mask per cell, so one flag
// covering every selected channel -- see EmbedCellsConfig).
//
// The shards themselves aren't staged: shards.parquet names each well's
// shards by their real paths under phenotyping_dir, where the nested
// snakemake's make_well_shards rule left them, and nextflow.config binds that
// directory in. Each sample's meta.json carries the cell's meta_* columns;
// batch_stem adds meta_batch.

include { threadEnv } from '../../../../fisseq-common/nextflow/modules/local/functions'

process EMBED_CELLS {
    errorStrategy 'ignore'
    label 'process_gpu'
    label 'process_high'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/embeddings/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(shards), val(phenotyping_dir)

    output:
    tuple val(batch_stem), path("embeddings.parquet"), emit: embeddings

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.embed \\
        output_dir=. \\
        shards_path=${shards} \\
        batch_stem=${batch_stem} \\
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
