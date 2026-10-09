"""Every file BUILD_CELL_IMAGES asks the nested snakemake for: each tile's
tables, and each well's WebDataset shards.

Imported by ``Snakefile`` (next to this file), so it runs in the ops env's
Python 3.10 snakemake interpreter: standard library only, and nothing from
``fisseq_embeddings_pipeline``. Unit tests load it by path.

Tiles are named the way starcall-workflow's own grid input functions name
them (``get_grid_filenames_seq``/``get_grid_filenames_pheno``):
``{well}_grid{grid_size}/tile{x:02}x{y:02}y``, every tile of the grid,
whether or not it exists yet -- the nested snakemake is what creates them.

A well's shards are one snakemake ``directory()`` output (``make_well_shards``)
next to its tiles, ``{well}_grid{grid_size}/{segmentation_type}_{image}_shards_
{window}_{shard_size}``, holding ``well_{n}_shard_{k}.tar.gz`` (``fisseq_embeddings_pipeline.well_shards``). How many it
holds depends on the well's cell count, so the manifest rule lists them once
they exist (:func:`shard_rows`).
"""

import os

MANIFEST_FIELDNAMES = [
    "well",
    "tile",
    "segmentation_csv",
    "reads_csv",
    "cellprofiler_csv",
]

SHARDS_MANIFEST_FIELDNAMES = ["well", "shard_tar"]


def tile_names(grid_size):
    """Every tile directory name of a ``grid_size`` x ``grid_size`` grid,
    sorted."""
    return sorted(
        "tile{:02}x{:02}y".format(x, y)
        for x in range(grid_size)
        for y in range(grid_size)
    )


def tile_rows(
    phenotyping_dir,
    sequencing_dir,
    wells,
    grid_size,
    segmentation_type,
    sequencing_reads_params="",
    cp_features=False,
    cellprofiler_cycle="",
    cellprofiler_pipeline="",
):
    """One manifest row (see ``MANIFEST_FIELDNAMES``) per tile of every well.

    ``phenotyping_dir``/``sequencing_dir`` are joined with ``/`` and may
    carry starcall's trailing slash or not. ``cellprofiler_csv`` is ``""``
    unless ``cp_features``.
    """
    phenotyping_dir = phenotyping_dir.rstrip("/")
    sequencing_dir = sequencing_dir.rstrip("/")
    rows = []
    for well in wells:
        grid_dir = "{}_grid{}".format(well, grid_size)
        for tile in tile_names(grid_size):
            tile_dir = "{}/{}/{}".format(phenotyping_dir, grid_dir, tile)
            seq_tile_dir = "{}/{}/{}".format(sequencing_dir, grid_dir, tile)
            cp_csv = ""
            if cp_features:
                cp_csv = "{}/cellprofiler{}_{}.csv".format(
                    tile_dir, cellprofiler_cycle, cellprofiler_pipeline
                )
            rows.append(
                {
                    "well": well,
                    "tile": tile,
                    "segmentation_csv": "{}/{}.csv".format(tile_dir, segmentation_type),
                    "reads_csv": "{}/{}_reads{}.csv".format(
                        seq_tile_dir, segmentation_type, sequencing_reads_params
                    ),
                    "cellprofiler_csv": cp_csv,
                }
            )
    return rows


def shard_size_name(shard_size):
    """``shard_size`` as it appears in a well's shard directory name:
    ``"all"`` for ``None`` (one shard per well)."""
    return "all" if shard_size is None else str(int(shard_size))


def well_shard_dirs(
    phenotyping_dir, wells, grid_size, segmentation_type, image, window, shard_size
):
    """Each well's shard directory, as ``{"well", "shard_dir"}`` rows.

    ``image`` (``"raw"``/``"corrected"``), ``window`` and ``shard_size`` are
    all in the name, so changing one requests new shards rather than
    reusing stale ones.
    """
    phenotyping_dir = phenotyping_dir.rstrip("/")
    return [
        {
            "well": well,
            "shard_dir": "{}/{}_grid{}/{}_{}_shards_{}_{}".format(
                phenotyping_dir,
                well,
                grid_size,
                segmentation_type,
                image,
                window,
                shard_size_name(shard_size),
            ),
        }
        for well in wells
    ]


def tile_shard_tars(grid_dir, grid_size, segmentation_type, image, window):
    """Every tile's own shard under ``grid_dir`` (``.../{well}_grid{N}``), in
    tile order: ``make_cell_shard``'s outputs, which ``make_well_shards``
    packs into the well's shards."""
    return [
        "{}/{}/{}_{}_shard_{}.tar".format(
            grid_dir.rstrip("/"), tile, segmentation_type, image, window
        )
        for tile in tile_names(grid_size)
    ]


def shard_targets(shard_dirs):
    """Each well's shard directory, alone: the ``fisseq_shards`` pass."""
    return [row["shard_dir"] for row in shard_dirs]


def shard_rows(shard_dirs):
    """One ``{"well", "shard_tar"}`` row (see ``SHARDS_MANIFEST_FIELDNAMES``)
    per shard in each well's directory, in shard order."""
    return [
        {"well": row["well"], "shard_tar": os.path.join(row["shard_dir"], name)}
        for row in shard_dirs
        for name in sorted(os.listdir(row["shard_dir"]))
        if name.endswith(".tar.gz")
    ]


def row_targets(rows):
    """The tables ``rows`` name, for snakemake to build: each tile's
    segmentation and reads tables, and CellProfiler table when there is
    one."""
    targets = []
    for row in rows:
        targets.extend([row["segmentation_csv"], row["reads_csv"]])
        if row["cellprofiler_csv"]:
            targets.append(row["cellprofiler_csv"])
    return targets
