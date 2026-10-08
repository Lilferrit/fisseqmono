"""Every tile file BUILD_CELL_IMAGES asks the nested snakemake for.

Imported by ``Snakefile`` (next to this file), so it runs in the ops env's
Python 3.10 snakemake interpreter: standard library only, and nothing from
``fisseq_embeddings_pipeline``. Unit tests load it by path.

Tiles are named the way starcall-workflow's own grid input functions name
them (``get_grid_filenames_seq``/``get_grid_filenames_pheno``):
``{well}_grid{grid_size}/tile{x:02}x{y:02}y``, every tile of the grid,
whether or not it exists yet -- the nested snakemake is what creates them.
"""

MANIFEST_FIELDNAMES = [
    "well",
    "tile",
    "segmentation_csv",
    "reads_csv",
    "cellprofiler_csv",
    "shard_tar",
]


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
    image,
    window,
    sequencing_reads_params="",
    cp_features=False,
    cellprofiler_cycle="",
    cellprofiler_pipeline="",
):
    """One manifest row (see ``MANIFEST_FIELDNAMES``) per tile of every well.

    ``phenotyping_dir``/``sequencing_dir`` are joined with ``/`` and may
    carry starcall's trailing slash or not. ``image`` is ``"raw"`` or
    ``"corrected"``. ``cellprofiler_csv`` is ``""`` unless ``cp_features``.
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
                    "shard_tar": "{}/{}_{}_shard_{}.tar".format(
                        tile_dir, segmentation_type, image, window
                    ),
                }
            )
    return rows


def row_targets(rows):
    """The files ``rows`` name, for snakemake to build: each tile's shard,
    segmentation and reads tables, and CellProfiler table when there is one.
    Never the whole-tile image or mask: those are ``temp()`` upstream, so
    snakemake deletes them once the shard is cut (docs/architecture.md
    decision 17)."""
    targets = []
    for row in rows:
        targets.extend([row["shard_tar"], row["segmentation_csv"], row["reads_csv"]])
        if row["cellprofiler_csv"]:
            targets.append(row["cellprofiler_csv"])
    return targets
