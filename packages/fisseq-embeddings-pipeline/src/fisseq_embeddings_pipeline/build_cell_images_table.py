"""BUILD_CELL_IMAGES, phase 3: build-table.

Hydra entry point (`python -m
fisseq_embeddings_pipeline.build_cell_images_table`), backing the third of
BUILD_CELL_IMAGES' three phases (the `build_cell_images` rule), run
after the rule's own nested `snakemake` invocation (phase 2, the one
step that still needs the separate `ops` conda env -- see the root
`Dockerfile`) has materialized every tile's segmentation/reads/CellProfiler
CSVs and every well's WebDataset shards.

Reads `manifest` (written in phase 2 by `snakemake/Snakefile`'s
`fisseq_tiles_manifest` rule) and writes `output` (`cell_table.parquet`),
the experiment's cell table in the data pipeline's shape: one row per cell,
the `meta_*` columns of :func:`tile_cell_meta` (the cell's key, the three
QC fields and the variant class -- exactly what each shard sample's
`meta.json` carries) and, if `cp_features`, the tile's CellProfiler
columns under their own names. BUILD_CELL_METADATA/BUILD_CP_FEATURES read
only this table, never starcall-workflow's tree.

The genotype column names come from `snakemake_config` (the nested run's
`--configfile`), so the table and the shards are built with the same ones.

Reads CSVs via pandas (matching starcall-workflow's own
``to_csv()``/``read_csv(index_col=0)`` convention), but writes the final
table via polars, matching this repo's own parquet-writing convention
(AGENTS.md: polars for tabular data) for the artifact everything downstream
actually reads. These CSV reads are the ONLY pandas in the pipeline
(``tile_shard.py`` reuses :func:`read_segmentation_table` rather than
reading the same CSV its own way).

Until this stage's Docker image merged starcall-workflow's own `ops` conda
env into this repo's main image (see the root `Dockerfile`), this logic
lived in a standalone `modules/local/build_cell_images_glue.py` (since
deleted) that
deliberately avoided importing `fisseq_embeddings_pipeline`, because it ran
inside a wholly separate container. That constraint no longer applies --
this module runs like every other stage, via this repo's own installed
package.
"""

import csv
import dataclasses
import logging
import pathlib
from typing import Any, Dict, List, Optional

import hydra
import pandas as pd
import polars as pl
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from fisseq_common.schema import (
    META_BARCODE_COL,
    META_EDIT_DISTANCE_COL,
    META_VARIANT_CLASS,
)
from fisseq_common.utils.log import setup_logging
from fisseq_common.variant import variant_type_expr

from .config import AppConfig
from .utils.cell_table import (
    CELL_META_SCHEMA,
    META_AA_CHANGES_COL,
    META_CELL_INDEX_COL,
    META_TILE_COL,
    META_WELL_COL,
)


@dataclasses.dataclass
class BuildCellImagesTableConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CELL_IMAGES' build-table phase.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    manifest : str
        Tile manifest CSV, written by `snakemake/Snakefile`'s
        `fisseq_tiles_manifest` rule (relative to `output_dir`). Defaults to
        ``"tiles_manifest.csv"``.
    output : str
        Output parquet filename (relative to `output_dir`). Defaults to
        ``"cell_table.parquet"``.
    shards_manifest : str
        Shard manifest CSV, written by the same rule (relative to
        `output_dir`). Defaults to ``"shards_manifest.csv"``.
    shards_output : str
        Output parquet filename (relative to `output_dir`) for the shard
        table -- see :func:`build_shards_table`. Defaults to
        ``"shards.parquet"``.
    snakemake_config : str
        The nested snakemake's ``--configfile`` (relative to `output_dir`),
        written by build_cell_images_prepare; its ``fisseq_*_col`` keys
        name the reads tables' genotype columns. Defaults to
        ``"snakemake_config.yaml"``.
    """

    manifest: str = "tiles_manifest.csv"
    shards_manifest: str = "shards_manifest.csv"
    output: str = "cell_table.parquet"
    shards_output: str = "shards.parquet"
    snakemake_config: str = "snakemake_config.yaml"


@dataclasses.dataclass(frozen=True)
class GenotypeColumns:
    """Which columns of a tile's reads table are the barcode, amino-acid
    changes and edit distance -- starcall's aux tables name them per
    experiment. Each experiment's ``barcode_col_name``/
    ``aa_changes_col_name``/``edit_distance_col_name``."""

    barcode: str = "upBarcode"
    aa_changes: str = "aaChanges"
    edit_distance: str = "editDistance"

    @classmethod
    def from_snakemake_config(cls, config: Dict[str, Any]) -> "GenotypeColumns":
        """The ``fisseq_*_col`` keys build_cell_images_prepare writes."""
        return cls(
            barcode=config["fisseq_barcode_col"],
            aa_changes=config["fisseq_aa_changes_col"],
            edit_distance=config["fisseq_edit_distance_col"],
        )


def _read_indexed_csv(path: str) -> pd.DataFrame:
    """Read a CSV the way starcall-workflow's own rules write it: a plain
    ``DataFrame.to_csv()`` with the default (unnamed) index column, read
    back via ``index_col=0``."""
    return pd.read_csv(path, index_col=0)


def read_segmentation_table(segmentation_csv: str) -> pd.DataFrame:
    """One tile's segmentation-side cell table, with its two cell indices.

    ``tile_cell_index`` is the CSV's own row index (what becomes
    ``meta_cell_index``); ``crop_index`` is the 0-based on-disk row
    position, so the cell's label in ``{segmentation_type}_mask.tif`` is
    ``crop_index + 1`` -- starcall-workflow's own
    ``enumerate(cell_table.index)`` convention. ``crop_index`` is computed
    independently of ``tile_cell_index``'s values, even though
    starcall-workflow's ``drop_duplicate_cells`` rule happens to make them
    equal today (a fresh per-tile ``RangeIndex(1, 1+N)`` every tile).

    Shared with ``tile_shard.py``, which crops each cell by that label, so
    the two can't disagree on which row is which cell.
    """
    seg = _read_indexed_csv(segmentation_csv)
    seg.index.name = "tile_cell_index"
    seg = seg.reset_index()
    seg["crop_index"] = range(len(seg))
    return seg


def tile_cell_meta(
    segmentation_csv: str,
    reads_csv: str,
    well: str,
    tile: str,
    columns: GenotypeColumns,
) -> pl.DataFrame:
    """One tile's :data:`CELL_META_SCHEMA` rows, one per cell.

    The one place a cell's metadata is built: ``cell_table.parquet``'s
    leading columns (:func:`build_tile_table`) and every shard sample's
    ``meta.json`` (``tile_shard.write_tile_shard``) both come from here.

    Parameters
    ----------
    segmentation_csv : str
        The tile's ``{segmentation_type}.csv`` (phenotyping_dir). Its own
        row index is the cell's ``meta_cell_index`` (see
        :func:`read_segmentation_table`).
    reads_csv : str
        The tile's ``{segmentation_type}_reads{params}.csv``
        (sequencing_dir), joined on by index value: starcall's
        ``combine_cell_reads``/``merge_final_tables`` keep the segmentation
        table's index unchanged.
    well, tile : str
        The tile's identifiers.
    columns : GenotypeColumns
        Which reads-table columns are the barcode, amino-acid changes and
        edit distance; names vary per experiment.

    Returns
    -------
    pl.DataFrame
        :data:`CELL_META_SCHEMA`, in the segmentation CSV's on-disk row
        order -- load-bearing: row ``i`` is mask label ``i + 1``.
        ``meta_variant_class`` is ``fisseq_common.variant``'s class of
        ``meta_aa_changes`` (null where it is).

    Raises
    ------
    ValueError
        If the two tables' cell index sets differ, or the reads table
        lacks one of ``columns``.
    """
    seg = read_segmentation_table(segmentation_csv)

    reads = _read_indexed_csv(reads_csv)
    reads.index.name = "tile_cell_index"
    reads = reads.reset_index()

    seg_keys = set(seg["tile_cell_index"])
    reads_keys = set(reads["tile_cell_index"])
    if seg_keys != reads_keys:
        raise ValueError(
            f"{well}/{tile}: segmentation table {segmentation_csv!r} and "
            f"reads table {reads_csv!r} have different tile_cell_index "
            f"sets (segmentation-only: {sorted(seg_keys - reads_keys)}, "
            f"reads-only: {sorted(reads_keys - seg_keys)}) -- expected an "
            "exact match (the reads table keeps the segmentation table's "
            "own index)."
        )
    wanted = [columns.barcode, columns.aa_changes, columns.edit_distance]
    missing = [c for c in wanted if c not in reads.columns]
    if missing:
        raise ValueError(
            f"{well}/{tile}: reads table {reads_csv!r} has no column(s) "
            f"{missing}; it has {list(reads.columns)}. Set barcode_col_name/"
            "aa_changes_col_name/edit_distance_col_name for this experiment."
        )
    if len(seg) == 0:
        return pl.DataFrame(schema=CELL_META_SCHEMA)

    merged = seg[["tile_cell_index"]].merge(
        reads[["tile_cell_index", *wanted]], on="tile_cell_index", how="left"
    )
    return (
        pl.from_pandas(merged)
        .select(
            pl.lit(well).alias(META_WELL_COL),
            pl.lit(tile).alias(META_TILE_COL),
            pl.col("tile_cell_index").cast(pl.Int64).alias(META_CELL_INDEX_COL),
            pl.col(columns.barcode).cast(pl.String).alias(META_BARCODE_COL),
            pl.col(columns.aa_changes).cast(pl.String).alias(META_AA_CHANGES_COL),
            pl.col(columns.edit_distance).cast(pl.Int64).alias(META_EDIT_DISTANCE_COL),
        )
        .with_columns(variant_type_expr(META_AA_CHANGES_COL).alias(META_VARIANT_CLASS))
        .cast(CELL_META_SCHEMA)
    )


def build_tile_table(
    segmentation_csv: str,
    reads_csv: str,
    cellprofiler_csv: Optional[str],
    well: str,
    tile: str,
    columns: GenotypeColumns,
) -> pl.DataFrame:
    """One tile's rows of ``cell_table.parquet``.

    :func:`tile_cell_meta`'s columns, then -- if ``cellprofiler_csv`` is
    given -- every column of the tile's CellProfiler CSV under its own
    CellProfiler name, joined by row position (CellProfiler numbers its
    objects in mask-label order, which is the segmentation table's row
    order; it shares no index with it).

    Raises
    ------
    ValueError
        As :func:`tile_cell_meta`, or if the CellProfiler CSV's row count
        differs from the segmentation table's.
    """
    meta = tile_cell_meta(segmentation_csv, reads_csv, well, tile, columns)
    if not cellprofiler_csv:
        return meta
    cp = _read_indexed_csv(cellprofiler_csv)
    if len(cp.index) != meta.height:
        raise ValueError(
            f"{well}/{tile}: segmentation table {segmentation_csv!r} has "
            f"{meta.height} row(s) but CellProfiler output "
            f"{cellprofiler_csv!r} has {len(cp.index)} row(s) -- the "
            "row-position join this relies on requires equal row counts."
        )
    if meta.height == 0:
        return meta
    return meta.hstack(pl.from_pandas(cp.reset_index(drop=True)))


def build_cell_table(
    tiles: List[Dict[str, Any]], columns: GenotypeColumns
) -> pl.DataFrame:
    """Every tile's rows (see :func:`build_tile_table`) as one
    per-experiment ``cell_table.parquet``-shaped frame.

    Parameters
    ----------
    tiles : list[dict]
        Each dict: ``well``, ``tile``, ``segmentation_csv``, ``reads_csv``,
        ``cellprofiler_csv`` (empty string/``None`` if ``cp_features`` is
        off for this experiment).
    columns : GenotypeColumns
        The reads tables' genotype column names.

    Returns
    -------
    pl.DataFrame
        Concatenated ``how="diagonal_relaxed"`` across tiles (CellProfiler
        columns aren't fixed). Just :data:`CELL_META_SCHEMA` if ``tiles`` is
        empty.
    """
    frames = [
        build_tile_table(
            segmentation_csv=tile_info["segmentation_csv"],
            reads_csv=tile_info["reads_csv"],
            cellprofiler_csv=tile_info.get("cellprofiler_csv") or None,
            well=tile_info["well"],
            tile=tile_info["tile"],
            columns=columns,
        )
        for tile_info in tiles
    ]
    if not frames:
        return pl.DataFrame(schema=CELL_META_SCHEMA)
    return pl.concat(frames, how="diagonal_relaxed")


SHARDS_SCHEMA: Dict[str, pl.DataType] = {
    "well": pl.String,
    "shard_tar": pl.String,
}


def build_shards_table(shards: List[Dict[str, Any]]) -> pl.DataFrame:
    """One row per WebDataset shard, in order: where the nested snakemake's
    ``make_well_shards`` rule (``snakemake/Snakefile``) left it, under
    phenotyping_dir, and its well.

    EMBED_CELLS reads its shards from this list. It's a sidecar rather
    than a ``cell_table.parquet`` column so ``cell_table.parquet`` stays
    purely per-cell.
    """
    return pl.DataFrame(
        [{key: shard[key] for key in SHARDS_SCHEMA} for shard in shards],
        schema=SHARDS_SCHEMA,
    )


def _read_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


_cs = ConfigStore.instance()
_cs.store(name="build_cell_images_table_main", node=BuildCellImagesTableConfig)


@hydra.main(
    version_base=None, config_path=None, config_name="build_cell_images_table_main"
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: join every tile's tables into one cell_table.parquet.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.build_cell_images_table \\
            output_dir=./out \\
            manifest=tiles_manifest.csv \\
            shards_manifest=shards_manifest.csv \\
            output=cell_table.parquet \\
            shards_output=shards.parquet \\
            snakemake_config=snakemake_config.yaml
    """
    table_cfg: BuildCellImagesTableConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(table_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_cfg.output_dir = str(output_dir)
    setup_logging(table_cfg, "build_cell_images_table")

    manifest_path = output_dir / table_cfg.manifest
    tiles = _read_manifest(str(manifest_path))
    with open(output_dir / table_cfg.snakemake_config) as f:
        columns = GenotypeColumns.from_snakemake_config(yaml.safe_load(f))
    table = build_cell_table(tiles, columns)

    output_path = output_dir / table_cfg.output
    table.write_parquet(output_path)
    shards = _read_manifest(str(output_dir / table_cfg.shards_manifest))
    build_shards_table(shards).write_parquet(output_dir / table_cfg.shards_output)

    logging.info(
        "Wrote %s (%d cell(s) across %d tile(s), in %d shard(s))",
        output_path,
        table.height,
        len(tiles),
        len(shards),
    )


if __name__ == "__main__":
    main()
