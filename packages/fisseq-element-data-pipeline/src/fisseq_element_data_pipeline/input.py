"""Load and merge raw Element cell tables from a YAML spec.

Hydra entry point (``python -m fisseq_element_data_pipeline.input``) / Nextflow process
``INPUT`` (runs once per ``experiments:`` entry of ``params.yaml``). Reads a hand-authored YAML
config (``config_path``, parsed separately from the Hydra CLI config) describing one or more
Element CellProfiler cell tables (CSV or Parquet), and merges them into a single
``input/``-ready cell-level Parquet file with the same identity columns as
fisseq-data-pipeline's INPUT, so every later stage runs unchanged.

Pipeline
--------
1. Load every file in ``input_paths`` (CSV or parquet, detected by extension), tagging each
   row with which file it came from and its row index within that file (``origin_file`` /
   ``origin_row_idx``, published as ``meta_origin_file`` / ``meta_origin_row_idx``).
   Concatenate everything into one lazy frame.
2. Build the identity columns from Element's (see "Identity columns"), prefix every metadata
   column with ``meta_``, and write the result to a single output Parquet file.

Identity columns
----------------
- ``upBarcode`` is the whole target name (``target_col``, ``MostAbundantTarget``):
  ``<variant>_<NNNN>`` for a single-variant target. Element target names had to be unique,
  ``|`` became ``_`` and ``*`` became ``.``, so a multi-variant target (``A502L_R644H``) has no
  barcode id but is still one target, and is kept as its own barcode. The variant prefix
  keeps barcode ids unique across variants: QC_FILTER counts cells per barcode alone.
- ``aaChanges`` is ``variant_col`` (``variant``), with ``wt_label`` relabelled ``WT``.
- ``editDistance`` is 0 for every cell: Element reports none, and QC_FILTER's edit-distance
  filter then passes every cell.

Cells with a null target or variant are dropped.

Metadata columns
----------------
``origin_file``, ``origin_row_idx``, the Element per-cell columns in
:data:`ELEMENT_METADATA_COLUMNS` and any other non-numeric column are prefixed with ``meta_``.
Every remaining column is a feature.

Config file
-----------
``config_path`` points to a YAML file with:

    input_paths: [/path/to/file1.parquet, /path/to/file2.csv]
    feature_allowlist_file: null      # optional, default null (no allowlist)
    feature_blocklist_file: null      # optional, default null (no blocklist)
    csv_schema_scan_rows: 100         # optional, default 100
    target_col: MostAbundantTarget    # optional, raw name of the target (barcode) column
    variant_col: variant              # optional, raw name of the variant column
    wt_label: WT                      # optional, this experiment's wildtype variant value

``feature_allowlist_file`` / ``feature_blocklist_file`` each point to a plain text file with
one fnmatch-style glob pattern per line (e.g. ``AreaShape_*``), matched against feature
column names. If an allowlist is given, only feature columns matching at least one of its
patterns are kept; if a blocklist is also given, matching columns are then dropped from what
remains (allowlist is applied first).

``csv_schema_scan_rows`` controls how many rows of each CSV ``input_paths`` source are scanned
to infer column dtypes (polars ``scan_csv``'s ``infer_schema_length``); ``null`` scans every
row. Has no effect on parquet sources.

Usage
-----
    uv run python -m fisseq_element_data_pipeline.input \\
        output_dir=./out \\
        config_path=/path/to/config.yaml
"""

import dataclasses
import fnmatch
import logging
import pathlib

import hydra
import polars as pl
import polars.selectors as cs
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from fisseq_common.utils.log import setup_logging
from fisseq_common.variant import WT, relabel_wildtype

from .config import AppConfig

logger = logging.getLogger(__name__)

# The output's identity columns, which QC_FILTER reads (conf/modules.config).
BARCODE_COL = "upBarcode"
AA_CHANGES_COL = "aaChanges"
EDIT_DISTANCE_COL = "editDistance"

# The raw Element identity columns.
TARGET_COL = "MostAbundantTarget"
VARIANT_COL = "variant"

KNOWN_METADATA_COLUMNS = {
    "origin_file",
    "origin_row_idx",
}

#: Element's per-cell bookkeeping columns: numeric, but not features.
ELEMENT_METADATA_COLUMNS = {
    "cell_id_global",
    "tile",
    "ImageNumber",
    "ObjectNumber",
    "origin_x",
    "origin_y",
    "MostAbundantTargetCount",
    "TotalCount",
    "Confidence",
}


@dataclasses.dataclass
class InputStageConfig(AppConfig):
    """
    Hydra structured configuration for the input-generation entry point.

    Attributes
    ----------
    config_path : str
        Path to the YAML config describing the input files (see the module docstring's
        "Config file" section). Parsed separately via ``yaml.safe_load``, since
        ``input_paths`` is a list of arbitrary length. Required.
    """

    config_path: str = MISSING


_cs = ConfigStore.instance()
_cs.store(name="input_main", node=InputStageConfig)


def load_and_tag(path: str, infer_schema_length: int | None = 100) -> pl.LazyFrame:
    """
    Scan a CSV or parquet file and tag each row with its file and row index.

    Parameters
    ----------
    path : str
        Path to the CSV or parquet file to scan.
    infer_schema_length : int or None
        Number of rows to sample when inferring CSV column dtypes (forwarded
        to :func:`polars.scan_csv`'s ``infer_schema_length``). ``None`` scans
        every row. Ignored for parquet files, which carry their own schema.
    """
    suffix = pathlib.Path(path).suffix.lower()
    if suffix == ".csv":
        lf = pl.scan_csv(path, infer_schema_length=infer_schema_length)
    elif suffix == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        raise ValueError(f"Unsupported input file extension '{suffix}' for {path}")

    # Workaround: unsigned integer columns have been observed to hide negative
    # values (silently reinterpreted), so force them to signed integers.
    lf = lf.with_columns(cs.unsigned_integer().cast(pl.Int64))

    return lf.with_row_index(name="origin_row_idx").with_columns(
        pl.lit(str(path)).alias("origin_file")
    )


def load_and_concat(
    paths: list[str], infer_schema_length: int | None = 100
) -> pl.LazyFrame:
    """
    Load and tag every input file, then concatenate them into one lazy frame.

    ``infer_schema_length`` is forwarded to each :func:`load_and_tag` call --
    see its docstring.
    """
    logger.info("Loading and tagging %d input file(s)", len(paths))
    for p in paths:
        logger.info("  - %s", p)
    lfs = [load_and_tag(p, infer_schema_length=infer_schema_length) for p in paths]
    # "vertical_relaxed" tolerates minor dtype mismatches across CSV/parquet
    # sources (e.g. int32 vs int64) by upcasting, rather than erroring.
    return pl.concat(lfs, how="vertical_relaxed")


def load_feature_patterns(path: str) -> list[str]:
    """Read one glob-style feature-column-name pattern per line from a text file."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def element_identity_columns(
    lf: pl.LazyFrame,
    target_col: str = TARGET_COL,
    variant_col: str = VARIANT_COL,
    wt_label: str = WT,
) -> pl.LazyFrame:
    """
    Add ``upBarcode`` (the whole target name), ``aaChanges`` (the variant, ``wt_label``
    relabelled ``WT``) and ``editDistance`` (0) as String/String/Int64, dropping cells with a
    null target or variant.
    """
    schema_names = lf.collect_schema().names()
    missing = [c for c in (target_col, variant_col) if c not in schema_names]
    if missing:
        raise ValueError(f"Identity column(s) {missing} not found in the input columns")

    return lf.filter(
        pl.col(target_col).is_not_null() & pl.col(variant_col).is_not_null()
    ).with_columns(
        pl.col(target_col).cast(pl.String).alias(BARCODE_COL),
        relabel_wildtype(pl.col(variant_col).cast(pl.String), wt_label).alias(
            AA_CHANGES_COL
        ),
        pl.lit(0, dtype=pl.Int64).alias(EDIT_DISTANCE_COL),
    )


def select_output_columns(
    lf: pl.LazyFrame,
    feature_allowlist: list[str] | None = None,
    feature_blocklist: list[str] | None = None,
    target_col: str = TARGET_COL,
    variant_col: str = VARIANT_COL,
    wt_label: str = WT,
) -> pl.LazyFrame:
    """
    Split columns into feature columns, the identity columns (see
    :func:`element_identity_columns`) and ``meta_``-prefixed metadata.

    Metadata is ``origin_file``/``origin_row_idx``, :data:`ELEMENT_METADATA_COLUMNS` and any
    other non-numeric column; everything else is a feature. The raw target and variant
    columns are replaced by ``upBarcode``/``aaChanges``.

    If `feature_allowlist` is given, only feature columns matching at least
    one of its fnmatch glob patterns are kept. If `feature_blocklist` is also
    given, feature columns matching any of its patterns are then dropped from
    what remains (allowlist is applied first).
    """
    identity_columns = {target_col, variant_col}
    schema = lf.collect_schema()
    metadata_cols = [
        c
        for c, dtype in schema.items()
        if c not in identity_columns
        and not c.startswith("meta_")
        and (
            c in KNOWN_METADATA_COLUMNS
            or c in ELEMENT_METADATA_COLUMNS
            or not dtype.is_numeric()
        )
    ]
    feature_cols = [
        c
        for c in schema.names()
        if c not in identity_columns
        and c not in metadata_cols
        and not c.startswith("meta_")
    ]
    if feature_allowlist is not None:
        feature_cols = [c for c in feature_cols if _matches_any(c, feature_allowlist)]
    if feature_blocklist is not None:
        feature_cols = [
            c for c in feature_cols if not _matches_any(c, feature_blocklist)
        ]

    lf = element_identity_columns(lf, target_col, variant_col, wt_label)
    return lf.select(
        cs.by_name(*feature_cols),
        cs.by_name(BARCODE_COL, EDIT_DISTANCE_COL, AA_CHANGES_COL),
        cs.by_name(*metadata_cols).name.prefix("meta_"),
    )


@hydra.main(version_base=None, config_path=None, config_name="input_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: load and merge raw Element cell tables from a YAML spec.

    Steps
    -----
    1. Read ``config_path`` (a separate, hand-authored YAML file — see the
       module docstring's "Config file" section).
    2. Load and concatenate all ``input_paths`` via :func:`load_and_concat`.
    3. Build the identity columns, select/rename output columns and write a single Parquet
       file.

    Output file
    -----------
    ``{prefix}output.parquet``, where ``prefix`` is ``{output_root}.`` when
    ``output_root`` is set, otherwise empty.
    """
    in_cfg: InputStageConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(in_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_cfg.output_dir = output_dir
    setup_logging(in_cfg, "input")

    logger.info("Reading config file %s", in_cfg.config_path)
    with open(in_cfg.config_path) as f:
        config = yaml.safe_load(f)

    input_paths = config["input_paths"]

    feature_allowlist = None
    if config.get("feature_allowlist_file"):
        logger.info(
            "Loading feature allowlist from %s", config["feature_allowlist_file"]
        )
        feature_allowlist = load_feature_patterns(config["feature_allowlist_file"])

    feature_blocklist = None
    if config.get("feature_blocklist_file"):
        logger.info(
            "Loading feature blocklist from %s", config["feature_blocklist_file"]
        )
        feature_blocklist = load_feature_patterns(config["feature_blocklist_file"])

    csv_schema_scan_rows = config.get("csv_schema_scan_rows", 100)
    data_lf = load_and_concat(input_paths, infer_schema_length=csv_schema_scan_rows)

    logger.info("Selecting output columns")
    combined_lf = select_output_columns(
        data_lf,
        feature_allowlist,
        feature_blocklist,
        target_col=config.get("target_col") or TARGET_COL,
        variant_col=config.get("variant_col") or VARIANT_COL,
        wt_label=config.get("wt_label") or WT,
    )

    prefix = f"{in_cfg.output_root}." if in_cfg.output_root is not None else ""
    output_path = output_dir / f"{prefix}output.parquet"
    logger.info("Writing data to %s", output_path)
    combined_lf.sink_parquet(output_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
