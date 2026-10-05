"""BLOCKLIST: the median-across-replicates reproducibility verdict for one aggregation method.

A column is reproducible (``feature_ok``) when the median of its per-replicate correlations is
at least ``minimum_correlation``. A null median (the correlation was undefined in every
replicate) counts as not reproducible. The output is sorted by ``feature``.
"""

import dataclasses
import glob
import logging
import pathlib

import polars as pl
from omegaconf import MISSING

from .config import AppConfig


def compute_blocklist(
    corr_df: pl.DataFrame, minimum_correlation: float
) -> pl.DataFrame:
    """
    Median-across-replicates verdict for every dimension.

    Parameters
    ----------
    corr_df : pl.DataFrame
        Concatenated CORRELATE_FEATURES output: ``feature``, ``r``, and
        (unused here) ``r_squared``, one row per (dimension, replicate).
    minimum_correlation : float
        Minimum median *r* a dimension needs to pass.

    Returns
    -------
    pl.DataFrame
        ``feature``, ``median_r``, ``feature_ok``, sorted by ``feature``
        so the file is byte-reproducible across runs.
    """
    return (
        corr_df.group_by("feature")
        .agg(pl.col("r").median().alias("median_r"))
        .with_columns(
            (pl.col("median_r") >= minimum_correlation)
            .fill_null(False)
            .alias("feature_ok")
        )
        .sort("feature")
    )


@dataclasses.dataclass
class BlocklistParams(AppConfig):
    """
    Hydra structured configuration for BLOCKLIST.

    Attributes
    ----------
    correlation_files : str
        Glob pattern matching every bootstrap replicate's
        CORRELATE_FEATURES output for ONE aggregation method. Required.
    minimum_correlation : float
        Minimum median Pearson *r* across replicates for a dimension to
        pass. Defaults to ``0.5``.
    output_name : str
        Basename (without ``.parquet``) of the single output file. The Nextflow module
        sets it to the aggregation method, so one experiment's per-method
        blocklists can share a directory. Defaults to ``"blocklist"``.
    """

    correlation_files: str = MISSING
    minimum_correlation: float = 0.5
    output_name: str = "blocklist"


def run_blocklist(cfg: BlocklistParams) -> None:
    """Run the stage with ``cfg`` (``output_dir`` must exist)."""
    output_dir = pathlib.Path(cfg.output_dir)

    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""

    paths = sorted(glob.glob(cfg.correlation_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {cfg.correlation_files!r}")
    logging.info("Found %d bootstrap replicate correlation file(s)", len(paths))
    corr_df = pl.concat([pl.read_parquet(p) for p in paths])

    blocklist_df = compute_blocklist(corr_df, cfg.minimum_correlation)
    n_ok = int(blocklist_df["feature_ok"].sum())
    logging.info(
        "%d/%d dimension(s) reproducible at median r >= %s",
        n_ok,
        blocklist_df.height,
        cfg.minimum_correlation,
    )

    out_path = output_dir / f"{prefix}{cfg.output_name}.parquet"
    logging.info("Writing %s", out_path)
    blocklist_df.write_parquet(out_path)

    logging.info("Done")
