"""COMBINE_BLOCKLISTS: one experiment's per-method blocklists in a single table, sorted by
``feature``."""

import dataclasses
import glob
import logging
import pathlib

import polars as pl
from omegaconf import MISSING

from .config import AppConfig


@dataclasses.dataclass
class CombineBlocklistsParams(AppConfig):
    """
    Hydra structured configuration for COMBINE_BLOCKLISTS.

    Attributes
    ----------
    blocklist_files : str
        Glob pattern matching every per-method BLOCKLIST output for one
        experiment. Required.
    """

    blocklist_files: str = MISSING


def run_combine_blocklists(cfg: CombineBlocklistsParams) -> None:
    """Run the stage with ``cfg`` (``output_dir`` must exist)."""
    output_dir = pathlib.Path(cfg.output_dir)

    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""

    paths = sorted(glob.glob(cfg.blocklist_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {cfg.blocklist_files!r}")
    logging.info("Found %d per-method blocklist file(s)", len(paths))
    combined = pl.concat([pl.read_parquet(p) for p in paths]).sort("feature")

    out_path = output_dir / f"{prefix}blocklist.parquet"
    logging.info(
        "Writing %s (%d/%d dimension(s) reproducible)",
        out_path,
        int(combined["feature_ok"].sum()),
        combined.height,
    )
    combined.write_parquet(out_path)

    logging.info("Done")
