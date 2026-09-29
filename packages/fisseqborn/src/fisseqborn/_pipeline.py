"""Locating files in a fisseq-data-pipeline output directory.

See the pipeline's ``docs/architecture.md`` ("Output layout")::

    <pipeline_dir>/
      feature_select_batchwise/<batch>/{aggregates,passthrough_aggregates,blocklists}/<type>.parquet
      ovwt_batchwise/<batch>/results.parquet
      global/<channel>/{feature_select/aggregate.parquet,ovwt_distinguishability/global_scores.parquet}
"""

import pathlib
from collections.abc import Sequence
from os import PathLike

import polars as pl

from . import _data

FEATURE_SELECT = "feature_select_batchwise"
OVWT = "ovwt_batchwise"


def batch_dirs(
    pipeline_dir: str | PathLike, stage: str, batches: Sequence[str] | None = None
) -> list[pathlib.Path]:
    """The ``<pipeline_dir>/<stage>/<batch>`` directories, in natural order (T2 < T10).

    ``batches`` selects (and orders) a subset; a missing one raises `FileNotFoundError`.
    """
    root = pathlib.Path(pipeline_dir) / stage
    if not root.is_dir():
        raise FileNotFoundError(f"No {stage!r} directory in {pipeline_dir}: {root} does not exist")
    if batches is None:
        found = sorted(
            (d for d in root.iterdir() if d.is_dir()), key=lambda d: _data._natural_key(d.name)
        )
        if not found:
            raise FileNotFoundError(f"No batch directories in {root}")
        return found
    dirs = [root / b for b in batches]
    missing = [d.name for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"Batch(es) {missing} not found in {root}")
    return dirs


def scan(path: pathlib.Path) -> pl.LazyFrame:
    """`polars.scan_parquet`, failing early with the full path if the file is missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline output not found: {path}")
    return pl.scan_parquet(path)


def tag(lf: pl.LazyFrame, batch: str, batch_col: str) -> pl.LazyFrame:
    return lf.with_columns(pl.lit(batch, dtype=pl.String).alias(batch_col))
