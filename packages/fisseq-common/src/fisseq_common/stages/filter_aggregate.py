"""Applying a combined blocklist to an aggregate table and attaching passthrough aggregates.

:func:`apply_blocklist` drops the columns the blocklist marks as not reproducible (a null
verdict counts as blocked; columns the blocklist doesn't mention are kept).
:func:`join_passthrough` left-joins passthrough aggregates (methods aggregated but never
blocklisted, e.g. p-values) onto the result. This is the embeddings pipeline's FILTER_AGGREGATE
stage (:func:`run_filter_aggregate`), and the blocklist/passthrough steps of the data pipeline's
FINALIZE_FEATURE_SELECT.
"""

import dataclasses
import glob
import logging
import pathlib
from typing import Optional

import polars as pl
from omegaconf import MISSING

from ..global_aggregation import blocked_features
from .config import AppConfig


def apply_blocklist(agg_df: pl.DataFrame, blocklist_df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop every blocked feature column from an aggregate table.

    Columns named in the blocklist but absent from ``agg_df`` are ignored
    (a method may have been removed from ``aggregate_methods`` since the
    blocklist was built); columns present but *unmentioned* are kept, so
    a partial blocklist can never silently delete data.
    """
    blocked = blocked_features(blocklist_df)
    to_drop = [c for c in agg_df.columns if c in blocked]
    logging.info(
        "Dropping %d non-reproducible column(s) of %d blocklisted; %d column(s) remain",
        len(to_drop),
        len(blocked),
        agg_df.width - len(to_drop),
    )
    return agg_df.drop(to_drop)


def join_passthrough(
    filtered_df: pl.DataFrame, pattern: Optional[str], label_column: str
) -> pl.DataFrame:
    """
    Left-join every passthrough aggregate onto the filtered table.

    Parameters
    ----------
    filtered_df : pl.DataFrame
        :func:`apply_blocklist`'s output.
    pattern : str or None
        Glob matching the AGGREGATE_PASSTHROUGH outputs. ``None``, or a
        pattern matching nothing, returns ``filtered_df`` unchanged --
        an empty ``aggregate_methods_passthrough`` is the default, so
        this is a warning, not the ``ValueError`` an empty blocklist glob
        would raise.
    label_column : str
        Join key.

    Returns
    -------
    pl.DataFrame
        A LEFT join, so a variant missing from a passthrough aggregate
        surfaces as nulls rather than vanishing from the output.

    Raises
    ------
    ValueError
        If a passthrough file carries a non-label column that already
        exists in ``filtered_df`` -- that would mean a method is in both
        ``aggregate_methods`` and ``aggregate_methods_passthrough``, which
        ``validate_config`` rejects up front, so reaching this is a bug.
    """
    if not pattern:
        return filtered_df

    paths = sorted(glob.glob(pattern))
    if not paths:
        logging.warning(
            "No files matched passthrough glob pattern %r; no passthrough "
            "columns will be joined",
            pattern,
        )
        return filtered_df

    logging.info("Joining passthrough aggregates from %d file(s)", len(paths))
    result = filtered_df
    for path in paths:
        pt_df = pl.read_parquet(path)
        collisions = (set(pt_df.columns) & set(result.columns)) - {label_column}
        if collisions:
            raise ValueError(
                f"Passthrough aggregate {path!r} collides with existing "
                f"column(s) {sorted(collisions)}. A method is either selected "
                "on or passed through, never both."
            )
        result = result.join(pt_df, on=label_column, how="left")
    return result


@dataclasses.dataclass
class FilterAggregateParams(AppConfig):
    """
    Hydra structured configuration for FILTER_AGGREGATE.

    Attributes
    ----------
    aggregate_file : str
        Path to AGGREGATE_EMBEDDINGS' aggregate.parquet. Required.
    blocklist_file : str
        Path to COMBINE_BLOCKLISTS' blocklist.parquet. Required.
    passthrough_files : str or None
        Glob matching this experiment's AGGREGATE_PASSTHROUGH outputs.
        ``None`` (the default) or an empty match means no passthrough
        columns -- the default, since ``aggregate_methods_passthrough``
        defaults to ``[]``.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    """

    aggregate_file: str = MISSING
    blocklist_file: str = MISSING
    passthrough_files: Optional[str] = None
    label_column: str = "meta_aa_changes"


def run_filter_aggregate(cfg: FilterAggregateParams) -> None:
    """Run the stage with ``cfg`` (``output_dir`` must exist)."""
    output_dir = pathlib.Path(cfg.output_dir)

    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""

    logging.info("Reading aggregate from %s", cfg.aggregate_file)
    agg_df = pl.read_parquet(cfg.aggregate_file)
    logging.info("Reading blocklist from %s", cfg.blocklist_file)
    blocklist_df = pl.read_parquet(cfg.blocklist_file)

    filtered_df = apply_blocklist(agg_df, blocklist_df)
    filtered_path = output_dir / f"{prefix}filtered_aggregate.parquet"
    logging.info("Writing %s", filtered_path)
    filtered_df.write_parquet(filtered_path)

    with_pt_df = join_passthrough(filtered_df, cfg.passthrough_files, cfg.label_column)
    with_pt_path = output_dir / f"{prefix}aggregate_with_passthrough.parquet"
    logging.info("Writing %s (%d column(s))", with_pt_path, with_pt_df.width)
    with_pt_df.write_parquet(with_pt_path)

    logging.info("Done")
