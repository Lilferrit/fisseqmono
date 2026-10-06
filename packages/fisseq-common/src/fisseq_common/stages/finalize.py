"""FINALIZE_FEATURE_SELECT: one experiment's per-variant table after feature selection.

Steps:

1. Join every selected method's aggregates (AGGREGATE_FEATURE_TYPE's
   ``aggregates/<method>.parquet``) on the variant label (:func:`join_feature_type_files`).
2. Drop the columns the combined blocklist marks as not reproducible (:func:`apply_blocklist`; a
   null verdict counts as blocked, columns the blocklist doesn't mention are kept). This
   reproducibility verdict is the only feature selection.
3. Mark the untagged synonymous variants ``meta_is_control``
   (:func:`~fisseq_common.stages.filter.variant_classification`; the column is kept in the
   output) and z-score every feature against them. In the pipelines the
   aggregates are already z-scored by AGGREGATE_FEATURE_TYPE, so this is effectively a no-op;
   it keeps the stage correct on raw aggregates. Needs at least two synonymous variants.
4. Optionally add the impact score: the cosine distance from the synonymous variants' median
   (:func:`fisseq_common.utils.vectors.compute_impact_score`).
5. Join the per-variant metadata counts, from the filter stage's ``filtered_keys.parquet``
   (:func:`fisseq_common.utils.metadata.get_aggregate_meta_data`).
6. Join the passthrough aggregates (methods aggregated but never blocklisted or normalized,
   e.g. p-values; :func:`join_passthrough`) -- last, so steps 3 and 4, which pick their inputs
   by ``FEATURE_SELECTOR``, never see them.

Writes ``output.parquet``, one row per variant, sorted by label.

Entry point: ``python -m fisseq_common.stages.finalize``.
"""

import dataclasses
import glob
import logging
import pathlib
from typing import Optional

import polars as pl
from omegaconf import MISSING

from fisseq_common.normalizer import Normalizer
from fisseq_common.utils.metadata import get_aggregate_meta_data
from fisseq_common.utils.vectors import compute_impact_score

from ..global_aggregation import blocked_features
from .config import AppConfig, stage_main
from .filter import variant_classification


def join_feature_type_files(paths: list[str], label_column: str) -> pl.DataFrame:
    """
    Join per-method aggregate files into one table on ``label_column``, in the order given.

    Each file holds ``[label_column] + <method's stat columns>``, one row per variant.
    """
    agg_df = pl.read_parquet(paths[0])
    for p in paths[1:]:
        agg_df = agg_df.join(pl.read_parquet(p), on=label_column)
    return agg_df


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
class FinalizeConfig(AppConfig):
    """
    FINALIZE_FEATURE_SELECT's configuration.

    Attributes
    ----------
    feature_type_files : str
        Glob matching the selected methods' aggregates. Required; a glob matching nothing is an
        error.
    block_list_file : str
        COMBINE_BLOCKLISTS' ``blocklist.parquet`` (``feature``, ``feature_ok``). Required.
    filtered_keys_file : str
        The filter stage's ``filtered_keys.parquet``: the per-variant metadata counts are
        computed from it. Required.
    passthrough_feature_type_files : str, optional
        Glob matching the passthrough aggregates. ``None`` or a glob matching nothing (an
        empty passthrough list) joins none. Defaults to ``None``.
    label_column : str
        Variant label column. Defaults to ``"meta_aa_changes"``.
    compute_impact_score : bool
        Add ``meta_impact_score``. Defaults to ``True``.
    output_name : str
        The output is ``{output_name}.parquet`` (``{output_root}.``-prefixed when that is set).
        Defaults to ``"output"``.
    """

    feature_type_files: str = MISSING
    block_list_file: str = MISSING
    filtered_keys_file: str = MISSING
    passthrough_feature_type_files: Optional[str] = None
    label_column: str = "meta_aa_changes"
    compute_impact_score: bool = True
    output_name: str = "output"


def finalize(cfg: FinalizeConfig) -> pl.DataFrame:
    """The per-variant table :class:`FinalizeConfig` describes (see the module docstring)."""
    logging.info("Loading per-method aggregates from %s", cfg.feature_type_files)
    paths = sorted(glob.glob(cfg.feature_type_files))
    if not paths:
        raise ValueError(f"No files matched glob pattern: {cfg.feature_type_files!r}")
    agg_df = join_feature_type_files(paths, cfg.label_column)

    logging.info("Applying the blocklist from %s", cfg.block_list_file)
    agg_df = apply_blocklist(agg_df, pl.read_parquet(cfg.block_list_file))

    logging.info("Z-scoring against the synonymous variants")
    selected_lf = variant_classification(agg_df.lazy(), cfg.label_column)
    normalizer = Normalizer.from_lazyframe(selected_lf, fit_only_on_control=True)
    selected_lf = normalizer.apply(selected_lf)
    if cfg.compute_impact_score:
        logging.info("Computing impact scores")
        selected_lf = compute_impact_score(selected_lf)

    logging.info("Joining per-variant metadata from %s", cfg.filtered_keys_file)
    meta_lf = get_aggregate_meta_data(
        pl.scan_parquet(cfg.filtered_keys_file), cfg.label_column
    )
    selected_df = selected_lf.join(meta_lf, on=cfg.label_column).collect()

    # Deliberately last: see the module docstring.
    selected_df = join_passthrough(
        selected_df, cfg.passthrough_feature_type_files, cfg.label_column
    )
    return selected_df.sort(cfg.label_column)


def run_finalize(cfg: FinalizeConfig) -> None:
    """Write :func:`finalize`'s table to ``{prefix}{output_name}.parquet``."""
    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""
    out_path = pathlib.Path(cfg.output_dir) / f"{prefix}{cfg.output_name}.parquet"
    selected_df = finalize(cfg)
    logging.info("Writing %s (%d column(s))", out_path, selected_df.width)
    selected_df.write_parquet(out_path)
    logging.info("Done")


main = stage_main("finalize_main", FinalizeConfig, run_finalize)

if __name__ == "__main__":
    main()
