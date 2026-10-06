"""Cross-experiment aggregation of the pipelines' per-experiment outputs.

Polars only (base install). fisseqborn's ``write_global`` (the ``fisseqborn-global`` command)
runs these on a pipeline run; they are the methods of the embeddings pipeline's former global
stages (GLOBAL_BLOCKLIST, GLOBAL_VARIANT_EMBEDDINGS, GLOBAL_VARIANT_DISTINGUISHABILITY and
their CellProfiler-track twins):

- :func:`blocklist_vote`: a feature is reproducible when every experiment that reports it
  marks it OK, or at least ``min_batches_ok`` do;
- :func:`drop_blocked`: drop the features the vote marks not OK from every experiment's
  aggregates (features the vote doesn't mention are kept);
- :func:`median_across_batches`: per variant, the median of each feature across experiments,
  over the features every experiment has;
- :func:`ovwt_global_scores`: z-score each experiment's OvWT AUROCs against its own
  synonymous variants, then take the median across experiments;
- :func:`n_components_for_variance` and :func:`reduced_pca`: the leading principal components
  that reach a cumulative variance explained, with an impact score computed on them.

The PCA fit itself needs scikit-learn and lives with its caller.
"""

import logging
from collections.abc import Sequence
from typing import Optional

import polars as pl

from .normalizer import Normalizer
from .schema import (
    CONTROL_COLUMN_NAME,
    CUMULATIVE_VARIANCE_EXPLAINED_COL,
    FEATURE_SELECTOR,
    PC_COL_PREFIX,
)
from .utils.vectors import compute_impact_score
from .variant import control_expr, variant_type_expr

#: The scalar OvWT score columns pooled by :func:`ovwt_global_scores`.
OVWT_SCORES: tuple[str, ...] = (
    "auroc_pooled",
    "auroc_median_barcode",
    "auroc_median_fold",
)


def mark_controls(lf: pl.LazyFrame, label_column: str) -> pl.LazyFrame:
    """Add ``meta_is_control``: untagged synonymous variants (the embeddings pipeline's
    control rule, :func:`fisseq_common.variant.control_expr`)."""
    return lf.with_columns(
        control_expr(variant_type_expr(label_column), label_column)
        .fill_null(False)
        .alias(CONTROL_COLUMN_NAME)
    )


# --- blocklists -----------------------------------------------------------------------------


def blocked_features(blocklist_df: pl.DataFrame) -> set[str]:
    """
    The column names a blocklist marks as not reproducible.

    Parameters
    ----------
    blocklist_df : pl.DataFrame
        A blocklist: ``feature`` and ``feature_ok`` (COMBINE_BLOCKLISTS' output, or
        :func:`blocklist_vote`'s).

    Returns
    -------
    set of str
        Every ``feature`` whose ``feature_ok`` is false. A null ``feature_ok`` counts as
        blocked.
    """
    return set(
        blocklist_df.filter(~pl.col("feature_ok").fill_null(False))["feature"].to_list()
    )


def blocklist_vote(
    blocklist_dfs: Sequence[pl.DataFrame], min_batches_ok: Optional[int] = None
) -> pl.DataFrame:
    """
    Vote each feature reproducible or not across per-experiment blocklists.

    Parameters
    ----------
    blocklist_dfs : sequence of pl.DataFrame
        One blocklist per experiment (``feature``, ``feature_ok``; COMBINE_BLOCKLISTS'
        ``blocklist.parquet``).
    min_batches_ok : int or None
        Minimum number of experiments that must mark a feature OK. ``None`` (the default)
        requires every experiment that reports it: a feature absent from one experiment's
        blocklist is judged on the ones that name it.

    Returns
    -------
    pl.DataFrame
        ``feature``, ``n_batches``, ``n_ok``, ``feature_ok``, sorted by ``feature``.
    """
    if not blocklist_dfs:
        raise ValueError("blocklist_dfs must be non-empty")
    combined = pl.concat([df.select("feature", "feature_ok") for df in blocklist_dfs])
    result = combined.group_by("feature").agg(
        pl.col("feature").count().alias("n_batches"),
        pl.col("feature_ok").sum().alias("n_ok"),
    )
    if min_batches_ok is None:
        ok_expr = pl.col("n_ok") == pl.col("n_batches")
    else:
        ok_expr = pl.col("n_ok") >= min_batches_ok
    return result.with_columns(ok_expr.alias("feature_ok")).sort("feature")


def drop_blocked(
    batch_lfs: Sequence[pl.LazyFrame], blocklist_df: pl.DataFrame
) -> list[pl.LazyFrame]:
    """Drop the columns ``blocklist_df`` marks not OK (:func:`blocked_features`) from every
    experiment's frame. Columns the blocklist doesn't mention are kept."""
    blocked = blocked_features(blocklist_df)
    logging.info("Applying the blocklist: %d non-reproducible feature(s)", len(blocked))
    return [
        lf.drop([c for c in lf.collect_schema().names() if c in blocked])
        for lf in batch_lfs
    ]


# --- pooling --------------------------------------------------------------------------------


def median_across_batches(
    batch_lfs: Sequence[pl.LazyFrame],
    label_column: str,
    batch_labels: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """
    Per variant, the median of each feature across experiments.

    Each experiment's frame is reduced to ``label_column`` plus the feature columns
    (``FEATURE_SELECTOR``: not ``meta_*``) that every experiment has; a feature missing from
    any experiment is dropped, with a warning naming it. Every other ``meta_*`` column is
    dropped. A variant in only one experiment keeps that experiment's values.

    Parameters
    ----------
    batch_lfs : sequence of pl.LazyFrame
        Each experiment's per-variant aggregates. Must be non-empty.
    label_column : str
        The variant label column, the group key.
    batch_labels : sequence of str, optional
        Experiment names, used only in the dropped-column warning (default: positions).

    Returns
    -------
    pl.DataFrame
        One row per variant: ``label_column`` and every common feature column.

    Raises
    ------
    ValueError
        If ``batch_lfs`` is empty, or no feature column is common to every experiment.
    """
    if not batch_lfs:
        raise ValueError("batch_lfs must be non-empty")
    if batch_labels is None:
        batch_labels = [str(i) for i in range(len(batch_lfs))]

    per_batch_feature_cols = [
        set(lf.select(FEATURE_SELECTOR).collect_schema().names()) - {label_column}
        for lf in batch_lfs
    ]
    common_feature_cols = sorted(set.intersection(*per_batch_feature_cols))
    if not common_feature_cols:
        raise ValueError(
            "No feature column is common to every batch; nothing to median across batches"
        )

    aligned_lfs = []
    for label, lf, batch_cols in zip(batch_labels, batch_lfs, per_batch_feature_cols):
        dropped = sorted(batch_cols - set(common_feature_cols))
        if dropped:
            logging.warning(
                "Batch %s: dropping %d feature column(s) not common to every batch's "
                "aggregate before cross-batch concat: %s",
                label,
                len(dropped),
                dropped,
            )
        aligned_lfs.append(lf.select([label_column, *common_feature_cols]))

    return (
        pl.concat(aligned_lfs)
        .group_by(label_column)
        .agg([pl.col(c).median() for c in common_feature_cols])
        .collect()
    )


def ovwt_global_scores(
    batch_score_dfs: Sequence[pl.DataFrame],
    label_column: str,
    scores: Sequence[str] = OVWT_SCORES,
) -> pl.DataFrame:
    """
    Per experiment, z-score the OvWT AUROCs against that experiment's synonymous variants;
    then, per variant, take the median across experiments.

    Raw AUROCs aren't comparable across experiments (cell counts, feature quality and batch
    effects all move where a neutral variant scores), so each experiment is first
    re-centred on its own untagged synonymous variants: a :class:`Normalizer` fit on their
    rows (mean, ``ddof=1`` std; a std below float32 epsilon, or fewer than two synonymous
    variants, gives null).

    Parameters
    ----------
    batch_score_dfs : sequence of pl.DataFrame
        Each experiment's OVWT_BATCHWISE ``results.parquet``. Must be non-empty.
    label_column : str
        The variant label column.
    scores : sequence of str
        The score columns to pool (default :data:`OVWT_SCORES`). Every other non-``meta_``
        column of ``results.parquet`` (e.g. the per-fold ``auroc_folds`` list) is dropped.

    Returns
    -------
    pl.DataFrame
        One row per variant: ``label_column``, ``meta_median_<score>`` per score, and
        ``meta_num_experiments``, the number of experiments whose z-scored first score is
        non-null.
    """
    if not batch_score_dfs:
        raise ValueError("batch_score_dfs must be non-empty")
    scores = list(scores)
    zscored = []
    for df in batch_score_dfs:
        missing = [s for s in scores if s not in df.columns]
        if missing:
            raise ValueError(f"OvWT results lack score column(s) {missing}")
        meta = [c for c in df.columns if c.startswith("meta_")]
        classified = mark_controls(df.select(*meta, *scores).lazy(), label_column)
        normalizer = Normalizer.from_lazyframe(classified, fit_only_on_control=True)
        zscored.append(
            normalizer.apply(classified).select(label_column, *scores).collect()
        )
    return (
        pl.concat(zscored)
        .group_by(label_column)
        .agg(
            *[pl.col(s).median().alias(f"meta_median_{s}") for s in scores],
            pl.col(scores[0]).count().alias("meta_num_experiments"),
        )
    )


# --- PCA ------------------------------------------------------------------------------------


def n_components_for_variance(
    variance_df: pl.DataFrame, cumulative_variance_explained: float
) -> int:
    """
    The fewest leading components whose ``meta_cumulative_variance_explained`` reaches
    ``cumulative_variance_explained``, or every component if none does.

    ``variance_df`` holds one row per component, in component order.
    """
    if not (0.0 < cumulative_variance_explained <= 1.0):
        raise ValueError(
            "cumulative_variance_explained must be in (0, 1], got "
            f"{cumulative_variance_explained}"
        )
    cumulative = variance_df[CUMULATIVE_VARIANCE_EXPLAINED_COL].to_list()
    for n, value in enumerate(cumulative, start=1):
        if value >= cumulative_variance_explained:
            return n
    return len(cumulative)


def reduced_pca(
    scores_df: pl.DataFrame, label_column: str, n_components: int
) -> pl.DataFrame:
    """
    The leading ``n_components`` PC scores, with ``meta_is_control`` (:func:`mark_controls`)
    and ``meta_impact_score`` (:func:`fisseq_common.utils.vectors.compute_impact_score`,
    the cosine distance to the controls' median) computed on those components.

    Parameters
    ----------
    scores_df : pl.DataFrame
        ``label_column`` and ``meta_pc_1 .. meta_pc_<n>``.
    label_column : str
        The variant label column.
    n_components : int
        How many leading components to keep (:func:`n_components_for_variance`).

    Returns
    -------
    pl.DataFrame
        ``label_column``, ``meta_pc_1 .. meta_pc_<n_components>``, ``meta_is_control``,
        ``meta_impact_score``.
    """
    pc_cols = [f"{PC_COL_PREFIX}{i}" for i in range(1, n_components + 1)]
    classified = mark_controls(
        scores_df.select(label_column, *pc_cols).lazy(), label_column
    )
    # compute_impact_score takes the non-meta_ columns as features: strip the meta_ prefix
    # of the PC columns for that call.
    strip = {c: c.removeprefix("meta_") for c in pc_cols}
    with_impact = compute_impact_score(classified.rename(strip))
    return with_impact.rename({v: k for k, v in strip.items()}).collect()
