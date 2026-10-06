"""Per-variant aggregation of cell-level features: the aggregators and the aggregation stages.

Each :class:`BaseAggregator` turns the cells of every non-control variant into one value per
feature column, suffixed with its statistic (``_mean``, ``_median``, ``_KS``, ...):

- location and spread: ``mean``, ``median``, ``MAD``, ``std``;
- distance from the control cells' distribution (the reference-based aggregators): ``KS``,
  ``signedKS``, ``QQ``, ``AUROC``, and ``KSnegLogP`` / ``AUROCnegLogP``, the ``-log10(p)`` of
  the KS and AUROC statistics (closed-form asymptotic approximations, no multiple-testing
  correction).

Every aggregator excludes the control rows (``meta_is_control``) before grouping by variant:
the controls define the reference distribution, they are not scored against it. Which rows are
controls is the filter stage's choice (:mod:`.filter`). Which columns are features is the
``feature_selector`` parameter.

Feature columns are aggregated ``feature_chunk_size`` at a time (:data:`DEFAULT_FEATURE_CHUNK_SIZE`):
peak memory scales with ``chunk_size x n_labels`` (and the cross-joined control pool, for the
reference-based aggregators). It is a pure memory dial; the output is identical at every chunk
size.

:func:`aggregate` runs one aggregator, :func:`aggregate_methods` several (joined on the label,
with per-variant metadata), and :func:`aggregate_cells` is the per-method stage both pipelines
run for a full experiment or one pseudo-replicate half: optional split selection, optional
control downsampling (:func:`downsample_control`), aggregation, and an optional z-score of the
result against the synonymous variants (:func:`zscore_to_synonymous`).

Entry point: ``python -m fisseq_common.stages.aggregate`` (:class:`AggregateConfig`), the
Nextflow processes ``AGGREGATE_FEATURE_TYPE`` (every cell, z-scored against the synonymous
variants), ``AGGREGATE_HALF`` (one bootstrap half, raw) and ``AGGREGATE_PASSTHROUGH`` (every
cell, raw).
"""

import abc
import dataclasses
import logging
import math
import pathlib
from collections import Counter
from typing import ClassVar, Optional, Sequence, Union

import numpy as np
import polars as pl
from omegaconf import MISSING

from fisseq_common.normalizer import Normalizer
from fisseq_common.schema import CONTROL_COLUMN, CONTROL_COLUMN_NAME, FEATURE_SELECTOR
from fisseq_common.utils.metadata import get_aggregate_meta_data
from fisseq_common.utils.splits import filter_by_split_file

from .config import AppConfig, CellsInput, feature_selector, row_keys, stage_main
from .filter import load_cells, variant_classification

#: Number of feature columns aggregated per Polars query when no explicit
#: ``feature_chunk_size`` is given. Aggregating every feature in one query is
#: what OOM-killed the production ``AGGREGATE_FEATURE_TYPE`` / ``AGGREGATE_HALF``
#: tasks: peak memory scales with ``chunk_size x n_labels`` (and, for the
#: reference-based aggregators, the cross-joined control pool on top).
#:
#: This default is sized to the memory one task is granted, not to a property
#: of the data -- ``params.aggregate_feature_chunk_size`` is the knob, and
#: ``params.yaml`` carries the sizing rule. ``32`` assumes >=128 GB per task at
#: production shape, where the reference-based aggregators cost roughly 3 GB
#: (AUROC) to 4.5 GB (KS) per feature in the chunk. See
#: ``tests/benchmarks/README.md``.
#:
#: ``None`` disables chunking entirely -- every feature in one query, the
#: pre-chunking behaviour. That is what OOM-killed production, so it is an
#: opt-in escape hatch (for small inputs, or for reproducing the old shape),
#: not a supported setting for a full-size batch.
DEFAULT_FEATURE_CHUNK_SIZE: int = 32


class BaseAggregator(abc.ABC):
    r"""
    Base class for all aggregators.

    Subclasses declare :attr:`_stat_suffix` (e.g. ``"_mean"``, ``"_KS"``)
    and implement :meth:`_feature_expr`, a native Polars list expression
    for one feature column. :meth:`aggregate` handles everything
    else: resolving the feature columns (``feature_selector``),
    building the reference pool (only for :class:`ReferenceBasedAggregator`
    subclasses -- see :meth:`_reference_lf`), grouping non-control rows
    into per-label list columns, and assembling the final per-dimension
    expressions -- entirely in Arrow, no numpy materialization, no
    per-(group, dimension) Python loop.

    Parameters
    ----------
    label_col : str
        Name of the column used to identify variant groups. Defaults to
        ``"meta_aa_changes"``.
    feature_selector : pl.Expr
        Polars selector expression identifying feature columns to
        aggregate. Defaults to ``FEATURE_SELECTOR`` (every non-``meta_``
        column); the embeddings pipeline passes ``EMBEDDING_SELECTOR``.
    """

    _stat_suffix: ClassVar[str]

    def __init__(
        self,
        label_col: str = "meta_aa_changes",
        feature_selector: pl.Expr = FEATURE_SELECTOR,
        feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE,
    ) -> None:
        if feature_chunk_size is not None and feature_chunk_size < 1:
            raise ValueError(
                f"feature_chunk_size must be >= 1 or None, got {feature_chunk_size}"
            )
        self.label_col = label_col
        self.feature_selector = feature_selector
        self.feature_chunk_size = feature_chunk_size

    def _feature_chunks(self, feature_cols: list[str]) -> list[list[str]]:
        """
        Split ``feature_cols`` into :attr:`feature_chunk_size`-wide chunks.

        A :attr:`feature_chunk_size` of ``None`` yields one chunk holding
        every dimension. An empty ``feature_cols`` still yields one (empty)
        chunk rather than no chunks at all -- :meth:`aggregate` must emit its
        one-row-per-label frame either way.
        """
        size = self.feature_chunk_size
        if size is None:
            return [feature_cols]
        chunks = [feature_cols[i : i + size] for i in range(0, len(feature_cols), size)]
        return chunks or [[]]

    def _feature_columns(self, lf: pl.LazyFrame) -> list[str]:
        return lf.select(self.feature_selector).collect_schema().names()

    @staticmethod
    def _native_clean(feat: str) -> pl.Expr:
        """
        Drop null, NaN, and Inf entries from a per-label list column for
        ``feat`` before any further native list-based computation.
        ``is_finite()`` on a Float64 element is ``False`` for null/NaN/Inf
        alike -- the same filter :meth:`ReferenceBasedAggregator._reference_lf`
        applies to the flat reference-pool columns.
        """
        return pl.col(feat).list.eval(pl.element().filter(pl.element().is_finite()))

    @staticmethod
    def _reference_lf(
        lf: pl.LazyFrame, feature_cols: list[str]
    ) -> Optional[pl.LazyFrame]:
        """
        Reference frame to cross-join before computing :meth:`_feature_expr`,
        or ``None``. Overridden by :class:`ReferenceBasedAggregator`; plain
        aggregators (mean/median) don't need a reference pool at all.
        """
        return None

    def _native_aggregate_feature_batch(
        self,
        lf: pl.LazyFrame,
        feature_cols: list[str],
        exprs: list[pl.Expr],
        reference_lf: Optional[pl.LazyFrame],
    ) -> pl.LazyFrame:
        """
        Shared group_by/select boilerplate: group non-control rows by
        ``self.label_col`` into per-group list columns, then select
        ``exprs`` (one aliased native-Polars expression per embedding
        dimension) against those list columns. Stays entirely in Arrow --
        no Python boxing.

        ``reference_lf``, if given, is the single-row
        :meth:`ReferenceBasedAggregator._reference_lf` output; it's
        cross-joined onto the per-group list frame so every group row also
        carries each dimension's ``{feat}_ref`` reference list column. A
        single-row cross join only broadcasts the reference row onto every
        existing group row -- it does not multiply row count.
        """
        variant_lists = (
            lf.filter(~CONTROL_COLUMN)
            .group_by(self.label_col)
            .agg([pl.col(f) for f in feature_cols])
        )
        if reference_lf is not None:
            variant_lists = variant_lists.join(reference_lf, how="cross")
        prep_exprs = [e for feat in feature_cols for e in self._prep_exprs(feat)]
        if prep_exprs:
            variant_lists = variant_lists.with_columns(prep_exprs)
        return variant_lists.select([self.label_col] + exprs)

    def _prep_exprs(self, feat: str) -> list[pl.Expr]:
        """
        Optional helper expressions to materialize via a ``with_columns``
        pass before ``_feature_expr`` runs, keyed by feature name. Empty
        by default (no extra pass, no behavior or performance change for
        aggregators that don't override this). See
        :class:`SignedKSAggregator` for the motivating case --
        kept as a hook for any future aggregator that needs to hoist a
        subexpression referenced more than once.
        """
        return []

    def aggregate(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute per-label statistics for every feature column.

        Dimensions are processed :attr:`feature_chunk_size` at a time. Each
        chunk projects ``lf`` down to the label column, the control flag and
        that chunk's dimensions *before* grouping -- that projection is the
        whole point, since it lets the Parquet scan read only those columns
        and keeps both the grouped list columns and the cross-joined
        reference pool proportional to the chunk width rather than to the
        total dimension count. The per-chunk results (one row per label, a
        few hundred columns each) are joined back together on the label,
        which costs nothing next to the grouped cell-level data.

        Parameters
        ----------
        lf : pl.LazyFrame
            Input LazyFrame containing the label column, a boolean
            ``CONTROL_COLUMN`` column, and feature
            columns.

        Returns
        -------
        pl.LazyFrame
            One row per non-control variant group with computed statistics,
            sorted by the label column.
        """
        feature_cols = self._feature_columns(lf)
        chunks = self._feature_chunks(feature_cols)
        logging.info(
            "%s: %d feature column(s) to aggregate in %d chunk(s) of <=%s",
            type(self).__name__,
            len(feature_cols),
            len(chunks),
            self.feature_chunk_size if self.feature_chunk_size is not None else "all",
        )

        meta_cols = [self.label_col, CONTROL_COLUMN_NAME]
        result: Optional[pl.DataFrame] = None
        for chunk_idx, chunk in enumerate(chunks, start=1):
            logging.debug(
                "%s: chunk %d/%d (%d column(s))",
                type(self).__name__,
                chunk_idx,
                len(chunks),
                len(chunk),
            )
            sub = lf.select(meta_cols + chunk)
            reference_lf = self._reference_lf(sub, chunk)
            exprs = [self._feature_expr(f) for f in chunk]
            part = self._native_aggregate_feature_batch(
                sub, chunk, exprs, reference_lf
            ).collect()
            result = (
                part
                if result is None
                else result.join(part, on=self.label_col, how="left")
            )

        # Required, not cosmetic: group_by is not order-preserving under
        # Polars' multithreaded execution, so without this the row order
        # (and therefore every downstream file) varies run to run.
        assert result is not None  # _feature_chunks never returns []
        return result.sort(self.label_col).lazy()

    @abc.abstractmethod
    def _feature_expr(self, feat: str) -> pl.Expr:
        """
        Native Polars expression computing this aggregator's statistic for
        one feature column, evaluated against the per-label list
        columns (and, for :class:`ReferenceBasedAggregator` subclasses, the
        cross-joined ``{feat}_ref`` column) built by :meth:`aggregate`.
        """
        raise NotImplementedError


class ReferenceBasedAggregator(BaseAggregator):
    """
    Base for aggregators that compare each variant group against a shared
    control/reference pool (KS, QQ, AUROC): builds the single-row reference
    frame and lets :meth:`BaseAggregator.aggregate` cross-join it in
    automatically.
    """

    @staticmethod
    def _reference_lf(lf: pl.LazyFrame, feature_cols: list[str]) -> pl.LazyFrame:
        """
        Single-row LazyFrame holding one ``{feat}_ref`` list column per
        feature with the finite control-row values for that feature.

        The reference pool is shared by every variant label (a single
        global control group, not split per-label), so this stays a single
        row and is cross-joined onto the per-label variant-list frame
        rather than collected eagerly: the reference pool and the
        per-group variant lists then live in the same lazy query graph,
        letting the streaming engine manage memory instead of Python
        holding numpy arrays for the whole aggregation call.

        ``is_finite()`` is ``False`` for null/NaN/Inf alike, so filtering
        on it drops all three in one pass — the same set
        :meth:`BaseAggregator._native_clean` drops from per-group list
        columns.
        """
        exprs = [
            pl.col(f).filter(pl.col(f).is_finite()).implode().alias(f"{f}_ref")
            for f in feature_cols
        ]
        return lf.filter(CONTROL_COLUMN).select(exprs)


class MeanAggregator(BaseAggregator):
    """Computes per-group mean for each feature column."""

    _stat_suffix = "_mean"

    def _feature_expr(self, feat: str) -> pl.Expr:
        return self._native_clean(feat).list.mean().alias(f"{feat}{self._stat_suffix}")


class MedianAggregator(BaseAggregator):
    """Computes per-group median for each feature column."""

    _stat_suffix = "_median"

    def _feature_expr(self, feat: str) -> pl.Expr:
        return (
            self._native_clean(feat).list.median().alias(f"{feat}{self._stat_suffix}")
        )


class MADAggregator(BaseAggregator):
    """Computes per-group median absolute deviation (MAD) for each feature column."""

    _stat_suffix = "_MAD"

    def _feature_expr(self, feat: str) -> pl.Expr:
        # Verified against np.median(np.abs(vals - np.median(vals))) for
        # empty, single-value, normal, and NaN/Inf-present cases — matches
        # exactly (see test_mad_native_matches_numpy_with_nan_inf_present).
        return (
            self._native_clean(feat)
            .list.eval((pl.element() - pl.element().median()).abs())
            .list.median()
            .alias(f"{feat}{self._stat_suffix}")
        )


class StdAggregator(BaseAggregator):
    """Computes per-group standard deviation for each feature column."""

    _stat_suffix = "_std"

    def _feature_expr(self, feat: str) -> pl.Expr:
        # Confirmed: list.std(ddof=1) returns null for lists of length < 2
        # (see test_std_native_single_value_group_returns_null).
        return (
            self._native_clean(feat)
            .list.std(ddof=1)
            .alias(f"{feat}{self._stat_suffix}")
        )


class KSAggregator(ReferenceBasedAggregator):
    """
    Computes per-group two-sample Kolmogorov-Smirnov statistics against
    the reference distribution for each feature column.
    """

    _stat_suffix = "_KS"

    def _ks_stat_expr(self, feat: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """
        Returns ``(ks_stat, n_group, n_ref)`` as unaliased, un-nulled
        exprs — the raw two-sample KS statistic and the group/reference
        sizes it was computed from, before :meth:`_feature_expr`'s
        null-handling and aliasing. Shared with
        :class:`KSNegLogPValueAggregator`, which reuses ``ks_stat`` and
        the sizes to derive a p-value instead of recomputing the
        statistic.
        """
        ref_list = pl.col(f"{feat}_ref")
        n_ref = ref_list.list.len()

        # Signed-weight cumulative-sum KS statistic: +1/n_group per variant
        # value, -1/n_ref per reference value. Sort the combined values,
        # cumsum the weights, and take the max |cumsum| — but only at the
        # LAST position of each run of tied values (ties must be resolved
        # together, not mid-tie, or a spurious intermediate extremum can
        # exceed the true statistic). Verified against
        # scipy.stats.ks_2samp across 500 randomized trials, including
        # tie-heavy integer data and n=1 groups.
        group_list = self._native_clean(feat)
        n_group = group_list.list.len()

        g_weight = group_list.list.eval((pl.element() * 0 + 1.0) / pl.element().count())
        ref_weight = ref_list.list.eval((pl.element() * 0 - 1.0) / pl.element().count())
        combined_val = pl.concat_list([group_list, ref_list])
        combined_w = pl.concat_list([g_weight, ref_weight])

        order = combined_val.list.eval(pl.element().arg_sort())
        val_sorted = combined_val.list.gather(order)
        w_sorted = combined_w.list.gather(order)

        cumsum = w_sorted.list.eval(pl.element().cum_sum())
        next_val = val_sorted.list.shift(-1)
        # `!=` between two List columns is whole-list equality, not
        # element-wise — subtract instead (arithmetic *is* element-wise
        # for List columns) and compare each element to 0 inside list.eval.
        diff = val_sorted - next_val
        is_last_f = diff.list.eval(
            ((pl.element() != 0) | pl.element().is_null()).cast(pl.Float64)
        )
        candidate = cumsum.list.eval(pl.element().abs())
        # Non-last positions become 0.0 (multiply, not pl.when — when/then
        # does not broadcast element-wise over a List(Boolean) predicate),
        # which never wins the subsequent max since a KS statistic is >= 0.
        ks_stat = (candidate * is_last_f).list.max()

        return ks_stat, n_group, n_ref

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ks_stat, n_group, n_ref = self._ks_stat_expr(feat)

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(n_group == 0)
            .then(None)
            .otherwise(ks_stat)
        )
        return result.alias(alias)


class KSNegLogPValueAggregator(KSAggregator):
    """
    Computes ``-log10(p)`` of the two-sample Kolmogorov-Smirnov statistic
    against the reference distribution, for each feature column.

    Reports evidence strength against the null hypothesis "this variant
    group's values are drawn from the same distribution as the reference
    control" — larger values mean more significant departures. This
    reuses the exact same KS D-statistic as :class:`KSAggregator` (via
    :meth:`KSAggregator._ks_stat_expr`) rather than recomputing it; only
    the D -> p-value conversion is new.

    The p-value is computed from the asymptotic (large-sample /
    "limiting") two-sided Kolmogorov distribution — the classical
    closed-form ``Q(x) = 2 * sum_{k=1}^inf (-1)^(k-1) exp(-2 k^2 x^2)``
    with ``x = D * sqrt(n_e)``, ``n_e = n_group * n_ref / (n_group +
    n_ref)`` — not an exact finite-sample distribution. This is the same
    conceptual caveat as scipy's ``ks_2samp(..., mode='asymp')``, and is
    appropriate here given the large per-well/per-guide cell counts this
    pipeline typically works with. Note, however, that this is
    numerically the classical limiting distribution (equivalent to
    ``scipy.stats.kstwobign.sf(D * sqrt(n_e))``, matches to ~1e-14), which
    is *not* the same number as scipy's own ``kstwo.sf``-based asymp
    p-value (``ks_2samp``'s default 'asymp' mode uses a more refined
    finite-sample algorithm from Marsaglia et al. that can differ from
    this classical formula by tens of percent in ``p`` even at n in the
    hundreds, since the classical KS limit theorem converges slowly) —
    see the test suite for a direct comparison of the two.

    The truncated alternating series this is computed from is reliable in
    the significant/large-D regime this aggregator is meant to rank
    variants by, but is numerically imprecise very close to D=0 (p near
    1, ``-log10(p)`` near 0). That's not a problem in practice here, since
    near-null variants aren't the ones being ranked by this statistic —
    don't read precision into values near 0.

    Benchmark (300 labels x 300 features, 30 cells/group synthetic data;
    see ``tests/benchmarks/benchmark_pvalue_aggregators.py``):
    ``KSAggregator`` took 2.53s / 306.6KB peak traced Python allocations
    (839.6MB RSS delta) vs. ``KSNegLogPValueAggregator``'s 14.24s / 702.1KB
    (973.5MB RSS delta) — computing the p-value adds ~5.6x time and
    ~134MB additional peak RSS over the base KS statistic alone (dominated
    by the ``k_terms=100``-wide logsumexp series evaluated per
    (label, feature) pair).
    """

    _stat_suffix = "_KSnegLogP"

    @staticmethod
    def _neg_log10_kolmogorov_pvalue_expr(
        d: pl.Expr, n_e: pl.Expr, k_terms: int = 100
    ) -> pl.Expr:
        """
        ``-log10(p)`` for the two-sided asymptotic KS p-value, computed
        via logsumexp over the alternating Kolmogorov series to stay
        finite for small p (large ``D * sqrt(n_e)``), where naively
        summing ``exp(...)`` terms in linear space underflows to exactly
        0.0 and produces ``-log10(0)`` = null/inf well before the true
        p-value is small enough to stop mattering for ranking hits.

        Writing ``S = sum_{k=1}^{k_terms} (-1)^(k-1) exp(-2 k^2 n_e D^2)``
        and ``m`` for the k=1 term's exponent (the largest, i.e. least
        negative, since the exponent is monotonically decreasing in k):
        every term factors as ``sign_k * exp(m) * exp(exponent_k - m)``,
        so ``S = exp(m) * R`` where ``R = sum_k sign_k * exp(exponent_k -
        m)``. Each ``exp(exponent_k - m)`` is bounded in ``(0, 1]`` (since
        ``exponent_k - m <= 0``), so ``R`` is a well-conditioned O(1) sum
        safe to compute in linear space, while ``m`` itself — the only
        part that can be arbitrarily large in magnitude — is never
        exponentiated: ``log10(p) = log10(2) + m / ln(10) + log10(R)`` is
        computed directly from ``m`` and ``log10(R)``, both finite.

        Validated against ``scipy.stats.kstwobign.sf(D * sqrt(n_e))``
        (max abs error ~6e-14 across 20 randomized trials, D in
        [0.01, 0.999], n_e in [2, 500]).
        """
        ks = np.arange(1, k_terms + 1, dtype=np.float64)
        # Compile-time literals (not per-row): k^2 and the alternating
        # sign for k=1..k_terms, each a 1-row List(Float64) that
        # broadcasts row-wise against the N-row `m`-derived exprs below —
        # same idiom as QQCorrelationAggregator._native_quantiles'
        # `probs_lit`.
        k_sq = pl.lit(pl.Series([(ks**2).tolist()]))
        sign = pl.lit(pl.Series([((-1.0) ** (ks - 1)).tolist()]))

        m = -2.0 * n_e * d.pow(2)  # k=1 exponent; largest (least negative)
        exponents = k_sq * m
        rel = exponents - m  # <= 0 elementwise, safe to exponentiate
        r_terms = sign * rel.list.eval(pl.element().exp())
        big_r = r_terms.list.sum()

        ln10 = np.log(10.0)
        neg_log10_p = -(np.log10(2.0) + m / ln10 + big_r.log10())
        # Clamps float noise near D~0 (p slightly > 1) and the degenerate
        # D=0 case, where the alternating series is a near-zero
        # oscillation (Grandi's-series-like) rather than a clean sum —
        # -log10(p) can never legitimately be negative (p <= 1 always).
        return neg_log10_p.clip(lower_bound=0.0)

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ks_stat, n_group, n_ref = self._ks_stat_expr(feat)
        n_e = n_group * n_ref / (n_group + n_ref)
        neg_log10_p = self._neg_log10_kolmogorov_pvalue_expr(ks_stat, n_e)

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(n_group == 0)
            .then(None)
            .otherwise(neg_log10_p)
        )
        return result.alias(alias)


class SignedKSAggregator(ReferenceBasedAggregator):
    """
    Computes per-group signed Kolmogorov-Smirnov statistics against the
    reference distribution for each feature column.

    Same magnitude as :class:`KSAggregator`, but signed by which empirical
    CDF is larger at the maximizing point: positive when the variant
    group's CDF exceeds the reference's there (group values skew lower
    than reference), negative when the reference's CDF is larger (group
    values skew higher than reference).
    """

    _stat_suffix = "_signedKS"
    _SENTINEL: ClassVar[float] = 10.0

    @staticmethod
    def _cumsum_col(feat: str) -> str:
        return f"__signedks_cumsum__{feat}"

    @staticmethod
    def _penalty_col(feat: str) -> str:
        return f"__signedks_penalty__{feat}"

    @staticmethod
    def _grouplen_col(feat: str) -> str:
        return f"__signedks_grouplen__{feat}"

    def _prep_exprs(self, feat: str) -> list[pl.Expr]:
        """
        Materializes, once per feature, the expensive shared
        subexpressions that :meth:`_feature_expr` needs twice (for its
        max and min reductions): the signed-weight cumulative sum and
        the tie-position penalty mask. See
        :meth:`BaseAggregator._prep_exprs` for why this hoisting is
        required for performance, not just style.

        Same signed-weight cumulative-sum construction as KSAggregator:
        the cumsum at a combined-sorted position IS
        F_group(x) - F_ref(x), so unlike the unsigned statistic we keep
        the sign of the maximizing value instead of discarding it via
        abs().
        """
        ref_list = pl.col(f"{feat}_ref")
        group_list = self._native_clean(feat)

        g_weight = group_list.list.eval((pl.element() * 0 + 1.0) / pl.element().count())
        ref_weight = ref_list.list.eval((pl.element() * 0 - 1.0) / pl.element().count())
        combined_val = pl.concat_list([group_list, ref_list])
        combined_w = pl.concat_list([g_weight, ref_weight])

        order = combined_val.list.eval(pl.element().arg_sort())
        val_sorted = combined_val.list.gather(order)
        w_sorted = combined_w.list.gather(order)

        cumsum = w_sorted.list.eval(pl.element().cum_sum())
        next_val = val_sorted.list.shift(-1)
        diff = val_sorted - next_val
        is_last_f = diff.list.eval(
            ((pl.element() != 0) | pl.element().is_null()).cast(pl.Float64)
        )
        # cumsum is provably bounded in [-1, 1]: g_weight sums to
        # exactly +1.0 over the group's values, ref_weight to exactly
        # -1.0 over the reference's, so any prefix sum is a difference
        # of two fractions each in [0, 1]. SENTINEL only needs to
        # exceed 2.0 to push a non-last position out of range in both
        # directions; 10.0 gives headroom. is_last_f is exactly 0.0/1.0,
        # so this additive masking has no 0 * inf risk.
        penalty = (1.0 - is_last_f) * self._SENTINEL

        return [
            cumsum.alias(self._cumsum_col(feat)),
            penalty.alias(self._penalty_col(feat)),
            group_list.list.len().alias(self._grouplen_col(feat)),
        ]

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ref_list = pl.col(f"{feat}_ref")
        n_ref = ref_list.list.len()

        cumsum = pl.col(self._cumsum_col(feat))
        penalty = pl.col(self._penalty_col(feat))
        group_len = pl.col(self._grouplen_col(feat))

        # Extremum via two reductions over the materialized cumsum/penalty
        # columns instead of arg_max + indexed gather (list.get).
        #
        # Tie-break convention (deliberate, documented): if the max and
        # min last-position cumsum values have exactly equal magnitude
        # but opposite sign, the POSITIVE value wins. See
        # test_signed_ks_aggregator_tie_break_prefers_positive_on_exact_magnitude_tie.
        masked_for_max = cumsum - penalty
        masked_for_min = cumsum + penalty
        pos_max = masked_for_max.list.max()
        neg_min = masked_for_min.list.min()
        signed_stat = (
            pl.when(pos_max.abs() >= neg_min.abs()).then(pos_max).otherwise(neg_min)
        )

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(group_len == 0)
            .then(None)
            .otherwise(signed_stat)
        )
        return result.alias(alias)


class QQCorrelationAggregator(ReferenceBasedAggregator):
    """
    Computes per-group Q-Q correlation against the reference distribution for
    each feature column.

    Parameters
    ----------
    label_col : str
        Variant label column name.
    n_quantiles : int, optional
        Number of quantile points to evaluate. Defaults to ``100``.
    feature_selector : pl.Expr
        Polars selector identifying feature columns. Defaults to ``FEATURE_SELECTOR``.
    feature_chunk_size : int or None
        Number of feature columns evaluated per Polars query. ``None``
        disables chunking. Defaults to :data:`DEFAULT_FEATURE_CHUNK_SIZE`.
    """

    _stat_suffix = "_QQ"

    def __init__(
        self,
        label_col: str = "meta_aa_changes",
        n_quantiles: int = 100,
        feature_selector: pl.Expr = FEATURE_SELECTOR,
        feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE,
    ) -> None:
        super().__init__(label_col, feature_selector, feature_chunk_size)
        self.quantile_points = np.linspace(0, 1, n_quantiles)

    @staticmethod
    def _native_quantiles(list_expr: pl.Expr, probs: np.ndarray) -> pl.Expr:
        """
        Per-row quantiles of a list column at fixed probability points
        ``probs``, matching ``np.quantile(..., interpolation="linear")``
        bit-for-bit (verified across 200 randomized trials, including
        single-element and empty lists).

        Computed via one sort + a per-row index gather rather than one
        ``list.eval(...quantile...)`` call per probability point: with 100
        quantile points and hundreds of features per aggregation call, the
        naive one-expression-per-point approach makes the ``.select()``
        call contain ``n_quantiles * n_features`` sub-expressions (50,000
        at the defaults), which measured *slower* than the numpy loop it
        replaces (247.9s vs a 157.4s baseline at 500 labels x 500 features
        x 100 quantiles). This version measured 22.5s at the same scale.

        ``null_on_oob=True`` on the gathers is required, not defensive
        padding: an empty list (``n=0``) produces negative/garbage indices
        after the ``(n - 1)`` scaling, and Polars' ``when/otherwise`` does
        not short-circuit — the ``otherwise`` branch is evaluated for every
        row before the mask is applied, so an out-of-bounds gather raises
        regardless of whether that row's result is later discarded
        (confirmed directly: without ``null_on_oob``, a single empty group
        anywhere in the data crashes the whole query with
        ``OutOfBoundsError``).
        """
        n = list_expr.list.len()
        sorted_list = list_expr.list.eval(pl.element().sort())
        probs_lit = pl.lit(pl.Series([probs.tolist()]))
        idx = probs_lit * (n - 1).cast(pl.Float64)
        lower_f = idx.list.eval(pl.element().floor())
        upper_f = idx.list.eval(
            (pl.element().floor() + 1).clip(upper_bound=pl.element().max())
        )
        frac = idx.list.eval(pl.element() - pl.element().floor())
        lower_idx = lower_f.list.eval(pl.element().cast(pl.UInt32))
        upper_idx = upper_f.list.eval(pl.element().cast(pl.UInt32))
        lower_vals = sorted_list.list.gather(lower_idx, null_on_oob=True)
        upper_vals = sorted_list.list.gather(upper_idx, null_on_oob=True)
        return lower_vals + frac * (upper_vals - lower_vals)

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        ref_list = pl.col(f"{feat}_ref")
        n_ref = ref_list.list.len()

        # Reference-side quantiles and their centering are fixed per
        # feature (the reference pool doesn't vary by group) but are now
        # computed per row via _native_quantiles rather than once in plain
        # numpy, since the reference pool lives in the query graph — every
        # row gets the identical broadcast reference list post-cross-join,
        # so this recomputes the same quantiles redundantly per group
        # rather than once per feature.
        ref_q = self._native_quantiles(ref_list, self.quantile_points)
        # Constant reference quantile profile (e.g. a single distinct
        # reference value) -> correlation is undefined for every group
        # regardless of its own data (scipy.stats.pearsonr would raise
        # ConstantInputWarning and return nan here).
        ref_constant = ref_q.list.max() == ref_q.list.min()
        y_centered = ref_q.list.eval(pl.element() - pl.element().mean())
        var_y_sum = y_centered.list.eval(pl.element() ** 2).list.sum()

        group_list = self._native_clean(feat)
        variant_q = self._native_quantiles(group_list, self.quantile_points)

        # Pearson r, computed manually (mean-center each side, then
        # dot-product over sqrt of sum-of-squares) rather than via
        # pl.corr, which correlates two *columns across rows* — not the
        # paired elements *within* one row's two quantile lists.
        x_centered = variant_q.list.eval(pl.element() - pl.element().mean())
        num = (x_centered * y_centered).list.sum()
        denom_x = x_centered.list.eval(pl.element() ** 2).list.sum()
        denom = (denom_x * var_y_sum).sqrt()

        # A group with a single distinct value has an exactly constant
        # quantile profile (quantile(0) == quantile(1) == that value, since
        # the grid includes both endpoints), so its correlation with the
        # reference is mathematically undefined — same case scipy flags via
        # ConstantInputWarning. This is checked on
        # group_list.list.n_unique(), not solely on `denom == 0`, because
        # mean-centering a repeated constant is *not* guaranteed to cancel
        # to bit-exact zero in floating point (e.g. 100 copies of
        # 0.09048978162787422 summed then divided by 100 comes back
        # ~2.8e-17 off), which would otherwise leak a bogus near-zero
        # correlation instead of null — caught directly via a real n=1
        # group during verification.
        is_constant_group = group_list.list.n_unique() <= 1

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(ref_constant)
            .then(None)
            .when(group_list.list.len() == 0)
            .then(None)
            .when(is_constant_group)
            .then(None)
            .when(denom == 0)
            .then(None)
            .otherwise(num / denom)
        )
        return result.alias(alias)


class AUROCAggregator(ReferenceBasedAggregator):
    """
    Computes per-group AUROC against the reference distribution for each
    feature column.

    Variant samples are labelled ``1`` and reference samples ``0``. ``0.5``
    indicates identical distributions; ``1.0`` indicates the variant
    group's values are consistently higher than the reference; ``0.0``
    indicates they are consistently lower. Unlike a typical classification
    AUROC, this value is *not* symmetrized to ``[0.5, 1]`` — it reports
    ``P(variant > reference) + 0.5 * P(variant == reference)`` directly, so
    the sign of separation is preserved in the value itself.
    """

    _stat_suffix = "_AUROC"

    def _auroc_ranks_expr(self, feat: str) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
        """
        Returns ``(combined, ranks, n_group, n_ref)``: the concatenated
        ``[group_values, ref_values]`` list for this feature, its
        ``rank(method="average")`` transform (feeds the rank-sum/U
        statistic below), and the group/reference sizes.

        Exposed separately from :meth:`_auroc_u_expr` so
        :class:`AUROCNegLogPValueAggregator` can derive its tie-correction
        term from the same combined value list (via
        ``rank(method="min"/"max")``) without rebuilding
        ``group_list``/``ref_list``/``concat_list`` from scratch.

        Rank-sum (Mann-Whitney U) identity: rank the combined pool with
        average ranks for ties (matches sklearn.metrics.roc_auc_score's
        own tie handling — verified across 300 randomized trials,
        continuous and tied data). concat_list preserves element order,
        so the group's own ranks are exactly the first n_group entries
        of the ranked combined list.
        """
        ref_list = pl.col(f"{feat}_ref")
        group_list = self._native_clean(feat)
        combined = pl.concat_list([group_list, ref_list])
        ranks = combined.list.eval(pl.element().rank(method="average"))
        return combined, ranks, group_list.list.len(), ref_list.list.len()

    def _auroc_u_expr(self, feat: str) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
        """
        Returns ``(u, n_group, n_ref)`` as unaliased, un-nulled exprs —
        the raw Mann-Whitney U statistic (numerator before dividing by
        ``n_group * n_ref``) and the group/reference sizes. Shared with
        :class:`AUROCNegLogPValueAggregator`, which reuses ``u`` and the
        sizes to derive a p-value instead of recomputing ranks.
        """
        _combined, ranks, n_group, n_ref = self._auroc_ranks_expr(feat)
        group_rank_sum = ranks.list.slice(0, n_group).list.sum()
        u = group_rank_sum - n_group * (n_group + 1) / 2
        return u, n_group, n_ref

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        u, n_group, n_ref = self._auroc_u_expr(feat)
        auroc = u / (n_group * n_ref)

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(n_group == 0)
            .then(None)
            .otherwise(auroc)
        )
        return result.alias(alias)


class AUROCNegLogPValueAggregator(AUROCAggregator):
    """
    Computes ``-log10(p)`` of the AUROC / Mann-Whitney U statistic against
    the reference distribution, for each feature column.

    Reports evidence strength against the null hypothesis "this variant
    group's values are drawn from the same distribution as the reference
    control" — larger values mean more significant departures. This
    reuses the exact same rank-sum/U statistic as :class:`AUROCAggregator`
    (via :meth:`AUROCAggregator._auroc_u_expr` and
    :meth:`AUROCAggregator._auroc_ranks_expr`) rather than recomputing
    ranks or U independently; only the U -> p-value conversion is new.

    The p-value comes from the standard asymptotic-normal approximation
    to the Mann-Whitney U distribution with a tie correction (``tie_term
    = sum over tie-groups of (t^3 - t)``, ``t`` = each tied value's group
    size):

        N = n_group + n_ref
        mu_U = n_group * n_ref / 2
        sigma2_U = (n_group * n_ref / 12) * ((N + 1) - tie_term / (N * (N - 1)))
        z = (U - mu_U) / sqrt(sigma2_U)
        p = 2 * (1 - Phi(|z|))

    Deliberately **no continuity correction** is applied (unlike
    ``scipy.stats.mannwhitneyu``'s default ``use_continuity=True``) — this
    matches the plain textbook asymptotic-normal formula above. Tests
    compare against ``scipy.stats.mannwhitneyu(..., use_continuity=False)``
    for exact agreement (~1e-8, limited only by the erfc rational
    approximation's own error bound); the continuity-corrected default
    differs slightly (a fixed 0.5 shift toward ``mu_U``), which matters
    more for small samples than the large per-well/per-guide cell counts
    these pipelines typically work with.

    Since Polars has no built-in normal CDF, ``Phi`` is implemented via a
    rational approximation to ``erfc`` — see :meth:`_log_erfc_expr` for
    the formula and citation. Critically, the final p-value is computed
    by staying in **log space end to end** (see
    :meth:`_neg_log10_two_sided_normal_pvalue_expr`): naively computing
    ``Phi(z)`` in linear space and then taking ``-log10(2*(1-Phi(|z|)))``
    underflows to exactly 0.0 (-> null/-inf) for exactly the
    most-significant, large-|z| hits this aggregator exists to rank —
    well before z is "impossibly significant" for these sample sizes.

    Benchmark (300 labels x 300 features, 30 cells/group synthetic data;
    see ``tests/benchmarks/benchmark_pvalue_aggregators.py``):
    ``AUROCAggregator`` took 0.93s / 283.1KB peak traced Python allocations
    vs. ``AUROCNegLogPValueAggregator``'s 2.66s / 660.4KB — computing the
    p-value adds ~2.9x time over the base AUROC statistic alone (the extra
    ``_prep_exprs`` pass for the tie-correction term and the rank/U
    recomputation inside it — see the note on that in the source); no
    measurable additional peak RSS at this scale.
    """

    _stat_suffix = "_AUROCnegLogP"

    @staticmethod
    def _u_col(feat: str) -> str:
        return f"__aurocnlp_u__{feat}"

    @staticmethod
    def _ngroup_col(feat: str) -> str:
        return f"__aurocnlp_ngroup__{feat}"

    @staticmethod
    def _nref_col(feat: str) -> str:
        return f"__aurocnlp_nref__{feat}"

    @staticmethod
    def _tieterm_col(feat: str) -> str:
        return f"__aurocnlp_tieterm__{feat}"

    def _prep_exprs(self, feat: str) -> list[pl.Expr]:
        """
        Materializes, once per feature, the quantities
        :meth:`_feature_expr` needs more than once downstream (``u``,
        ``n_group``, ``n_ref``, ``tie_term``) — see
        :meth:`BaseAggregator._prep_exprs` for why this hoisting is
        required for performance, not just style (same reasoning
        documented for :class:`SignedKSAggregator`).

        ``tie_term`` uses the identity
        ``sum_groups(t^3 - t) == sum_elements(tie_size(x_i)^2 - 1)``
        (verified against a ``collections.Counter``-based reference
        computation across 20 randomized tie-heavy integer trials),
        computed natively from per-element min/max rank without any
        groupby or ``value_counts`` inside ``list.eval``: a tied run's
        every element shares the same ``[rank_min, rank_max]`` window, so
        ``tie_size = rank_max - rank_min + 1`` recovers each element's
        tie-group size directly.
        """
        u, n_group, n_ref = self._auroc_u_expr(feat)
        combined, _ranks, _n_group, _n_ref = self._auroc_ranks_expr(feat)
        rank_min = combined.list.eval(pl.element().rank(method="min"))
        rank_max = combined.list.eval(pl.element().rank(method="max"))
        # rank(...) is UInt32-typed; cast to Float64 first (elementwise
        # multiply, not `**`/`pow`, since Polars' `pow` doesn't support a
        # List-typed base).
        tie_size = (rank_max - rank_min + 1).list.eval(pl.element().cast(pl.Float64))
        tie_term = (tie_size * tie_size - 1).list.sum()
        return [
            u.alias(self._u_col(feat)),
            n_group.alias(self._ngroup_col(feat)),
            n_ref.alias(self._nref_col(feat)),
            tie_term.alias(self._tieterm_col(feat)),
        ]

    @staticmethod
    def _log_erfc_expr(x: pl.Expr) -> pl.Expr:
        """
        Natural log of ``erfc(x)`` for ``x >= 0``, via the Numerical
        Recipes rational/Chebyshev approximation (Press, Teukolsky,
        Vetterling & Flannery, *Numerical Recipes*, 3rd ed., S6.2.2,
        "erfcc"; documented fractional error < 1.2e-7 across the entire
        domain — not just a tail expansion). Verified here against
        ``scipy.special.erfc`` up to x=30 (max relative error ~1.05e-7,
        consistent with the cited bound).

        Kept in log space throughout — never forms ``erfc(x)`` or
        ``exp(-x^2)`` directly — because for the ``|z| > 5`` tails this
        is meant to serve, ``exp(-x^2)`` itself underflows to 0.0 in
        float64 (around ``x > ~26``) long before the true p-value stops
        mattering for ranking hits.
        """
        t = 1.0 / (1.0 + 0.5 * x)
        poly = -1.26551223 + t * (
            1.00002368
            + t
            * (
                0.37409196
                + t
                * (
                    0.09678418
                    + t
                    * (
                        -0.18628806
                        + t
                        * (
                            0.27886807
                            + t
                            * (
                                -1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))
                            )
                        )
                    )
                )
            )
        )
        return t.log() + (-x.pow(2) + poly)

    @staticmethod
    def _standard_normal_cdf_expr(z: pl.Expr) -> pl.Expr:
        """
        Rational approximation to ``Phi(z)``, built from
        :meth:`_log_erfc_expr` (see there for the citation and peak error
        bound of ~1.2e-7). Verified against ``scipy.stats.norm.cdf``
        across randomized z values, including tail values (``|z| > 5``),
        before being trusted downstream. Provided for standalone
        testing/verification — :meth:`_neg_log10_two_sided_normal_pvalue_expr`
        does not call this; it stays in log space end to end instead.
        """
        a = z.abs() / math.sqrt(2.0)
        log_erfc_a = AUROCNegLogPValueAggregator._log_erfc_expr(a)
        erfc_a = log_erfc_a.exp()
        erfc_z_over_sqrt2 = pl.when(z >= 0).then(erfc_a).otherwise(2.0 - erfc_a)
        return 1.0 - 0.5 * erfc_z_over_sqrt2

    @staticmethod
    def _neg_log10_two_sided_normal_pvalue_expr(z: pl.Expr) -> pl.Expr:
        """
        ``-log10(2 * (1 - Phi(|z|))) == -log10(erfc(|z| / sqrt(2)))``,
        computed by staying in log space via :meth:`_log_erfc_expr` end
        to end — never forming ``Phi(z)`` or ``1 - Phi(z)`` in linear
        space, which is exactly what underflows to 0.0 (-> null/-inf) for
        the large-``|z|`` hits this aggregator ranks by most.
        """
        a = z.abs() / math.sqrt(2.0)
        log_erfc_a = AUROCNegLogPValueAggregator._log_erfc_expr(a)
        neg_log10_p = -log_erfc_a / math.log(10.0)
        # -log10(p) can never legitimately be negative (p <= 1 always);
        # this guards float noise near z~0 (p slightly > 1).
        return neg_log10_p.clip(lower_bound=0.0)

    def _feature_expr(self, feat: str) -> pl.Expr:
        alias = f"{feat}{self._stat_suffix}"
        u = pl.col(self._u_col(feat))
        n_group = pl.col(self._ngroup_col(feat))
        n_ref = pl.col(self._nref_col(feat))
        tie_term = pl.col(self._tieterm_col(feat))

        n = n_group + n_ref
        mu_u = n_group * n_ref / 2.0
        sigma2_u = (n_group * n_ref / 12.0) * ((n + 1) - tie_term / (n * (n - 1)))
        z = (u - mu_u) / sigma2_u.sqrt()
        neg_log10_p = self._neg_log10_two_sided_normal_pvalue_expr(z)

        result = (
            pl.when(n_ref == 0)
            .then(None)
            .when(n_group == 0)
            .then(None)
            .otherwise(neg_log10_p)
        )
        return result.alias(alias)


def downsample_control(
    lf: pl.LazyFrame, downsample_wt: Union[float, int], seed: int
) -> pl.LazyFrame:
    """
    Reproducibly downsample control (wildtype) rows to a target size.

    Non-control rows are left untouched. Selection is deterministic given
    ``seed``: each control row gets a seeded hash of a fresh row index, rows
    are ranked by that hash, and the lowest ``target`` ranks are kept — the
    same lazy hash-and-rank idiom used by
    :func:`fisseq_common.stages.qcfilter.add_downsampled_pseudo_variants`, which
    avoids collecting the full control pool up front and reproduces exactly
    across runs for a fixed seed.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame containing a boolean ``CONTROL_COLUMN`` column.
    downsample_wt : float or int
        Target control-pool size. A float in ``(0, 1)`` is interpreted as
        the fraction of control rows to keep; an int is interpreted as an
        absolute target count (a no-op if the control pool is already at or
        below the target).
    seed : int
        Random seed for the downsample draw.

    Returns
    -------
    pl.LazyFrame
        ``lf`` with control rows downsampled to the target size.
    """
    control = lf.filter(CONTROL_COLUMN).with_row_index("_tmp_row_idx")
    non_control = lf.filter(~CONTROL_COLUMN)

    ranked = control.with_columns(
        pl.col("_tmp_row_idx").hash(seed=seed).rank(method="ordinal").alias("_rank"),
    )
    if isinstance(downsample_wt, float):
        target = (pl.len() * downsample_wt).floor()
    else:
        target = pl.lit(downsample_wt)

    downsampled_control = ranked.filter(pl.col("_rank") <= target).drop(
        ["_tmp_row_idx", "_rank"]
    )
    return pl.concat([non_control, downsampled_control])


_AGGREGATORS: dict[str, type[BaseAggregator]] = {
    "mean": MeanAggregator,
    "median": MedianAggregator,
    "MAD": MADAggregator,
    "std": StdAggregator,
    "KS": KSAggregator,
    "signedKS": SignedKSAggregator,
    "QQ": QQCorrelationAggregator,
    "AUROC": AUROCAggregator,
    "KSnegLogP": KSNegLogPValueAggregator,
    "AUROCnegLogP": AUROCNegLogPValueAggregator,
}


def aggregate(
    lf: pl.LazyFrame,
    label_col: str,
    aggregator_name: str,
    feature_selector: pl.Expr = FEATURE_SELECTOR,
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE,
) -> pl.LazyFrame:
    """
    Run the specified aggregator on cell-level data and return per-label statistics.

    Control rows (where ``CONTROL_COLUMN`` is ``True``) are passed as the
    reference distribution to all aggregators that require one.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level LazyFrame. Must contain a boolean ``CONTROL_COLUMN`` column.
    label_col : str
        Name of the column identifying variant labels.
    aggregator_name : str
        Aggregation method. One of: ``mean``, ``median``, ``MAD``, ``std``,
        ``KS``, ``signedKS``, ``QQ``, ``AUROC``, ``KSnegLogP``,
        ``AUROCnegLogP``.
    feature_selector : pl.Expr
        Polars selector identifying feature columns. Defaults to ``FEATURE_SELECTOR``.
    feature_chunk_size : int or None
        Number of feature columns evaluated per Polars query. Lower it if a
        task is OOM-killed; runtime is essentially flat in this value for the
        expensive aggregators, so it is a memory dial rather than a
        speed/memory trade-off. ``None`` disables chunking entirely (every
        feature in one query), which is the shape that OOM-killed production
        -- use it only for small inputs. Defaults to
        :data:`DEFAULT_FEATURE_CHUNK_SIZE`.

    Returns
    -------
    pl.LazyFrame
        One row per non-control variant group with computed statistics,
        sorted by ``label_col``.
    """
    valid = set(_AGGREGATORS)
    if aggregator_name not in valid:
        raise ValueError(
            f"Unknown aggregator {aggregator_name!r}. Choose from: {sorted(valid)}"
        )

    agg = _AGGREGATORS[aggregator_name](
        label_col=label_col,
        feature_selector=feature_selector,
        feature_chunk_size=feature_chunk_size,
    )
    return agg.aggregate(lf)


def aggregate_methods(
    filtered_lf: pl.LazyFrame,
    label_column: str,
    aggregators: Sequence[str] = ("median",),
    feature_selector: pl.Expr = FEATURE_SELECTOR,
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE,
    include_metadata: bool = True,
) -> pl.DataFrame:
    """
    Aggregate normalized cells per variant via one or more methods.

    Runs each requested aggregator (see :data:`_AGGREGATORS`) against
    ``filtered_lf`` and joins their outputs together on ``label_column``,
    then joins in :func:`fisseq_common.utils.metadata.get_aggregate_meta_data`
    for `meta_num_cells`/`meta_barcode_num_unique`/etc. The control/WT-label
    metadata row is naturally dropped by this join -- no aggregator ever
    produces a control-row group to join against, so the inner join between
    aggregator output and metadata output already excludes it without any
    separate filtering step.

    Parameters
    ----------
    filtered_lf : pl.LazyFrame
        QC-passed, normalized cell-level features, as rebuilt by
        :func:`fisseq_common.stages.filter.load_filtered_cells` -- carries a boolean
        ``CONTROL_COLUMN`` column, the feature columns, and ``label_column``.
    label_column : str
        Name of the column identifying variant labels.
    aggregators : Sequence[str]
        Aggregation method(s) to run, in the given order: keys of :data:`_AGGREGATORS`.
        Defaults to ``("median",)``.
    feature_selector : pl.Expr
        Polars selector identifying feature columns, forwarded to each
        aggregator. Defaults to ``FEATURE_SELECTOR`` (every non-``meta_``
        column); the embeddings pipeline passes ``EMBEDDING_SELECTOR``.
    feature_chunk_size : int or None
        Number of feature columns each aggregator evaluates per Polars
        query -- a pure memory dial with no effect on the output. ``None``
        disables chunking (every dimension in one query). Defaults to
        :data:`DEFAULT_FEATURE_CHUNK_SIZE`.
    include_metadata : bool
        When ``True`` (the default), join in
        :func:`~fisseq_common.utils.metadata.get_aggregate_meta_data`'s
        ``meta_num_cells``/``meta_barcode_num_unique``/... columns. When
        ``False``, return the lean ``[label_column] + stat columns`` frame
        only -- what AGGREGATE_HALF and AGGREGATE_PASSTHROUGH want, since
        those feed a per-column correlation or a join and would otherwise
        each carry a redundant (and, for the halves, *wrong*) copy of the
        per-variant cell counts.

    Returns
    -------
    pl.DataFrame
        One row per non-control variant group, sorted by ``label_column``. Each feature
        column is suffixed by its aggregator's ``_stat_suffix`` (e.g. ``f_mean``, ``f_KS``).

    Raises
    ------
    ValueError
        If ``aggregators`` is empty, contains a name not in
        :data:`_AGGREGATORS`, or contains a duplicate name.
    """
    aggregators = tuple(aggregators)
    if not aggregators:
        raise ValueError("aggregators must not be empty")

    unknown = sorted(set(aggregators) - set(_AGGREGATORS))
    if unknown:
        raise ValueError(
            f"Unknown aggregator(s) {unknown}. Choose from: {sorted(_AGGREGATORS)}"
        )

    duplicates = sorted(
        name for name, count in Counter(aggregators).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate aggregator(s) requested: {duplicates}")

    result_lf: Optional[pl.LazyFrame] = None
    for name in aggregators:
        agg_lf = _AGGREGATORS[name](
            label_col=label_column,
            feature_selector=feature_selector,
            feature_chunk_size=feature_chunk_size,
        ).aggregate(filtered_lf)
        result_lf = (
            agg_lf
            if result_lf is None
            else result_lf.join(agg_lf, on=label_column, how="inner")
        )

    # Sorted, not merely collected. Each aggregator's own output is already
    # sorted (BaseAggregator.aggregate), but a join is not order-preserving
    # under Polars' multithreaded execution, so the multi-method join above
    # and the metadata join below can both reshuffle rows. Without this the
    # same input produces the same numbers in a different order run to run,
    # which makes aggregate.parquet non-byte-reproducible and defeats
    # `--rerun-triggers`-style change detection downstream.
    if not include_metadata:
        return result_lf.sort(label_column).collect()

    meta_lf = get_aggregate_meta_data(filtered_lf, label_column)
    result_lf = result_lf.join(meta_lf, on=label_column, how="inner")
    return result_lf.sort(label_column).collect()


def zscore_to_synonymous(agg_lf: pl.LazyFrame, label_column: str) -> pl.LazyFrame:
    """
    Z-score every aggregate column against the experiment's synonymous variants.

    Fits a :class:`~fisseq_common.normalizer.Normalizer` on the untagged synonymous rows
    (:func:`~fisseq_common.stages.filter.variant_classification`, ``ddof=1``) and applies it to
    every row. Needs at least two synonymous variants; with one, every column comes out null.
    The data pipeline runs it on its per-variant aggregates so they are comparable across
    experiments; it requires that the synonymous variants survive aggregation, i.e. that they
    are not the cell-level controls.
    """
    agg_lf = variant_classification(agg_lf, label_column)
    normalizer = Normalizer.from_lazyframe(agg_lf, fit_only_on_control=True)
    return normalizer.apply(agg_lf).drop(CONTROL_COLUMN_NAME)


def aggregate_cells(
    cells_lf: pl.LazyFrame,
    label_column: str,
    aggregator: str,
    *,
    join_keys: Sequence[str],
    split_file: Optional[str] = None,
    feature_selector: pl.Expr = FEATURE_SELECTOR,
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE,
    downsample_controls: Optional[Union[float, int]] = None,
    seed: int = 0,
    normalize_to_synonymous: bool = False,
) -> pl.DataFrame:
    """
    Aggregate one method over an experiment's cells, or over one pseudo-replicate half.

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        QC-passed, normalized cells with ``meta_is_control``.
    label_column : str
        Variant label column.
    aggregator : str
        A key of :data:`_AGGREGATORS`.
    join_keys : sequence of str
        The columns a split file names a row by (:func:`~.config.row_keys`).
    split_file : str, optional
        A GENERATE_SPLIT half; ``None`` aggregates every cell.
    feature_selector : pl.Expr
        Which columns are features.
    feature_chunk_size : int or None
        See :data:`DEFAULT_FEATURE_CHUNK_SIZE`.
    downsample_controls : float or int, optional
        Downsample the control rows before aggregating (:func:`downsample_control`): a
        float in ``(0, 1)`` keeps that fraction, an int that many. ``None`` keeps them all.
    seed : int
        Seed for ``downsample_controls``.
    normalize_to_synonymous : bool
        Z-score the result against the synonymous variants (:func:`zscore_to_synonymous`).

    Returns
    -------
    pl.DataFrame
        ``[label_column] + <stat columns>``, one row per non-control variant, sorted by label.
    """
    lf = filter_by_split_file(cells_lf, split_file, list(join_keys))
    if downsample_controls is not None:
        if isinstance(downsample_controls, float) and not 0 < downsample_controls < 1:
            raise ValueError(
                f"downsample_wt float must satisfy 0 < x < 1, got {downsample_controls}"
            )
        if isinstance(downsample_controls, int) and downsample_controls <= 0:
            raise ValueError(
                f"downsample_wt int must be positive, got {downsample_controls}"
            )
        logging.info(
            "Downsampling control rows: %s, seed=%d", downsample_controls, seed
        )
        lf = downsample_control(lf, downsample_controls, seed)
    logging.info(
        "Running %s aggregator (feature_chunk_size=%s)", aggregator, feature_chunk_size
    )
    agg_df = aggregate_methods(
        lf,
        label_column,
        [aggregator],
        feature_selector=feature_selector,
        feature_chunk_size=feature_chunk_size,
        include_metadata=False,
    )
    if normalize_to_synonymous:
        logging.info("Z-scoring aggregates against synonymous variants")
        agg_df = zscore_to_synonymous(agg_df.lazy(), label_column).collect()
    return agg_df


@dataclasses.dataclass
class AggregateConfig(CellsInput, AppConfig):
    """
    The aggregation stage's configuration: one method over the normalized cells
    (:class:`~.config.CellsInput`), or over one GENERATE_SPLIT half.

    Attributes
    ----------
    label_column : str
        Variant label column. Defaults to ``"meta_aa_changes"``.
    aggregator : str
        A key of :data:`_AGGREGATORS`. Required.
    split_file : str, optional
        A GENERATE_SPLIT half; ``None`` aggregates every cell. Defaults to ``None``.
    downsample_wt : float or int, optional
        Downsample the control (wildtype) rows first (:func:`downsample_control`), seeded with
        ``random_seed``. AGGREGATE_HALF passes ``random_seed + bootstrap_idx * 2 + half_num``,
        so every half of every replicate draws its own subsample. ``None`` keeps them all.
    feature_chunk_size : int or None
        See :data:`DEFAULT_FEATURE_CHUNK_SIZE`.
    normalize_to_synonymous : bool
        Z-score the result against the synonymous variants (:func:`zscore_to_synonymous`); needs
        at least two synonymous variants. Defaults to ``False``.
    output_name : str
        The output is ``{output_name}.parquet`` (``{output_root}.``-prefixed when that is set).
        Defaults to ``"aggregate"``.
    """

    label_column: str = "meta_aa_changes"
    aggregator: str = MISSING
    split_file: Optional[str] = None
    downsample_wt: Optional[Union[float, int]] = None
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE
    normalize_to_synonymous: bool = False
    output_name: str = "aggregate"


def run_aggregate(cfg: AggregateConfig) -> None:
    """Aggregate the cells ``cfg`` names with :func:`aggregate_cells` and write the lean
    ``[label_column] + <stat columns>`` table."""
    agg_df = aggregate_cells(
        load_cells(cfg),
        cfg.label_column,
        cfg.aggregator,
        join_keys=row_keys(cfg.join_keys),
        split_file=cfg.split_file,
        feature_selector=feature_selector(cfg.feature_selector),
        feature_chunk_size=cfg.feature_chunk_size,
        downsample_controls=cfg.downsample_wt,
        seed=cfg.random_seed,
        normalize_to_synonymous=cfg.normalize_to_synonymous,
    )
    prefix = f"{cfg.output_root}." if cfg.output_root is not None else ""
    out_path = pathlib.Path(cfg.output_dir) / f"{prefix}{cfg.output_name}.parquet"
    logging.info("Writing %s (%d row(s))", out_path, agg_df.height)
    agg_df.write_parquet(out_path)
    logging.info("Done")


main = stage_main("aggregate_main", AggregateConfig, run_aggregate)

if __name__ == "__main__":
    main()
