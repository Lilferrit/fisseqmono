"""Variant-vs-wild-type (OvWT) distinguishability scores from the pipeline."""

import pathlib
from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal, Self

import polars as pl

from . import _pipeline, _variants
from .dataset import Dataset

#: Score columns in ``ovwt_batchwise/<batch>/results.parquet``.
SCORES: tuple[str, ...] = ("auroc_pooled", "auroc_median_barcode", "auroc_median_fold")


class OvwtScores(Dataset):
    """OvWT classifier AUROCs: how well each variant's cells separate from wild type.

    `from_pipeline` gives one row per (variant, batch). Correct the batches onto a common
    scale with `correct`, then collapse to one row per variant with `per_variant`, or pass
    the per-batch scores straight to `Profiles.distinguishability`, which does both:

    >>> scores = fb.OvwtScores.from_pipeline(run_dir)
    >>> (scores.variant_type()
    ...        .correct("auroc_pooled")
    ...        .per_variant("auroc_pooled_corrected"))
    >>> profiles.distinguishability(scores)

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame | Dataset
        The scores.
    variant_col : str, default "meta_aa_changes"
        Variant label column.
    batch_col : str, default "meta_experiment"
        Batch column, while there is one row per (variant, batch).
    """

    def __init__(
        self,
        data: "pl.DataFrame | pl.LazyFrame | Dataset",
        *,
        variant_col: str = "meta_aa_changes",
        batch_col: str = "meta_experiment",
    ) -> None:
        super().__init__(data, variant_col=variant_col)
        self.batch_col = batch_col

    @classmethod
    def from_pipeline(
        cls,
        pipeline_dir: str | PathLike,
        *,
        batches: Sequence[str] | None = None,
        variant_col: str = "meta_aa_changes",
        batch_col: str = "meta_experiment",
    ) -> Self:
        """Read ``ovwt_batchwise/<batch>/results.parquet`` for every batch (or
        ``batches``), tagging each row with its batch in ``batch_col``.

        Columns are those the pipeline writes: ``auroc_pooled``,
        ``auroc_median_barcode``, ``auroc_median_fold``, the per-fold ``auroc_folds``
        list, ``meta_n_barcodes`` and ``meta_n_cells``.
        """
        frames = [
            _pipeline.tag(_pipeline.scan(d / "results.parquet"), d.name, batch_col)
            for d in _pipeline.batch_dirs(pipeline_dir, _pipeline.OVWT, batches)
        ]
        return cls(
            pl.concat(frames, how="diagonal_relaxed"), variant_col=variant_col, batch_col=batch_col
        )

    @classmethod
    def from_global(cls, pipeline_dir: str | PathLike, channel: str, **kw: Any) -> Self:
        """Read the pipeline's cross-experiment scores,
        ``global/<channel>/ovwt_distinguishability/global_scores.parquet``.

        One row per variant with ``meta_median_auroc_pooled``,
        ``meta_median_auroc_median_barcode`` and ``meta_median_auroc_median_fold`` (each
        experiment z-scored against its synonymous variants, then the median taken) and
        ``meta_num_experiments``.
        """
        path = (
            pathlib.Path(pipeline_dir)
            / "global"
            / channel
            / "ovwt_distinguishability"
            / "global_scores.parquet"
        )
        return cls(_pipeline.scan(path), **kw)

    def correct(
        self,
        score: str = "auroc_pooled",
        *,
        reference: Literal["synonymous", "all"] = "synonymous",
        output_col: str | None = None,
    ) -> Self:
        """Put every batch's ``score`` on a common scale.

        Each batch is shifted and scaled so that the mean and standard deviation of its
        reference variants (its synonymous controls, or ``"all"`` of its variants) equal
        the median of those statistics across batches. Scores stay on the AUROC scale, so
        0.5 still marks chance. The result goes in ``output_col`` (default
        ``<score>_corrected``).
        """
        self._require(self.variant_col, self.batch_col, score)
        if reference == "synonymous":
            is_reference = _variants.control_expr(
                _variants.variant_type_expr(self.variant_col), self.variant_col
            )
        elif reference == "all":
            is_reference = pl.lit(True)
        else:
            raise ValueError(f"reference must be 'synonymous' or 'all', got {reference!r}")
        output_col = output_col or f"{score}_corrected"
        value = pl.col(score).cast(pl.Float64)

        stats = self._lf.group_by(self.batch_col).agg(
            value.filter(is_reference).mean().alias("_batch_mean"),
            value.filter(is_reference).std().alias("_batch_std"),
        )
        target = stats.select(
            pl.col("_batch_mean").median().alias("_target_mean"),
            pl.col("_batch_std").median().alias("_target_std"),
        )
        corrected = (value - pl.col("_batch_mean")) / pl.col("_batch_std") * pl.col(
            "_target_std"
        ) + pl.col("_target_mean")
        return self._replace(
            self._lf.join(stats, on=self.batch_col, how="left", maintain_order="left")
            .join(target, how="cross", maintain_order="left")
            .with_columns(corrected.alias(output_col))
            .drop("_batch_mean", "_batch_std", "_target_mean", "_target_std")
        )

    def per_variant(
        self,
        score: str = "auroc_pooled",
        *,
        agg: Literal["median", "mean"] = "median",
        n_col: str = "meta_n_experiments",
    ) -> Self:
        """Collapse to one row per variant: the ``agg`` of ``score`` across batches.

        Also counts the batches with a score (``n_col``), sums ``meta_n_cells`` and
        ``meta_n_barcodes`` when present, and keeps the first value of any other ``meta_``
        column. Other score columns are dropped.
        """
        self._require(self.variant_col, self.batch_col, score)
        value = pl.col(score)
        summed = [c for c in ("meta_n_cells", "meta_n_barcodes") if c in self.columns]
        first = [
            c for c in self.meta if c not in {self.variant_col, self.batch_col, n_col, *summed}
        ]
        return self._replace(
            self._lf.group_by(self.variant_col, maintain_order=True).agg(
                *[pl.col(c).first() for c in first],
                *[pl.col(c).sum() for c in summed],
                value.count().cast(pl.UInt32).alias(n_col),
                value.median() if agg == "median" else value.mean(),
            )
        )
