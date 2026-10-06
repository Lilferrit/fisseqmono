"""Variant-vs-wild-type (OvWT) distinguishability scores from the pipeline."""

import logging
import pathlib
import warnings
from collections.abc import Sequence
from os import PathLike
from typing import Any, Literal, Self

import polars as pl

from . import _pipeline, _transforms, _variants
from .dataset import Dataset

logger = logging.getLogger(__name__)

#: Score columns in ``ovwt_batchwise/<batch>/results.parquet``. ``auroc_folds`` (the
#: per-fold list) is not a score column.
SCORES: tuple[str, ...] = ("auroc_pooled", "auroc_median_barcode", "auroc_median_fold")

#: Renames applied to the older ``results.parquet`` schema (``variant``, ``test_auroc``, ...).
#: The variant column is renamed to the ``variant_col`` passed to `OvwtScores.from_pipeline`.
#: Score columns (``train_auroc``, ``val_auroc``, ``test_auroc`` and the accuracies) keep
#: their names, so pass e.g. ``score="test_auroc"``.
LEGACY_RENAMES: dict[str, str] = {
    "meta_num_cells": "meta_n_cells",
    "meta_barcode_num_unique": "meta_n_barcodes",
}
_LEGACY_VARIANT_COL = "variant"


def _read_results(path: pathlib.Path, variant_col: str) -> pl.LazyFrame:
    """Scan one batch's results, mapping the legacy schema onto the current names."""
    lf = _pipeline.scan(path)
    names = lf.collect_schema().names()
    if _LEGACY_VARIANT_COL in names and variant_col not in names:
        logger.info("Reading %s with the legacy OvWT results schema", path)
        renames = {_LEGACY_VARIANT_COL: variant_col}
        renames |= {old: new for old, new in LEGACY_RENAMES.items() if old in names}
        lf = lf.rename(renames)
    return lf


class OvwtScores(Dataset):
    """OvWT classifier AUROCs: how well each variant's cells separate from wild type.

    `from_pipeline` gives one row per (variant, batch). Correct the batches onto a common
    scale with `correct`, then collapse to one row per variant with `per_variant`, or pass
    the per-batch scores straight to `Profiles.distinguishability`, which does both:

    >>> scores = fb.OvwtScores.from_pipeline(run_dir)
    >>> scores.correct("auroc_pooled").per_variant("auroc_pooled_corrected")
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
        pipeline_dir: "str | PathLike | _pipeline.Source",
        *,
        batches: Sequence[str] | None = None,
        exclude: "_pipeline.Patterns | None" = None,
        download_dir: str | PathLike | None = None,
        refresh: bool = False,
        layout: "_pipeline.LayoutSpec" = None,
        track: "_pipeline.Track" = "embeddings",
        variant_col: str = "meta_aa_changes",
        batch_col: str = "meta_experiment",
    ) -> Self:
        """Read each batch's OvWT ``results.parquet`` (``ovwt_batchwise/<batch>/``; the
        embeddings pipeline's CellProfiler track: ``ovwt_batchwise_cp_features/<batch>/``)
        for every batch (or ``batches``, minus those matching ``exclude``), tagging each
        row with its batch in ``batch_col``. A remote ``pipeline_dir``
        (``"user@host:/path"``, with ``download_dir`` and ``refresh``), ``layout`` and
        ``track`` are handled as in `Profiles.from_pipeline`.

        Columns are those the pipeline writes: ``auroc_pooled``,
        ``auroc_median_barcode``, ``auroc_median_fold``, the per-fold ``auroc_folds``
        list, ``meta_n_barcodes`` and ``meta_n_cells``.

        Older runs (e.g. 2026-08-10) wrote ``variant``, ``train_auroc`` / ``val_auroc`` /
        ``test_auroc``, ``meta_num_cells`` and ``meta_barcode_num_unique``. They are
        detected per batch: ``variant`` becomes ``variant_col`` and the counts are renamed
        by `LEGACY_RENAMES`, while the score columns keep their names, e.g.
        ``profiles.distinguishability(ovwt, score="test_auroc")``.
        """
        src = _pipeline.source(pipeline_dir, download_dir, refresh, layout, track)
        names = src.batches("ovwt", batches, exclude)
        paths = src.files([src.layout.ovwt_results(b) for b in names])
        frames = [
            _pipeline.tag(_read_results(path, variant_col), b, batch_col)
            for b, path in zip(names, paths)
        ]
        return cls(
            pl.concat(frames, how="diagonal_relaxed"),
            variant_col=variant_col,
            batch_col=batch_col,
        )

    @classmethod
    def from_global(cls, pipeline_dir: str | PathLike, channel: str, **kw: Any) -> Self:
        """Read ``global/<channel>/ovwt_distinguishability/global_scores.parquet`` from
        an older pipeline run.

        One row per variant with ``meta_median_auroc_pooled``,
        ``meta_median_auroc_median_barcode`` and ``meta_median_auroc_median_fold`` (each
        experiment z-scored against its synonymous variants, then the median taken) and
        ``meta_num_experiments``.

        .. deprecated::
            The pipeline no longer writes ``global/``. The ``fisseqborn-global`` command
            (or `fisseqborn.write_global`) writes the same table to
            ``<out>/ovwt_distinguishability/global_scores.parquet``; read it with
            `OvwtScores.read`. It is ``correct(rescale=False)`` followed by `per_variant`.
        """
        warnings.warn(
            "OvwtScores.from_global reads the pipeline's global/ directory, which it no "
            "longer writes. Run `fisseqborn-global <pipeline_dir> --out <dir>` (or "
            "fisseqborn.write_global) and use "
            "OvwtScores.read('<dir>/ovwt_distinguishability/global_scores.parquet'), or "
            "compute it with .correct(rescale=False).per_variant(...).",
            DeprecationWarning,
            stacklevel=2,
        )
        path = (
            pathlib.Path(pipeline_dir)
            / "global"
            / channel
            / "ovwt_distinguishability"
            / "global_scores.parquet"
        )
        return cls(_pipeline.scan(path), **kw)

    def _score_columns(self, score: str | Sequence[str] | None) -> list[str]:
        """``score`` as a list of numeric score columns; ``None`` means every column of
        `SCORES` present."""
        if score is None:
            scores = [s for s in SCORES if s in self.columns]
            if not scores:
                raise ValueError(
                    f"None of the score columns {list(SCORES)} found; pass score= "
                    "(e.g. score='test_auroc' for legacy results)"
                )
        else:
            scores = [score] if isinstance(score, str) else list(score)
            if not scores:
                raise ValueError("Pass at least one score column")
        self._require(*scores)
        schema = self.schema
        bad = [s for s in scores if not schema[s].is_numeric()]
        if bad:
            raise ValueError(
                f"{bad} are not numeric score columns (auroc_folds holds the per-fold "
                f"AUROC lists, use auroc_median_fold); choose from {list(SCORES)}"
            )
        return scores

    def correct(
        self,
        score: str | Sequence[str] | None = None,
        *,
        reference: Literal["synonymous", "all"] = "synonymous",
        rescale: bool = True,
        output_col: str | None = None,
    ) -> Self:
        """Put every batch's scores on a common scale.

        ``score`` is a score column or a list of them (default: every column of `SCORES`
        present). Each is corrected separately into ``<score>_corrected`` (or
        ``output_col``, for a single score), using each batch's reference variants: its
        synonymous controls, or ``"all"`` of its variants.

        With ``rescale=True`` (the default) each batch is shifted and scaled so that the
        mean and standard deviation of its reference scores equal the median of those
        statistics across batches. Scores stay on the AUROC scale, so 0.5 still marks
        chance. With ``rescale=False`` the result is the plain z-score
        ``(x - mean) / std`` of the reference scores, as the old pipeline's global OvWT
        stage computed it.

        The standard deviation uses ``ddof=1``; when it is below float32 epsilon (or a
        batch has fewer than two reference scores, which is logged as a warning) every
        corrected score of that batch is null.
        """
        scores = self._score_columns(score)
        if output_col is not None and len(scores) != 1:
            raise ValueError(
                "output_col needs a single score; the others get '_corrected'"
            )
        self._require(self.variant_col, self.batch_col)
        if reference == "synonymous":
            is_reference = _variants.control_expr(
                _variants.variant_type_expr(self.variant_col), self.variant_col
            )
        elif reference == "all":
            is_reference = pl.lit(True)
        else:
            raise ValueError(
                f"reference must be 'synonymous' or 'all', got {reference!r}"
            )
        outputs = (
            [output_col]
            if output_col is not None
            else [f"{s}_corrected" for s in scores]
        )

        def value(s: str) -> pl.Expr:
            return pl.col(s).cast(pl.Float64).fill_nan(None)

        stats = (
            self._lf.group_by(self.batch_col, maintain_order=True)
            .agg(
                agg
                for i, s in enumerate(scores)
                for agg in (
                    value(s).filter(is_reference).count().alias(f"_n_{i}"),
                    value(s).filter(is_reference).mean().alias(f"_mean_{i}"),
                    value(s).filter(is_reference).std().alias(f"_std_{i}"),
                )
            )
            .collect()
        )
        for row in stats.iter_rows(named=True):
            few = [s for i, s in enumerate(scores) if row[f"_n_{i}"] < 2]
            if few:
                logger.warning(
                    "Experiment %s has fewer than 2 %s variants with a score for %s; its "
                    "corrected scores are all null",
                    row[self.batch_col],
                    "synonymous control" if reference == "synonymous" else "reference",
                    ", ".join(few),
                )
        stats = stats.with_columns(
            pl.when(pl.col(f"_std_{i}") >= _transforms.EPS).then(pl.col(f"_std_{i}"))
            for i in range(len(scores))
        )
        exprs = []
        for i, (s, out) in enumerate(zip(scores, outputs)):
            z = (value(s) - pl.col(f"_mean_{i}")) / pl.col(f"_std_{i}")
            if rescale:
                target_std = pl.lit(stats[f"_std_{i}"].median(), dtype=pl.Float64)
                target_mean = pl.lit(stats[f"_mean_{i}"].median(), dtype=pl.Float64)
                z = z * target_std + target_mean
            exprs.append(z.alias(out))
        return self._replace(
            self._lf.join(
                stats.lazy(), on=self.batch_col, how="left", maintain_order="left"
            )
            .with_columns(exprs)
            .drop(stats.columns[1:])
        )

    def per_variant(
        self,
        score: str | Sequence[str] = "auroc_pooled",
        *,
        agg: Literal["median", "mean"] = "median",
        n_col: str = "meta_n_experiments",
    ) -> Self:
        """Collapse to one row per variant: the ``agg`` of each ``score`` (a column or a
        list of them) across batches.

        Also counts the batches with a score (``n_col``; with several scores, the batches
        where any of them is present), sums ``meta_n_cells`` and ``meta_n_barcodes`` when
        present, and keeps the first value of any other ``meta_`` column. Other score
        columns, and ``auroc_folds``, are dropped.
        """
        scores = self._score_columns(score)
        self._require(self.variant_col, self.batch_col)
        summed = [c for c in ("meta_n_cells", "meta_n_barcodes") if c in self.columns]
        first = [
            c
            for c in self.meta
            if c not in {self.variant_col, self.batch_col, n_col, *summed}
        ]
        present = pl.any_horizontal([pl.col(s).is_not_null() for s in scores])
        return self._replace(
            self._lf.group_by(self.variant_col, maintain_order=True).agg(
                *[pl.col(c).first() for c in first],
                *[pl.col(c).sum() for c in summed],
                present.sum().cast(pl.UInt32).alias(n_col),
                *[
                    pl.col(s).median() if agg == "median" else pl.col(s).mean()
                    for s in scores
                ],
            )
        )
