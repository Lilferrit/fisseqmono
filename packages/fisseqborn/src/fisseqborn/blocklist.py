"""Per-batch feature blocklists from the pipeline's feature selection."""

import posixpath
from collections.abc import Sequence
from os import PathLike
from typing import Literal, Self

import polars as pl

from . import _pipeline
from .dataset import Dataset


class Blocklists(Dataset):
    """Which features passed each batch's replicate-reproducibility filter.

    One row per (feature, statistic, batch) with ``feature`` (the full column name, e.g.
    ``AreaShape_Area_median``), ``median_r`` (the median split-half correlation),
    ``feature_ok``, ``meta_experiment`` and ``meta_feature_type``. Use `consensus` to get
    the features that pass everywhere, and pass them to `Profiles.keep_features`, or
    `table` for the per-feature counts behind it:

    >>> ok = fb.Blocklists.from_pipeline(run_dir).rethreshold(0.7).consensus()
    >>> profiles.keep_features(ok)

    Parameters
    ----------
    data : pl.DataFrame | pl.LazyFrame | Dataset
        The blocklist rows.
    batch_col : str, default "meta_experiment"
        Batch column.
    """

    def __init__(
        self,
        data: "pl.DataFrame | pl.LazyFrame | Dataset",
        *,
        batch_col: str = "meta_experiment",
    ) -> None:
        super().__init__(data, variant_col="feature")
        self.batch_col = batch_col

    @classmethod
    def from_pipeline(
        cls,
        pipeline_dir: "str | PathLike | _pipeline.Source",
        types: Sequence[str] | None = None,
        *,
        batches: Sequence[str] | None = None,
        exclude: "_pipeline.Patterns | None" = None,
        download_dir: str | PathLike | None = None,
        refresh: bool = False,
        layout: "_pipeline.LayoutSpec" = None,
        track: "_pipeline.Track" = "embeddings",
        batch_col: str = "meta_experiment",
    ) -> Self:
        """Read each batch's per-method blocklists
        (``feature_select_batchwise/<batch>/blocklists/<type>.parquet``) for every batch (or
        ``batches``, minus those matching ``exclude``) and statistic type (default: every
        file present). A remote ``pipeline_dir``, ``layout`` and ``track`` are handled as in
        `Profiles.from_pipeline`: with ``types=None`` the blocklists are listed with one
        ssh call, and only they are downloaded. The embeddings pipeline's CellProfiler
        track has no blocklists."""
        src = _pipeline.source(pipeline_dir, download_dir, refresh, layout, track)
        lay = src.layout
        names = src.batches("feature_select", batches, exclude)
        if types is not None:
            pairs = [(b, lay.method_blocklist(b, t)) for b in names for t in types]
        else:
            patterns = [lay.method_blocklist(b, "*") for b in names]
            if None in patterns:
                raise ValueError(f"{lay!r} writes no blocklists")
            found = src.glob(patterns)
            pairs = []
            for b, pattern in zip(names, patterns):
                directory = posixpath.dirname(pattern)
                mine = sorted(r for r in found if posixpath.dirname(r) == directory)
                if not mine:
                    raise FileNotFoundError(f"No blocklists in {src}/{directory}")
                pairs += [(b, r) for r in mine]
        if any(r is None for _b, r in pairs):
            raise ValueError(f"{lay!r} writes no blocklists")
        frames = [
            _pipeline.tag(_pipeline.scan(path), b, batch_col).with_columns(
                pl.lit(path.stem, dtype=pl.String).alias("meta_feature_type")
            )
            for (b, _rel), path in zip(pairs, src.files([r for _b, r in pairs]))
        ]
        return cls(pl.concat(frames, how="diagonal_relaxed"), batch_col=batch_col)

    def rethreshold(self, min_r: float) -> Self:
        """Recompute ``feature_ok`` as ``median_r >= min_r`` (the pipeline's
        ``min_correlation`` rule). A null ``median_r`` is not OK."""
        self._require("median_r")
        return self.with_columns(
            (pl.col("median_r") >= min_r).fill_null(False).alias("feature_ok")
        )

    def table(
        self,
        min_batches: int | None = None,
        *,
        missing: Literal["fail", "ignore"] = "fail",
    ) -> pl.DataFrame:
        """One row per feature: ``feature``, ``n_batches`` (the batches whose blocklist
        reports it), ``n_ok`` (those where it is OK) and the consensus ``feature_ok``.

        A feature is OK when ``n_ok >= min_batches``. Without ``min_batches`` it has to be
        OK in every batch: with ``missing="fail"`` every batch loaded, so a batch that
        doesn't report the feature counts against it; with ``missing="ignore"`` only the
        batches that report it (the old pipeline's global rule).
        """
        self._require("feature", "feature_ok", self.batch_col)
        if missing not in ("fail", "ignore"):
            raise ValueError(f"missing must be 'fail' or 'ignore', got {missing!r}")
        lf = self._lf
        counts = lf.group_by("feature", maintain_order=True).agg(
            pl.col(self.batch_col).n_unique().cast(pl.UInt32).alias("n_batches"),
            pl.col("feature_ok").fill_null(False).sum().cast(pl.UInt32).alias("n_ok"),
        )
        if min_batches is not None:
            required = pl.lit(min_batches)
        elif missing == "ignore":
            required = pl.col("n_batches")
        else:
            required = pl.lit(
                lf.select(pl.col(self.batch_col).n_unique()).collect().item()
            )
        return counts.with_columns(
            (pl.col("n_ok") >= required).alias("feature_ok")
        ).collect()

    def consensus(
        self,
        min_batches: int | None = None,
        *,
        missing: Literal["fail", "ignore"] = "fail",
    ) -> list[str]:
        """Features that are OK in every batch, or in at least ``min_batches`` of them.

        With ``missing="fail"`` (the default) a feature absent from a batch's blocklist
        counts as not OK there; ``missing="ignore"`` judges it only on the batches that
        report it. See `table`.
        """
        table = self.table(min_batches, missing=missing)
        return table.filter("feature_ok").get_column("feature").to_list()
