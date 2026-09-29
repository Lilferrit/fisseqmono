"""Per-batch feature blocklists from the pipeline's feature selection."""

from collections.abc import Sequence
from os import PathLike
from typing import Self

import polars as pl

from . import _pipeline
from .dataset import Dataset


class Blocklists(Dataset):
    """Which features passed each batch's replicate-reproducibility filter.

    One row per (feature, statistic, batch) with ``feature`` (the full column name, e.g.
    ``AreaShape_Area_median``), ``median_r`` (the median split-half correlation),
    ``feature_ok``, ``meta_experiment`` and ``meta_feature_type``. Use `consensus` to get
    the features that pass everywhere, and pass them to `Profiles.keep_features`:

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
        self, data: "pl.DataFrame | pl.LazyFrame | Dataset", *, batch_col: str = "meta_experiment"
    ) -> None:
        super().__init__(data, variant_col="feature")
        self.batch_col = batch_col

    @classmethod
    def from_pipeline(
        cls,
        pipeline_dir: str | PathLike,
        types: Sequence[str] | None = None,
        *,
        batches: Sequence[str] | None = None,
        batch_col: str = "meta_experiment",
    ) -> Self:
        """Read ``feature_select_batchwise/<batch>/blocklists/<type>.parquet`` for every
        batch (or ``batches``) and statistic type (default: every file present)."""
        frames = []
        for batch_dir in _pipeline.batch_dirs(pipeline_dir, _pipeline.FEATURE_SELECT, batches):
            if types is None:
                paths = sorted((batch_dir / "blocklists").glob("*.parquet"))
                if not paths:
                    raise FileNotFoundError(f"No blocklists in {batch_dir / 'blocklists'}")
            else:
                paths = [batch_dir / "blocklists" / f"{t}.parquet" for t in types]
            for path in paths:
                frames.append(
                    _pipeline.tag(_pipeline.scan(path), batch_dir.name, batch_col).with_columns(
                        pl.lit(path.stem, dtype=pl.String).alias("meta_feature_type")
                    )
                )
        return cls(pl.concat(frames, how="diagonal_relaxed"), batch_col=batch_col)

    def rethreshold(self, min_r: float) -> Self:
        """Recompute ``feature_ok`` as ``median_r > min_r``."""
        self._require("median_r")
        return self.with_columns((pl.col("median_r") > min_r).fill_null(False).alias("feature_ok"))

    def consensus(self, min_batches: int | None = None) -> list[str]:
        """Features that are OK in every batch, or in at least ``min_batches`` of them.

        A feature absent from a batch's blocklist counts as not OK there.
        """
        self._require("feature", "feature_ok", self.batch_col)
        lf = self._lf
        if min_batches is None:
            min_batches = lf.select(pl.col(self.batch_col).n_unique()).collect().item()
        return (
            lf.group_by("feature", maintain_order=True)
            .agg(pl.col("feature_ok").fill_null(False).sum().alias("n_ok"))
            .filter(pl.col("n_ok") >= min_batches)
            .collect()
            .get_column("feature")
            .to_list()
        )
