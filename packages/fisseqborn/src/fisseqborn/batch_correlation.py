"""Batch-vs-batch correlation matrices computed from long-form scores."""

from collections.abc import Sequence
from itertools import combinations_with_replacement
from typing import Any, Literal

import numpy as np
import polars as pl
from matplotlib.axes import Axes

from . import _data
from .heatmap import Heatmap, pairwise_correlation


class BatchCorrelationHeatmap(Heatmap):
    """Heatmap of the correlation of ``score`` between every pair of batches, with one point
    per ``label`` the two batches share (e.g. replicate-vs-replicate OVWT AUROCs, one point
    per variant).

    Pairs of batches with fewer than ``min_shared`` shared labels, such as replicates from
    different tiles, are left missing and drawn in ``nan_color`` (black by default).

    Parameters
    ----------
    data : pl.DataFrame
        Long form: one row per ``(batch, label)``.
    batch : str
        Column whose values become the matrix rows and columns.
    label : str
        Column matched between batches, e.g. the variant.
    score : str
        Numeric column that is correlated.
    method : {"spearman", "pearson"}
    min_shared : int
        Minimum number of shared labels (with finite scores in both batches) for a pair to
        get a correlation.
    aggregate : {"mean", "median", "first", ...} | None
        polars aggregation for duplicate ``(batch, label)`` rows, which are otherwise an error.
    order : Sequence | None
        Batch order for both rows and columns. Default: natural sort.
    **kw
        Passed to `Heatmap`. Defaults: ``cmap="Blues", vmin=0, vmax=1, annot=True,
        nan_color="black", linewidths=0.5, linecolor="black"``.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        batch: str,
        label: str,
        score: str,
        method: Literal["spearman", "pearson"] = "spearman",
        min_shared: int = 10,
        aggregate: str | None = None,
        order: Sequence[Any] | None = None,
        figsize: tuple[float, float] | None = None,
        **kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        _data.require_columns(data, batch, label, score)
        if method not in ("spearman", "pearson"):
            raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")
        if aggregate is None and data.select(batch, label).is_duplicated().any():
            raise ValueError(
                f"Duplicate ({batch!r}, {label!r}) rows; pass aggregate='mean' (or similar)"
            )
        self.batch, self.label, self.score = batch, label, score
        self.method, self.min_shared = method, min_shared
        self.batch_aggregate = aggregate

        wide = data.select(
            pl.col(batch).cast(pl.String), label, pl.col(score).cast(pl.Float64)
        ).pivot(on=batch, index=label, values=score, aggregate_function=aggregate)  # type: ignore[arg-type]
        self._scores = wide.drop(label)
        self._pairs = self._compute_pairs()

        batches = self._scores.columns
        mat = np.full((len(batches), len(batches)), np.nan)
        pos = {b: i for i, b in enumerate(batches)}
        for a, b, r in self._pairs.select("batch_a", "batch_b", "r").iter_rows():
            mat[pos[a], pos[b]] = mat[pos[b], pos[a]] = r
        matrix = pl.DataFrame({"batch": batches}).with_columns(
            pl.Series(b, mat[:, i]) for i, b in enumerate(batches)
        )

        if figsize is None:
            n = len(batches)
            figsize = (max(6.0, 0.65 * n + 1.0), max(5.0, 0.6 * n + 0.6))
        kw = {"cmap": "Blues", "vmin": 0, "vmax": 1, "annot": True, "nan_color": "black",
              "linewidths": 0.5, "linecolor": "black", "row_order": order, "col_order": order,
              **kw}
        super().__init__(matrix, index="batch", columns=list(batches), figsize=figsize, **kw)

    def _compute_pairs(self) -> pl.DataFrame:
        batches = self._scores.columns
        corr, n_shared = pairwise_correlation(
            self._scores.fill_null(np.nan).to_numpy(), self.method, self.min_shared
        )
        rows = [
            (batches[i], batches[j], float(corr[i, j]), int(n_shared[i, j]))
            for i, j in combinations_with_replacement(range(len(batches)), 2)
        ]
        return pl.DataFrame(
            rows, schema={"batch_a": pl.String, "batch_b": pl.String, "r": pl.Float64,
                          "n_shared": pl.Int64},
            orient="row",
        )

    def pairs(self) -> pl.DataFrame:
        """One row per unordered batch pair (including each batch with itself):
        ``batch_a, batch_b, r, n_shared``."""
        return self._pairs

    def _draw(self, ax: Axes) -> None:
        super()._draw(ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set(xlabel="", ylabel="")
