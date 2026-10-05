"""Matrix heatmaps."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from scipy import stats

from . import _data
from ._base import Plot


def _natural_sorted(values: Sequence[Any]) -> list[Any]:
    return sorted(values, key=_data._natural_key)


_CORR_LABELS = {"pearson": "Pearson r", "spearman": "Spearman ρ", "cosine": "Cosine similarity"}


def correlation_label(method: str, squared: bool = False) -> str:
    """Colorbar label for a correlation ``method``, e.g. ``"Spearman ρ²"``."""
    return _CORR_LABELS[method] + ("²" if squared else "")


def with_cbar_label(kw: dict[str, Any], label: str) -> dict[str, Any]:
    """``kw`` with ``label`` as the default colorbar label; a caller's ``cbar_kws`` wins."""
    return {**kw, "cbar_kws": {"label": label, **(kw.get("cbar_kws") or {})}}


def pairwise_correlation(
    mat: np.ndarray,
    method: Literal["pearson", "spearman", "cosine"] = "pearson",
    min_shared: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlation (or cosine similarity) between the columns of ``mat``, each pair computed
    over the rows where both columns are finite.

    Spearman ranks within each pair's shared rows, matching `scipy.stats.spearmanr` on that
    pair. Cells with fewer than ``min_shared`` shared rows, or zero variance, are NaN.

    Returns ``(corr, n_shared)``, both ``(n_cols, n_cols)``.
    """
    finite = np.isfinite(mat)
    n_cols = mat.shape[1]
    corr = np.full((n_cols, n_cols), np.nan)
    n_shared = (finite.T.astype(int) @ finite.astype(int)).astype(int)
    for i in range(n_cols):
        for j in range(i, n_cols):
            if n_shared[i, j] < min_shared:
                continue
            ok = finite[:, i] & finite[:, j]
            x, y = mat[ok, i], mat[ok, j]
            if method == "spearman":
                x, y = stats.rankdata(x), stats.rankdata(y)
            if method != "cosine":
                x, y = x - x.mean(), y - y.mean()
            denom = np.sqrt((x @ x) * (y @ y))
            if denom > 0:
                corr[i, j] = corr[j, i] = np.clip((x @ y) / denom, -1.0, 1.0)
    return corr, n_shared


class Heatmap(Plot):
    """Heatmap of a matrix given in long form (``index``/``columns``/``values``) or wide
    form (``index`` plus a list of matrix ``columns``).

    Parameters
    ----------
    data : pl.DataFrame
    index : str
        Row-label column.
    columns : str | Sequence[str] | None
        Long form: the column-label column (requires ``values``). Wide form: the matrix
        columns; ``None`` means every numeric column other than ``index``.
    values : str | None
        Long form: the value column. Duplicate ``(index, columns)`` pairs are an error
        unless ``aggregate`` is given.
    aggregate : {"mean", "median", "first", ...} | None
        polars pivot aggregation for duplicate pairs.
    symmetric : bool
        Fill each missing cell ``(i, j)`` from ``(j, i)``, e.g. for pairwise matrices
        stored once per pair. Rows and columns get the same (union) labels.
    fill_value : float | None
        Value for cells still missing.
    nan_color : color | None
        Color shown for missing cells (the axes background).
    row_order, col_order : Sequence | None
        Label orders. Default: natural sort.
    cmap, vmin, vmax, center, annot, fmt, **kw
        Passed to `seaborn.heatmap`.
    """

    default_figsize = (6.0, 5.0)

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        index: str,
        columns: str | Sequence[str] | None = None,
        values: str | None = None,
        aggregate: str | None = None,
        symmetric: bool = False,
        fill_value: float | None = None,
        nan_color: Any = None,
        row_order: Sequence[Any] | None = None,
        col_order: Sequence[Any] | None = None,
        cmap: Any = None,
        vmin: float | None = None,
        vmax: float | None = None,
        center: float | None = None,
        annot: bool = False,
        fmt: str = ".2f",
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        if values is not None:
            if not isinstance(columns, str):
                raise ValueError("Long form (values=...) needs a single `columns` column name")
            _data.require_columns(data, index, columns, values)
        else:
            if isinstance(columns, str):
                raise ValueError("A single `columns` name needs `values` (long form)")
            if columns is None:
                columns = [c for c in data.columns if c != index and _data.is_numeric(data, c)]
            _data.require_columns(data, index, *columns)
        self.index, self.columns, self.values = index, columns, values
        self.aggregate, self.symmetric, self.fill_value = aggregate, symmetric, fill_value
        self.nan_color = nan_color
        #: Shared-row counts per cell, set by `Heatmap.correlation`.
        self.n_shared: pd.DataFrame | None = None
        self.row_order, self.col_order = row_order, col_order
        self.heatmap_kw = {"cmap": cmap, "vmin": vmin, "vmax": vmax, "center": center,
                           "annot": annot, "fmt": fmt, **kw}

    @classmethod
    def correlation(
        cls,
        data: pl.DataFrame,
        columns: Sequence[str] | None = None,
        *,
        method: Literal["pearson", "spearman", "cosine"] = "pearson",
        squared: bool = False,
        min_shared: int = 10,
        **kw: Any,
    ) -> Self:
        """Heatmap of the pairwise correlation (or cosine similarity) between ``columns``
        (default: all numeric columns). Each cell is computed over the rows where both of
        its columns are finite, so sparse inputs (e.g. one column per batch, with batches
        covering different tiles) still work; see `pairwise_correlation`.

        Cells with fewer than ``min_shared`` shared rows are NaN, drawn in ``nan_color``.
        The shared-row counts are available as the plot's ``n_shared`` (a pandas DataFrame).
        ``squared=True`` shows the squared correlation (e.g. Pearson r²) instead.

        Defaults to ``cmap="vlag", vmin=-1, vmax=1, center=0, annot=True``, or
        ``cmap="Blues", vmin=0, vmax=1, annot=True`` when ``squared``. The colorbar is
        labeled with the quantity, e.g. ``"Pearson r²"``.
        """
        data = _data.as_frame(data)
        if columns is None:
            columns = [c for c in data.columns if _data.is_numeric(data, c)]
        _data.require_columns(data, *columns)
        columns = list(columns)
        mat = data.select(columns).cast(pl.Float64).fill_null(np.nan).to_numpy()
        corr, n_shared = pairwise_correlation(mat, method, min_shared)
        if squared:
            corr = corr**2
        wide = pl.DataFrame({"column": columns}).with_columns(
            pl.Series(c, corr[:, i]) for i, c in enumerate(columns)
        )
        scale = ({"cmap": "Blues", "vmin": 0, "vmax": 1} if squared
                 else {"cmap": "vlag", "vmin": -1, "vmax": 1, "center": 0})
        kw = with_cbar_label(
            {**scale, "annot": True, "row_order": columns, "col_order": columns, **kw},
            correlation_label(method, squared),
        )
        plot = cls(wide, index="column", columns=columns, **kw).set(ylabel="")
        plot.n_shared = pd.DataFrame(n_shared, index=columns, columns=columns)
        return plot

    def matrix(self) -> pd.DataFrame:
        """The matrix that will be drawn, as a pandas DataFrame."""
        if self.values is not None:
            wide = self.data.pivot(
                on=self.columns, index=self.index, values=self.values,  # type: ignore[arg-type]
                aggregate_function=self.aggregate,  # type: ignore[arg-type]
            )
            col_labels = [c for c in wide.columns if c != self.index]
        else:
            wide = self.data.select(self.index, *self.columns)
            col_labels = list(self.columns)
        mat = wide.to_pandas().set_index(self.index)
        mat.index = mat.index.map(str)  # match pivot's string column labels
        mat.columns = pd.Index(col_labels)
        if self.symmetric:
            labels = _natural_sorted(list(set(mat.index) | set(mat.columns)))
            mat = mat.reindex(index=labels, columns=labels)
            mat = mat.combine_first(mat.T).reindex(index=labels, columns=labels)
        rows = list(self.row_order) if self.row_order is not None else _natural_sorted(list(mat.index))
        cols = list(self.col_order) if self.col_order is not None else _natural_sorted(list(mat.columns))
        mat = mat.reindex(index=rows, columns=cols)
        if self.fill_value is not None:
            mat = mat.fillna(self.fill_value)
        mat.index.name = self.index
        mat.columns.name = self.columns if isinstance(self.columns, str) else None
        return mat

    def _draw(self, ax: Axes) -> None:
        if self.nan_color is not None:
            ax.set_facecolor(self.nan_color)
        sns.heatmap(self.matrix(), ax=ax, **self.heatmap_kw)
