"""Matrix heatmaps."""

from collections.abc import Sequence
from typing import Any, Self

import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from . import _data
from ._base import Plot
from .metrics import Metric, MetricMatrix


def _natural_sorted(values: Sequence[Any]) -> list[Any]:
    return sorted(values, key=_data._natural_key)


def with_cbar_label(kw: dict[str, Any], label: str) -> dict[str, Any]:
    """``kw`` with ``label`` as the default colorbar label; a caller's ``cbar_kws`` wins."""
    return {**kw, "cbar_kws": {"label": label, **(kw.get("cbar_kws") or {})}}


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
                raise ValueError(
                    "Long form (values=...) needs a single `columns` column name"
                )
            _data.require_columns(data, index, columns, values)
        else:
            if isinstance(columns, str):
                raise ValueError("A single `columns` name needs `values` (long form)")
            if columns is None:
                columns = [
                    c for c in data.columns if c != index and _data.is_numeric(data, c)
                ]
            _data.require_columns(data, index, *columns)
        self.index, self.columns, self.values = index, columns, values
        self.aggregate, self.symmetric, self.fill_value = (
            aggregate,
            symmetric,
            fill_value,
        )
        self.nan_color = nan_color
        #: Shared-row counts per cell, set by `Heatmap.correlation`.
        self.n_shared: pd.DataFrame | None = None
        self.row_order, self.col_order = row_order, col_order
        self.heatmap_kw = {
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "center": center,
            "annot": annot,
            "fmt": fmt,
            **kw,
        }

    @classmethod
    def from_matrix(
        cls, matrix: MetricMatrix, *, index: str | None = None, **kw: Any
    ) -> Self:
        """Heatmap of a `MetricMatrix`, computed (e.g. `Metric.pairwise`,
        `Profiles.distances`) or loaded (`MetricMatrix.from_array`, `MetricMatrix.load`).

        The rows and columns keep the matrix's order. The color scale defaults to the
        metric's (`Metric.default_scale`: ``vlag`` from -1 to 1 for a similarity,
        ``Blues`` from 0 to 1 for a squared one, ``Blues_r`` from 0 for a distance), the
        colorbar is labeled with it, and cells are annotated when the matrix is at most
        20 × 20. The matrix's ``n_shared`` (if any) becomes the plot's ``n_shared``.
        ``index`` names the label column (default: the matrix's ``key`` or ``"label"``).
        """
        index = index or matrix.key or "label"
        wide = matrix.to_frame(index).with_columns(pl.col(index).cast(pl.String))
        rows = wide.get_column(index).to_list()
        columns = [c for c in wide.columns if c != index]
        kw = with_cbar_label(
            {
                **matrix.metric.default_scale,
                "annot": max(matrix.shape) <= 20,
                "row_order": rows,
                "col_order": columns,
                **kw,
            },
            matrix.metric.label,
        )
        plot = cls(wide, index=index, columns=columns, **kw)
        if matrix.n_shared is not None:
            plot.n_shared = pd.DataFrame(matrix.n_shared, index=rows, columns=columns)
        return plot

    @classmethod
    def correlation(
        cls,
        data: pl.DataFrame,
        columns: Sequence[str] | None = None,
        *,
        y_columns: Sequence[str] | None = None,
        metric: str | Metric = "pearson",
        min_shared: int = 10,
        **kw: Any,
    ) -> Self:
        """Heatmap of ``metric`` (see `fisseqborn.metrics`; default Pearson r) between
        every pair of ``columns`` (default: all numeric columns). Each cell is computed
        over the rows where both of its columns are finite, so sparse inputs (e.g. one
        column per batch, with batches covering different tiles) still work; see
        `Metric.pairwise`.

        Cells with fewer than ``min_shared`` shared rows are NaN, drawn in ``nan_color``.
        The shared-row counts are available as the plot's ``n_shared`` (a pandas DataFrame).

        ``y_columns`` puts different columns on the rows (y axis) than ``columns`` on the
        x axis, e.g. one score per batch against another score per batch. Each cell then
        compares its row's column with its column's column (`Metric.cross`), so the
        matrix isn't symmetric and its diagonal needn't be 1.

        The color scale is the metric's (see `from_matrix`), with ``annot=True``; the
        colorbar is labeled with the metric, e.g. ``"Pearson r²"``.
        """
        data = _data.as_frame(data)
        if columns is None:
            columns = [c for c in data.columns if _data.is_numeric(data, c)]
        _data.require_columns(data, *columns, *(y_columns or []))
        resolved = Metric.get(metric)
        if y_columns is None:
            matrix = resolved.pairwise(data, list(columns), min_shared=min_shared)
        else:
            matrix = resolved.cross(
                data.select(y_columns), data.select(columns), min_shared=min_shared
            )
        return cls.from_matrix(matrix, index="column", **{"annot": True, **kw}).set(
            ylabel=""
        )

    def matrix(self) -> pd.DataFrame:
        """The matrix that will be drawn, as a pandas DataFrame."""
        if self.values is not None:
            wide = self.data.pivot(
                on=self.columns,
                index=self.index,
                values=self.values,  # type: ignore[arg-type]
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
        rows = (
            list(self.row_order)
            if self.row_order is not None
            else _natural_sorted(list(mat.index))
        )
        cols = (
            list(self.col_order)
            if self.col_order is not None
            else _natural_sorted(list(mat.columns))
        )
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
