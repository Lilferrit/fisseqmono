"""Pairwise scatter grids (``sns.pairplot``) with the fisseq palettes and orders."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib as mpl
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import _data
from ._base import FigurePlot, config


class PairPlot(FigurePlot):
    """Grid of pairwise scatter plots of ``vars``, with each variable's distribution on
    the diagonal (figure-level, wraps `seaborn.pairplot`).

    >>> fb.PairPlot(profiles, vars=[f"X_{i}" for i in range(5)], hue="meta_variant_type",
    ...             title="Pairwise PCA components").save("vis/pcs.png")

    `plot` returns ``(fig, PairGrid)``; layers are applied to the top-left axes.

    Parameters
    ----------
    data : pl.DataFrame
    vars : Sequence[str] | polars selector
        The columns to plot against each other.
    hue : str | None
        Column to color by. Categorical levels get the fisseq order and palette when they
        are known (e.g. variant types), otherwise natural order (cluster ``"2"`` before
        ``"10"``) and seaborn's default palette.
    hue_order, palette
        Override the level order and colors.
    kind : {"scatter", "kde", "hist", "reg"}
        Off-diagonal plot kind.
    diag_kind : {"kde", "hist", None}, default "kde"
    corner : bool
        Only draw the lower triangle.
    height : float, default 2.4
        Size of each panel in inches (the figure size follows from it, so ``figsize``
        isn't accepted).
    plot_kws, diag_kws : mapping | None
        Passed to the off-diagonal / diagonal plotting functions. ``plot_kws`` defaults to
        ``alpha=0.6, s=15, linewidth=0`` for scatter plots.
    **pairplot_kw
        Passed to `seaborn.pairplot`.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        vars: Any,
        *,
        hue: str | None = None,
        hue_order: Sequence[Any] | None = None,
        palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        kind: Literal["scatter", "kde", "hist", "reg"] = "scatter",
        diag_kind: Literal["kde", "hist"] | None = "kde",
        corner: bool = False,
        height: float = 2.4,
        plot_kws: Mapping[str, Any] | None = None,
        diag_kws: Mapping[str, Any] | None = None,
        title: str | None = None,
        dpi: int | None = None,
        **pairplot_kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, dpi=dpi)
        if isinstance(vars, str):
            vars = [vars]
        if isinstance(vars, Sequence):
            _data.require_columns(data, *vars)
            self.vars = list(vars)
        else:
            self.vars = data.select(vars).columns
        if not self.vars:
            raise ValueError("PairPlot needs at least one variable")
        _data.require_columns(data, hue)
        self.hue = hue
        categorical = hue is not None and not _data.is_numeric(data, hue)
        self.hue_order = (
            _data.resolve_order(data, hue, hue_order) if categorical else None
        )
        self.palette = (
            _data.resolve_palette(self.hue_order, palette)  # type: ignore[arg-type]
            if self.hue_order is not None
            else palette
        )
        self.kind, self.diag_kind, self.corner, self.height = (
            kind,
            diag_kind,
            corner,
            height,
        )
        default_kws = (
            {"alpha": 0.6, "s": 15, "linewidth": 0} if kind == "scatter" else {}
        )
        self.plot_kws = {**default_kws, **(plot_kws or {})}
        self.diag_kws = dict(diag_kws or {})
        self.pairplot_kw = pairplot_kw

    def _draw_figure(self) -> tuple[Figure, sns.PairGrid]:
        df = _data.to_pandas(self.data, *self.vars, self.hue)
        if self.hue_order is not None:
            # levels are matched by value, so keep the hue column's own values
            df = df[df[self.hue].isin(self.hue_order)]
        with mpl.rc_context({"figure.dpi": self.dpi or config.dpi}):
            grid = sns.pairplot(
                df,
                vars=self.vars,
                hue=self.hue,
                hue_order=self.hue_order,
                palette=self.palette,
                kind=self.kind,
                diag_kind=self.diag_kind,
                corner=self.corner,
                height=self.height,
                plot_kws=self.plot_kws,
                diag_kws=self.diag_kws,
                **self.pairplot_kw,
            )
        return grid.figure, grid

    def _main_ax(self, handle: sns.PairGrid) -> Axes:
        return handle.axes[0, 0]
