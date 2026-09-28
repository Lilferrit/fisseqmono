"""Volcano plots of per-feature effect sizes against -log10 p-values."""

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any, Self

import numpy as np
import polars as pl
import polars.selectors as cs
from matplotlib.axes import Axes

from . import _data, fisseq
from ._base import Plot


@dataclasses.dataclass(frozen=True)
class _PointLayer:
    where: pl.Expr
    label: str | None
    color: Any
    scatter_kw: Mapping[str, Any]


class VolcanoPlot(Plot):
    """Volcano plot of long-form ``(variant, feature, effect, -log10 p)`` rows.

    Points are added in groups with chained `layer` calls, each selecting its rows with a
    polars expression over the metadata columns. Layers are drawn in the order they are
    added, so add the largest group first and the controls last to keep them visible.
    Without any layers, every row is drawn as a single grey group. Rows with a missing or
    non-finite ``x`` or ``y`` are not drawn or counted.

    Use `from_wide` for one row per variant with ``<feature>_median`` /
    ``<feature>_KSnegLogP`` column pairs.

    >>> (fb.VolcanoPlot(long_df, x="median", y="KSnegLogP")
    ...    .layer(pl.col("meta_variant_type") == "Single Missense", label="Single Missense")
    ...    .layer(pl.col("meta_variant_type") == "Synonymous", label="Synonymous")
    ...    .save("vis/volcano.png"))

    Parameters
    ----------
    data : pl.DataFrame
        One row per (variant, feature) pair, plus any metadata columns the layers filter on.
    x : str
        Effect-size column, e.g. the median normalized feature value.
    y : str
        ``-log10(p)`` column.
    alpha : float | None, default 0.05
        Significance level drawn as a dashed horizontal line and listed in the legend.
        ``None`` draws no line.
    bonferroni : bool, default True
        Divide ``alpha`` by the number of plotted points (the tests across all layers).
    x_quantiles : tuple[float, float] | None, default (0.001, 0.999)
        Limit the x-axis to these quantiles of the plotted points, so a few extreme
        values don't squash the rest. ``None`` shows every point.
    **scatter_kw
        Defaults for every layer's ``ax.scatter`` call; `layer` kwargs override them.
    """

    default_figsize = (7.0, 6.0)

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        x: str,
        y: str,
        alpha: float | None = 0.05,
        bonferroni: bool = True,
        x_quantiles: tuple[float, float] | None = (0.001, 0.999),
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **scatter_kw: Any,
    ) -> None:
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        _data.require_columns(data, x, y)
        self.x, self.y = x, y
        self.alpha, self.bonferroni = alpha, bonferroni
        self.x_quantiles = x_quantiles
        self.scatter_kw = scatter_kw
        self._point_layers: list[_PointLayer] = []

    @classmethod
    def from_wide(
        cls,
        data: pl.DataFrame,
        *,
        x_suffix: str = "_median",
        y_suffix: str = "_KSnegLogP",
        metadata: cs.Selector | None = None,
        **kw: Any,
    ) -> Self:
        """Volcano plot of a wide frame with one row per variant.

        Every feature with both a ``<feature><x_suffix>`` and a ``<feature><y_suffix>``
        column becomes one point per variant. The long-form data has a ``feature`` column
        (the feature name without its suffix), the ``metadata`` columns for `layer` to
        filter on, and ``x``/``y`` columns named after the suffixes without their leading
        underscore (by default ``"median"`` and ``"KSnegLogP"``). ``metadata`` defaults to
        every ``meta_`` column. ``kw`` is passed to the constructor.
        """
        x, y = x_suffix.lstrip("_"), y_suffix.lstrip("_")
        features = [
            c.removesuffix(x_suffix)
            for c in data.columns
            if c.endswith(x_suffix) and c.removesuffix(x_suffix) + y_suffix in data.columns
        ]
        if not features:
            raise ValueError(
                f"No feature has both a {x_suffix!r} and a {y_suffix!r} column"
            )
        index = data.select(
            cs.starts_with("meta_") if metadata is None else metadata
        ).columns

        def unpivot(suffix: str, name: str) -> pl.DataFrame:
            return data.unpivot(
                on=[f + suffix for f in features],
                index=index,
                variable_name="feature",
                value_name=name,
            )

        # Both unpivots walk the features in the same order, so their rows line up
        long = unpivot(x_suffix, x).with_columns(
            pl.col("feature").str.strip_suffix(x_suffix),
            unpivot(y_suffix, y).get_column(y),
        )
        return cls(long, x=x, y=y, **kw)

    # ----- layers ---------------------------------------------------------------------

    def layer(
        self,
        where: pl.Expr,
        *,
        label: str | None = None,
        color: Any = None,
        **scatter_kw: Any,
    ) -> Self:
        """Add a group of points: the rows matching ``where``, drawn above earlier layers.

        ``color`` defaults to the fisseq palette entry for ``label`` (e.g.
        ``"Synonymous"`` is dark green), else grey. The legend shows ``label`` with the
        number of points drawn.
        """
        if color is None:
            color = fisseq.PALETTE.get(label, "grey")
        new = copy.copy(self)
        new._point_layers = [
            *self._point_layers, _PointLayer(where, label, color, scatter_kw)
        ]
        new._figure = None
        return new

    # ----- drawing --------------------------------------------------------------------

    def significance_threshold(self) -> float | None:
        """The ``-log10(p)`` height of the significance line, or ``None`` without one."""
        return self._threshold(sum(rows.height for _, rows in self._points()))

    def _threshold(self, n_tests: int) -> float | None:
        if self.alpha is None:
            return None
        alpha = self.alpha / max(n_tests, 1) if self.bonferroni else self.alpha
        return float(-np.log10(alpha))

    def _points(self) -> list[tuple[_PointLayer, pl.DataFrame]]:
        """Each layer with its drawable rows (only the ``x`` and ``y`` columns)."""
        layers = self._point_layers or [_PointLayer(pl.lit(True), None, "grey", {})]
        finite = self.data.filter(
            pl.col(self.x).cast(pl.Float64).is_finite(),
            pl.col(self.y).cast(pl.Float64).is_finite(),
        )
        return [(lyr, finite.filter(lyr.where).select(self.x, self.y)) for lyr in layers]

    def _draw(self, ax: Axes) -> None:
        points = self._points()
        for zorder, (lyr, rows) in enumerate(points, start=1):
            kw = {
                "s": 1,
                "alpha": 0.1,
                "linewidths": 0,
                "rasterized": True,
                **self.scatter_kw,
                **lyr.scatter_kw,
            }
            ax.scatter(
                rows.get_column(self.x).to_numpy(),
                rows.get_column(self.y).to_numpy(),
                color=lyr.color,
                label=None if lyr.label is None else f"{lyr.label} (n={rows.height:,})",
                zorder=zorder,
                **kw,
            )

        threshold = self._threshold(sum(rows.height for _, rows in points))
        if threshold is not None:
            name = "Bonferroni p" if self.bonferroni else "p"
            ax.axhline(threshold, color="black", linestyle="--", linewidth=1,
                       zorder=len(points) + 1, label=f"{name} = {self.alpha:g}")

        if self.x_quantiles is not None:
            xs = pl.concat([rows.get_column(self.x) for _, rows in points])
            if xs.len():
                lo, hi = self.x_quantiles
                ax.set_xlim(xs.quantile(lo), xs.quantile(hi))

        ax.set(xlabel=self.x, ylabel=self.y)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            legend = ax.legend(markerscale=10, frameon=False, loc="upper left")
            for handle in legend.legend_handles:
                handle.set_alpha(1)
