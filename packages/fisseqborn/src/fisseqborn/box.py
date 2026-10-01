"""Grouped boxplots with optional point overlays and significance annotations."""

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from . import _data
from ._base import Plot

_BOX_WIDTH = 0.8


class BoxPlot(Plot):
    """Boxplot of ``y`` for each level of ``x`` (optionally split by ``hue``).

    Parameters
    ----------
    data : pl.DataFrame
    x, y : str
        Categorical grouping column and numeric value column.
    hue : str | None
        Optional second grouping column, dodged within each ``x`` level. When omitted,
        boxes are colored by ``x``.
    order, hue_order : Sequence | None
        Level orders. Default: fisseq canonical order for known levels, then natural sort.
    palette : mapping | str | list | None
        Colors for the ``hue`` (or ``x``) levels. Default: the fisseq palette when it
        covers every level.
    points : {None, "strip", "density"}
        Overlay individual points: ``"strip"`` as translucent black jittered points,
        ``"density"`` as jittered points colored by per-box KDE density. Fliers are
        hidden whenever points are drawn.
    show_counts : bool, default True
        Append ``(n = N)`` to each x tick label.
    point_kw : mapping | None
        Extra kwargs for the point overlay.
    **box_kw
        Passed to `seaborn.boxplot`.
    """

    default_figsize = (6.0, 6.0)

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        x: str,
        y: str,
        hue: str | None = None,
        order: Sequence[Any] | None = None,
        hue_order: Sequence[Any] | None = None,
        palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        points: Literal["strip", "density"] | None = None,
        show_counts: bool = True,
        point_kw: Mapping[str, Any] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **box_kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        _data.require_columns(data, x, y, hue)
        if points not in (None, "strip", "density"):
            raise ValueError(f"points must be None, 'strip' or 'density', got {points!r}")
        self.x, self.y, self.hue = x, y, hue
        self.order = _data.resolve_order(data, x, order)
        self.hue_order = _data.resolve_order(data, hue, hue_order) if hue else None
        color_levels = self.hue_order if hue else self.order
        self.palette = _data.resolve_palette(color_levels, palette)
        self.points = points
        self.show_counts = show_counts
        self.point_kw = dict(point_kw or {})
        self.box_kw = box_kw

    # ----- layers ---------------------------------------------------------------------

    def annotate_pairs(
        self,
        pairs: Literal["all"] | Sequence[tuple[Any, Any]] = "all",
        *,
        test: str = "Mann-Whitney",
        text_format: str = "star",
        loc: Literal["inside", "outside"] = "inside",
        **configure_kw: Any,
    ) -> Self:
        """Add significance brackets (via `statannotations`) between pairs of boxes.

        The test always uses this plot's own ``x``/``y``/``hue`` columns and orders.

        Parameters
        ----------
        pairs : "all" or list of pairs
            ``"all"`` compares every pair of ``x`` levels, or with ``hue``, every pair of
            hue levels within each ``x`` level. Otherwise a list of pairs in the
            `statannotations` format: ``(x_a, x_b)``, or with hue
            ``((x, hue_a), (x, hue_b))``. Pairs involving an empty group are skipped.
        test, text_format, loc, **configure_kw
            Passed to ``Annotator.configure``.
        """

        def layer(plot: BoxPlot, ax: Axes) -> None:
            from statannotations.Annotator import Annotator

            resolved = plot._resolve_pairs(pairs)
            if not resolved:
                return
            annotator = Annotator(
                ax,
                resolved,
                data=plot._pdf(),
                x=plot.x,
                y=plot.y,
                hue=plot.hue,
                order=plot.order,
                hue_order=plot.hue_order,
            )
            annotator.configure(
                test=test, text_format=text_format, loc=loc, verbose=0, **configure_kw
            )
            annotator.apply_and_annotate()

        return self._with_layer(layer)

    # ----- drawing --------------------------------------------------------------------

    def _valid(self) -> pl.DataFrame:
        """Rows with a usable (non-null, non-NaN) ``y``."""
        return self.data.filter(pl.col(self.y).is_not_null() & pl.col(self.y).is_not_nan())

    def _pdf(self) -> pd.DataFrame:
        return _data.to_pandas(self._valid(), self.x, self.y, self.hue)

    def _group_sizes(self) -> dict[tuple[Any, ...], int]:
        keys = [self.x] if self.hue is None else [self.x, self.hue]
        counts = self._valid().group_by(keys).len()
        return {tuple(row[:-1]): row[-1] for row in counts.iter_rows()}

    def _resolve_pairs(self, pairs: Any) -> list[Any]:
        sizes = self._group_sizes()
        if pairs == "all":
            if self.hue is None:
                pairs = list(combinations(self.order, 2))
            else:
                pairs = [
                    ((xl, a), (xl, b))
                    for xl in self.order
                    for a, b in combinations(self.hue_order or [], 2)
                ]

        def nonempty(group: Any) -> bool:
            key = tuple(group) if self.hue is not None else (group,)
            return sizes.get(key, 0) > 0

        return [p for p in pairs if nonempty(p[0]) and nonempty(p[1])]

    def _draw(self, ax: Axes) -> None:
        pdf = self._pdf()
        hue = self.hue or self.x
        box_kw = {"showfliers": self.points is None, **self.box_kw}
        sns.boxplot(
            data=pdf,
            x=self.x,
            y=self.y,
            hue=hue,
            order=self.order,
            hue_order=self.hue_order if self.hue else self.order,
            palette=self.palette,
            legend=self.hue is not None,
            ax=ax,
            **box_kw,
        )
        if self.points == "strip":
            self._draw_strip(ax, pdf)
        elif self.points == "density":
            self._draw_density(ax)
        if self.show_counts:
            self._label_counts(ax)
        if self.hue is not None and ax.get_legend() is not None:
            ax.legend(title=self.hue)

    def _draw_strip(self, ax: Axes, pdf: pd.DataFrame) -> None:
        kw = {"color": "black", "size": 3, "alpha": 0.5, **self.point_kw}
        if self.hue is None:
            sns.stripplot(data=pdf, x=self.x, y=self.y, order=self.order, ax=ax, **kw)
        else:
            color = kw.pop("color")
            sns.stripplot(
                data=pdf,
                x=self.x,
                y=self.y,
                hue=self.hue,
                order=self.order,
                hue_order=self.hue_order,
                palette={lvl: color for lvl in self.hue_order or []},
                dodge=True,
                legend=False,
                ax=ax,
                **kw,
            )

    def _draw_density(self, ax: Axes) -> None:
        """Jittered points colored by the KDE density of ``y`` within each box."""
        rng = np.random.default_rng(0)
        valid = self._valid()
        hue_levels = self.hue_order or [None]
        n_hue = len(hue_levels)
        width = _BOX_WIDTH / n_hue / 2
        kw = {"cmap": "viridis", "s": 8, "linewidths": 0, "zorder": 3, **self.point_kw}
        for xi, xl in enumerate(self.order):
            for hi, hl in enumerate(hue_levels):
                cond = pl.col(self.x) == xl
                if self.hue is not None:
                    cond &= pl.col(self.hue) == hl
                values = valid.filter(cond).get_column(self.y).to_numpy()
                if values.size == 0:
                    continue
                if values.size > 1 and np.ptp(values) > 0:
                    density = gaussian_kde(values)(values)
                else:
                    density = np.ones_like(values, dtype=float)
                center = xi + (hi - (n_hue - 1) / 2) * width
                jitter = rng.uniform(-width * 0.3, width * 0.3, size=values.size)
                ax.scatter(center + jitter, values, c=density, **kw)

    def _label_counts(self, ax: Axes) -> None:
        counts = dict.fromkeys(self.order, 0)
        for key, n in self._group_sizes().items():
            if key[0] in counts:
                counts[key[0]] += n
        ax.set_xticks(
            range(len(self.order)),
            [f"{lvl}\n(n = {counts[lvl]})" for lvl in self.order],
        )
