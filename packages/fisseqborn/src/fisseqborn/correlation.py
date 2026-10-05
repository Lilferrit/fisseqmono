"""x-vs-y scatter / density plots with correlation statistics."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from scipy import stats

from . import _data
from ._base import Plot

_CORNERS: dict[str, tuple[float, float, str, str]] = {
    "upper left": (0.05, 0.95, "left", "top"),
    "upper right": (0.95, 0.95, "right", "top"),
    "lower left": (0.05, 0.05, "left", "bottom"),
    "lower right": (0.95, 0.05, "right", "bottom"),
}
_STAT_NAMES = {"pearson": "Pearson r", "spearman": "Spearman ρ"}


class CorrelationPlot(Plot):
    """Compare two numeric columns: replicate vs. replicate, filtered vs. unfiltered,
    score vs. cell count, ...

    Parameters
    ----------
    data : pl.DataFrame
    x, y : str
        Numeric columns.
    hue, hue_order, palette
        Optional categorical coloring (scatter) or per-level densities (kde).
    kind : {"scatter", "kde"}
        Points, or a filled 2-D KDE (with a density colorbar when there's no hue).
    stat : {"pearson", "spearman", None}
        Correlation shown in a text box, computed over rows where both values are finite.
    stat_loc : {"lower right", "lower left", "upper right", "upper left"}
    fit : {None, "linear", "lowess"}
        Draw a dashed black fit line through all points.
    identity : bool
        Draw the dashed ``y = x`` line and give both axes the same limits.
    lims : tuple[float, float] | None
        Limits applied to both axes, e.g. ``(0, 1)`` for AUROCs.
    count_sides : bool
        Annotate how many points fall above / below the ``y = x`` line.
    fit_kw : mapping | None
        Extra kwargs for `seaborn.regplot`.
    **kw
        Passed to `seaborn.scatterplot` / `seaborn.kdeplot`.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        x: str,
        y: str,
        hue: str | None = None,
        hue_order: Sequence[Any] | None = None,
        palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        kind: Literal["scatter", "kde"] = "scatter",
        stat: Literal["pearson", "spearman"] | None = "pearson",
        stat_loc: str = "lower right",
        fit: Literal["linear", "lowess"] | None = None,
        identity: bool = False,
        lims: tuple[float, float] | None = None,
        count_sides: bool = False,
        fit_kw: Mapping[str, Any] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        _data.require_columns(data, x, y, hue)
        if kind not in ("scatter", "kde"):
            raise ValueError(f"kind must be 'scatter' or 'kde', got {kind!r}")
        if stat not in (None, *_STAT_NAMES):
            raise ValueError(
                f"stat must be None, 'pearson' or 'spearman', got {stat!r}"
            )
        if fit not in (None, "linear", "lowess"):
            raise ValueError(f"fit must be None, 'linear' or 'lowess', got {fit!r}")
        if stat_loc not in _CORNERS:
            raise ValueError(
                f"stat_loc must be one of {list(_CORNERS)}, got {stat_loc!r}"
            )
        self.x, self.y, self.hue = x, y, hue
        self.hue_order = _data.resolve_order(data, hue, hue_order) if hue else None
        self.palette = _data.resolve_palette(self.hue_order, palette) if hue else None
        self.kind, self.stat, self.stat_loc, self.fit = kind, stat, stat_loc, fit
        self.identity, self.lims, self.count_sides = identity, lims, count_sides
        self.fit_kw = dict(fit_kw or {})
        self.kw = kw

    def _finite(self) -> pl.DataFrame:
        """Rows where both ``x`` and ``y`` are finite numbers."""
        return self.data.filter(
            pl.col(self.x).cast(pl.Float64).is_finite()
            & pl.col(self.y).cast(pl.Float64).is_finite()
        )

    def correlation(self) -> tuple[float, float, int]:
        """``(statistic, p_value, n)`` for this plot's ``stat`` (Pearson if ``stat=None``)."""
        df = self._finite()
        xs = df.get_column(self.x).to_numpy()
        ys = df.get_column(self.y).to_numpy()
        if len(xs) < 3:
            return float("nan"), float("nan"), len(xs)
        fn = stats.spearmanr if self.stat == "spearman" else stats.pearsonr
        result = fn(xs, ys)
        return float(result.statistic), float(result.pvalue), len(xs)

    def _draw(self, ax: Axes) -> None:
        df = self._finite()
        pdf = _data.to_pandas(df, self.x, self.y, self.hue)
        if self.kind == "scatter":
            sns.scatterplot(
                data=pdf,
                x=self.x,
                y=self.y,
                hue=self.hue,
                hue_order=self.hue_order,
                palette=self.palette,
                ax=ax,
                **{"alpha": 0.8, "linewidth": 0, **self.kw},
            )
        else:
            kde_kw: dict[str, Any] = {"fill": True, **self.kw}
            if self.hue is None:
                kde_kw = {
                    "cmap": "viridis",
                    "cbar": True,
                    "cbar_kws": {"label": "Density"},
                    **kde_kw,
                }
            sns.kdeplot(
                data=pdf,
                x=self.x,
                y=self.y,
                hue=self.hue,
                hue_order=self.hue_order,
                palette=self.palette,
                ax=ax,
                **kde_kw,
            )

        if self.fit is not None:
            sns.regplot(
                data=pdf,
                x=self.x,
                y=self.y,
                scatter=False,
                ax=ax,
                lowess=self.fit == "lowess",
                **{
                    "color": "black",
                    "ci": None,
                    "line_kws": {"linestyle": "--", "linewidth": 1},
                    **self.fit_kw,
                },
            )

        if self.lims is not None:
            ax.set(xlim=self.lims, ylim=self.lims)
        elif self.identity and len(df):
            lo = min(df.get_column(self.x).min(), df.get_column(self.y).min())  # type: ignore[type-var]
            hi = max(df.get_column(self.x).max(), df.get_column(self.y).max())  # type: ignore[type-var]
            pad = 0.05 * (hi - lo) if hi > lo else 0.5
            ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad))
        if self.identity:
            ax.axline((0, 0), slope=1, linestyle="--", color="black", linewidth=1)
            ax.set_aspect("equal", adjustable="box")

        if self.stat is not None:
            r, p, n = self.correlation()
            p_text = "p < 1e-300" if p == 0 else f"p = {p:.2e}"
            self._corner_text(
                ax,
                self.stat_loc,
                f"{_STAT_NAMES[self.stat]} = {r:.2f}, {p_text}\nn = {n}",
                boxed=True,
            )
        if self.count_sides:
            n_above = df.filter(pl.col(self.y) > pl.col(self.x)).height
            n_below = df.filter(pl.col(self.y) < pl.col(self.x)).height
            self._corner_text(ax, "upper left", f"n = {n_above}")
            below_loc = "lower right"
            if self.stat is not None and self.stat_loc == "lower right":
                below_loc = "lower right, raised"
            self._corner_text(ax, below_loc, f"n = {n_below}")
            if ax.get_legend() is not None:  # keep the upper-left count visible
                sns.move_legend(ax, "center left", bbox_to_anchor=(1.0, 0.5))

    @staticmethod
    def _corner_text(ax: Axes, loc: str, text: str, boxed: bool = False) -> None:
        raised = loc.endswith(", raised")
        x, y, ha, va = _CORNERS[loc.removesuffix(", raised")]
        if raised:
            y += 0.12
        bbox = (
            {
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.7,
                "edgecolor": "gray",
            }
            if boxed
            else None
        )
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            fontsize=9,
            bbox=bbox,
            zorder=10,
        )
