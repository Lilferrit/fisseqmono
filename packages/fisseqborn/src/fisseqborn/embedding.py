"""2-D embedding (UMAP / PCA / MDS) scatter and hexbin plots."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

from . import _data
from ._base import Plot


def _mode(values: np.ndarray) -> int:
    return int(np.bincount(np.asarray(values, dtype=int)).argmax())


def add_legend_entry(ax: Axes, handle: Any, label: str) -> None:
    """Append ``handle`` to the axes' legend, keeping any existing (e.g. seaborn) entries."""
    old = ax.get_legend()
    if old is None:
        handles, labels = ax.get_legend_handles_labels()
        title = None
    else:
        handles = list(old.legend_handles)
        labels = [t.get_text() for t in old.get_texts()]
        title = old.get_title().get_text() or None
    if handle not in handles:
        handles.append(handle)
        labels.append(label)
    ax.legend(handles, labels, title=title)


class EmbeddingPlot(Plot):
    """Scatter or hexbin plot of a 2-D embedding, colored by ``hue``.

    Parameters
    ----------
    data : pl.DataFrame
    x, y : str
        Embedding coordinate columns, e.g. ``"meta_notebook_umap_1"`` / ``"..._2"``.
    hue : str | None
        Column to color by. Categorical columns get a discrete palette; numeric columns
        get a colormap and colorbar.
    kind : {"scatter", "hexbin"}
        ``"hexbin"`` aggregates points into hexagons: numeric ``hue`` is averaged,
        categorical ``hue`` shows the most common level, no ``hue`` shows counts.
    hue_order, palette
        Categorical levels (legend order) and their colors.
    draw_order : Sequence | None
        Order in which categorical levels are drawn, bottom first. Default: largest group
        first, so rare classes (e.g. synonymous controls) stay visible on top.
    cmap, vmin, vmax
        Colormap and limits for numeric ``hue`` / hexbin counts.
    center : float | None
        Use a range symmetric around ``center`` (e.g. 0 for z-scores). The half-width is
        the largest absolute deviation, capped at ``clip``.
    clip : float | None, default 3.0
        Cap for the symmetric half-width when ``center`` is set.
    gridsize : int, default 50
        Hexbin grid size.
    colorbar_label : str | None
        Defaults to the ``hue`` column name.
    **kw
        Passed to the underlying ``sns.scatterplot`` / ``ax.scatter`` / ``ax.hexbin``.
    """

    default_figsize = (8.0, 8.0)

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        x: str,
        y: str,
        hue: str | None = None,
        kind: Literal["scatter", "hexbin"] = "scatter",
        hue_order: Sequence[Any] | None = None,
        palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        draw_order: Sequence[Any] | None = None,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        center: float | None = None,
        clip: float | None = 3.0,
        gridsize: int = 50,
        colorbar_label: str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        _data.require_columns(data, x, y, hue)
        if kind not in ("scatter", "hexbin"):
            raise ValueError(f"kind must be 'scatter' or 'hexbin', got {kind!r}")
        self.x, self.y, self.hue, self.kind = x, y, hue, kind
        self.numeric_hue = hue is not None and _data.is_numeric(data, hue)
        self.hue_order = (
            _data.resolve_order(data, hue, hue_order)
            if hue is not None and not self.numeric_hue
            else None
        )
        self.palette = (
            _data.resolve_palette(self.hue_order, palette) if self.hue_order is not None else None
        )
        self.draw_order = list(draw_order) if draw_order is not None else None
        self.cmap = cmap
        self.vmin, self.vmax, self.center, self.clip = vmin, vmax, center, clip
        self.gridsize = gridsize
        self.colorbar_label = colorbar_label
        self.kw = kw

    # ----- layers ---------------------------------------------------------------------

    def highlight(
        self,
        where: pl.Expr,
        *,
        label: str | None = None,
        color: Any = "red",
        marker: str = "o",
        **scatter_kw: Any,
    ) -> Self:
        """Overlay the rows matching ``where`` on top of the plot.

        E.g. ``.highlight(pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC,
        label=fisseq.PATHOGENIC, color="red", marker="^")``.
        """
        zorder = 3 + len(self._layers)  # later highlights draw on top of earlier ones

        def layer(plot: EmbeddingPlot, ax: Axes) -> None:
            subset = plot.data.filter(where)
            kw = {
                "edgecolors": "black",
                "linewidths": 0.5,
                "zorder": zorder,
                **scatter_kw,
            }
            handle = ax.scatter(
                subset.get_column(plot.x).to_numpy(),
                subset.get_column(plot.y).to_numpy(),
                color=color,
                marker=marker,
                label=label,
                **kw,
            )
            if label is not None:
                add_legend_entry(ax, handle, label)

        return self._with_layer(layer)

    # ----- drawing --------------------------------------------------------------------

    def _norm(self, values: np.ndarray) -> tuple[Normalize, str]:
        """Color normalization and colorbar ``extend`` for numeric values."""
        return _data.color_norm(
            values, vmin=self.vmin, vmax=self.vmax, center=self.center, clip=self.clip
        )

    def _draw(self, ax: Axes) -> None:
        if self.kind == "hexbin":
            self._draw_hexbin(ax)
        elif self.numeric_hue:
            self._draw_numeric_scatter(ax)
        else:
            self._draw_categorical_scatter(ax)
        ax.set(xlabel=self.x, ylabel=self.y)

    def _draw_categorical_scatter(self, ax: Axes) -> None:
        df = self.data
        if self.hue is not None:
            order = self.draw_order
            if order is None:
                counts = df.get_column(self.hue).value_counts(name="n")
                sizes = dict(counts.iter_rows())
                order = sorted(self.hue_order or [], key=lambda lvl: -sizes.get(lvl, 0))
            rank = {lvl: i for i, lvl in enumerate(order)}
            df = (
                df.filter(pl.col(self.hue).is_in(list(rank)))
                .with_columns(
                    pl.col(self.hue)
                    .replace_strict(rank, return_dtype=pl.Int64)
                    .alias("__draw_rank")
                )
                .sort("__draw_rank", maintain_order=True)
            )
        sns.scatterplot(
            data=_data.to_pandas(df, self.x, self.y, self.hue),
            x=self.x,
            y=self.y,
            hue=self.hue,
            hue_order=self.hue_order,
            palette=self.palette,
            ax=ax,
            **{"linewidth": 0, "zorder": 1, **self.kw},
        )

    def _draw_numeric_scatter(self, ax: Axes) -> None:
        assert self.hue is not None
        df = self.data.sort(self.hue, nulls_last=False)
        values = df.get_column(self.hue).cast(pl.Float64).to_numpy()
        norm, extend = self._norm(values)
        sc = ax.scatter(
            df.get_column(self.x).to_numpy(),
            df.get_column(self.y).to_numpy(),
            c=values,
            cmap=self.cmap,
            norm=norm,
            **{"s": 15, "edgecolors": "none", "zorder": 1, **self.kw},
        )
        ax.get_figure().colorbar(
            sc, ax=ax, label=self.colorbar_label or self.hue, extend=extend
        )

    def _draw_hexbin(self, ax: Axes) -> None:
        x = self.data.get_column(self.x).to_numpy()
        y = self.data.get_column(self.y).to_numpy()
        kw = {"gridsize": self.gridsize, "edgecolors": "none", "zorder": 1, **self.kw}
        fig = ax.get_figure()
        if self.hue is None:
            hb = ax.hexbin(x, y, cmap=self.cmap or "Greys", vmin=self.vmin, vmax=self.vmax, **kw)
            fig.colorbar(hb, ax=ax, label=self.colorbar_label or "count")
        elif self.numeric_hue:
            values = self.data.get_column(self.hue).cast(pl.Float64).to_numpy()
            norm, extend = self._norm(values)
            keep = np.isfinite(values)
            hb = ax.hexbin(
                x[keep], y[keep], C=values[keep], reduce_C_function=np.mean,
                cmap=self.cmap, norm=norm, **kw,
            )
            fig.colorbar(hb, ax=ax, label=self.colorbar_label or self.hue, extend=extend)
        else:
            levels = self.hue_order or []
            code = {lvl: i for i, lvl in enumerate(levels)}
            codes = np.array(
                [code.get(v, -1) for v in self.data.get_column(self.hue).to_list()]
            )
            keep = codes >= 0
            cmap = ListedColormap([self.palette[lvl] for lvl in levels])  # type: ignore[index]
            norm = Normalize(vmin=-0.5, vmax=len(levels) - 0.5)
            ax.hexbin(
                x[keep], y[keep], C=codes[keep], reduce_C_function=_mode,
                cmap=cmap, norm=norm, **kw,
            )
            cbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                label=self.colorbar_label or self.hue,
            )
            cbar.set_ticks(range(len(levels)), labels=[str(lvl) for lvl in levels])

