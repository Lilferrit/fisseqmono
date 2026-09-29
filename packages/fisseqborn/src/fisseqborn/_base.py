"""Base classes shared by every plot.

A plot is configured once in ``__init__`` (data + column names + drawing options), then
optionally extended with chained *layers* (reference lines, stat annotations, highlighted
points, ``ax.set`` calls). Each chained method returns a new plot, so a base plot can be
reused, e.g. inside a loop, without accumulating layers. Nothing is drawn until
`Plot.plot` or `Plot.save` is called.
"""

import copy
import dataclasses
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from os import PathLike
from typing import Any, Self

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from . import _data


@dataclasses.dataclass
class Config:
    """Package-wide defaults; modify ``fisseqborn.config`` to change them."""

    dpi: int = 150
    figsize: tuple[float, float] = (5.0, 5.0)


config = Config()

Layer = Callable[[Any, Axes], None]


class Plot(ABC):
    """An axes-level plot: draws onto a single `Axes`."""

    #: Figure size used when the subclass has a better default than ``config.figsize``.
    default_figsize: tuple[float, float] | None = None

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
    ) -> None:
        self.data = _data.as_frame(data)
        self.title = title
        self.figsize = figsize
        self.dpi = dpi
        self._layers: list[Layer] = []
        self._figure: Figure | None = None

    # ----- chaining -------------------------------------------------------------------

    def _with_layer(self, layer: Layer) -> Self:
        new = copy.copy(self)
        new._layers = [*self._layers, layer]
        new._figure = None
        return new

    def refline(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        diagonal: bool = False,
        **line_kw: Any,
    ) -> Self:
        """Add dashed black reference lines: vertical at ``x``, horizontal at ``y`` and/or
        the ``y = x`` diagonal. ``line_kw`` overrides the line style."""
        kw = {"linestyle": "--", "color": "black", "linewidth": 1, **line_kw}

        def layer(_plot: Plot, ax: Axes) -> None:
            if x is not None:
                ax.axvline(x, **kw)
            if y is not None:
                ax.axhline(y, **kw)
            if diagonal:
                ax.axline((0, 0), slope=1, **kw)

        return self._with_layer(layer)

    def set(self, **ax_kwargs: Any) -> Self:
        """Apply ``ax.set(**ax_kwargs)`` after drawing, e.g. ``.set(ylim=(0, 1), xlabel="...")``."""
        return self._with_layer(lambda _plot, ax: ax.set(**ax_kwargs))

    def legend(self, *, outside: bool = False, **legend_kw: Any) -> Self:
        """Restyle the legend. ``outside=True`` places it to the right of the axes."""
        if outside:
            legend_kw = {"loc": "center left", "bbox_to_anchor": (1.0, 0.5), **legend_kw}

        def layer(_plot: Plot, ax: Axes) -> None:
            if ax.get_legend() is not None:
                ax.legend(**legend_kw)

        return self._with_layer(layer)

    # ----- rendering ------------------------------------------------------------------

    @abstractmethod
    def _draw(self, ax: Axes) -> None:
        """Draw the plot's main content onto ``ax``."""

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Axes]:
        """Draw the plot (and all layers). Creates a new figure unless ``ax`` is given."""
        created = ax is None
        if ax is None:
            fig, ax = plt.subplots(
                figsize=self.figsize or self.default_figsize or config.figsize,
                dpi=self.dpi or config.dpi,
            )
        else:
            fig = ax.get_figure()
        self._draw(ax)
        for layer in self._layers:
            layer(self, ax)
        if self.title is not None:
            ax.set_title(self.title)
        if created:
            fig.tight_layout()
        self._figure = fig
        return fig, ax

    def save(self, path: str | PathLike, **savefig_kw: Any) -> Self:
        """Save the figure (drawing it first if needed), creating parent directories."""
        if self._figure is None:
            self.plot()
        assert self._figure is not None
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(path, **{"bbox_inches": "tight", **savefig_kw})
        return self

    def show(self) -> None:
        """Draw (if needed) and display the figure."""
        if self._figure is None:
            self.plot()
        plt.show()


class FigurePlot(Plot):
    """A figure-level plot (e.g. a seaborn ``ClusterGrid``) that owns its whole figure.

    Layers are applied to the axes returned by `_main_ax`.
    """

    @abstractmethod
    def _draw_figure(self) -> tuple[Figure, Any]:
        """Create and draw the figure; return it and the plot's grid/handle object."""

    @abstractmethod
    def _main_ax(self, handle: Any) -> Axes:
        """The axes that layers are applied to."""

    def _draw(self, ax: Axes) -> None:  # pragma: no cover - figure-level plots don't use it
        raise NotImplementedError

    def plot(self, ax: Axes | None = None) -> tuple[Figure, Any]:  # type: ignore[override]
        if ax is not None:
            raise TypeError(
                f"{type(self).__name__} is figure-level and cannot draw onto an existing Axes"
            )
        fig, handle = self._draw_figure()
        main_ax = self._main_ax(handle)
        for layer in self._layers:
            layer(self, main_ax)
        if self.title is not None:
            fig.suptitle(self.title, y=1.02)
        self._figure = fig
        return fig, handle
