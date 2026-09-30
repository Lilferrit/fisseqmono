"""Cumulative explained variance and scree plots of a PCA fit."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from matplotlib.axes import Axes

from . import _data
from ._base import Plot


class ExplainedVariancePlot(Plot):
    """Cumulative explained variance (and / or a scree plot) of every principal component,
    with the components kept at chosen thresholds or at the noise floor marked.

    >>> reduced = profiles.pca_reduce(variance=0.9, noise_floor=True)
    >>> fb.ExplainedVariancePlot(reduced, thresholds=[0.7, 0.8, 0.9]).save("vis/pca.png")

    Parameters
    ----------
    data : pl.DataFrame | Profiles
        One row per component, in order, with an ``explained_variance_ratio`` column, such
        as `Profiles.pca_explained_variance`. A `Profiles` returned by
        `Profiles.pca_reduce` is also accepted: its ``pca_explained_variance`` is used, and
        its ``pca_noise_floor`` unless ``noise_floor`` is given.
    kind : {"cumulative", "scree", "both"}
        ``"both"`` draws the scree bars on a second y axis behind the cumulative curve.
    x : {"components", "fraction"}
        Put the number of components on the x axis, or the fraction of all components.
    thresholds : Sequence[float]
        Cumulative-variance ratios to mark: a dashed horizontal line at the ratio and a
        vertical line at the fewest components reaching it (as `Profiles.pca_reduce`
        chooses), labelled e.g. ``"0.9: 37 PCs"``.
    noise_floor : float | None
        Variance ratio of the first component of shuffled data (see
        ``pca_reduce(noise_floor=True)``). Marked as a horizontal line on the scree plot
        and as a vertical line at the number of components above it.
    palette : mapping | str | list | None
        Colors of the threshold lines, keyed by threshold.
    """

    def __init__(
        self,
        data: Any,
        *,
        kind: Literal["cumulative", "scree", "both"] = "cumulative",
        x: Literal["components", "fraction"] = "components",
        thresholds: Sequence[float] = (),
        noise_floor: float | None = None,
        palette: Any = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
    ) -> None:
        if hasattr(data, "pca_explained_variance"):
            if data.pca_explained_variance is None:
                raise ValueError("No PCA fitted; chain .pca_reduce() first")
            if noise_floor is None:
                noise_floor = data.pca_noise_floor
            data = data.pca_explained_variance
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        _data.require_columns(data, "explained_variance_ratio")
        if kind not in ("cumulative", "scree", "both"):
            raise ValueError(f"kind must be 'cumulative', 'scree' or 'both', got {kind!r}")
        if x not in ("components", "fraction"):
            raise ValueError(f"x must be 'components' or 'fraction', got {x!r}")
        self.kind, self.x = kind, x
        self.ratios = data.get_column("explained_variance_ratio").cast(pl.Float64).to_numpy()
        self.thresholds = list(thresholds)
        self.noise_floor = noise_floor
        self.palette = _data.resolve_palette(self.thresholds, palette)

    def n_components(self, threshold: float) -> int:
        """The fewest components whose cumulative explained variance reaches ``threshold``."""
        k = int(np.searchsorted(np.cumsum(self.ratios), threshold - 1e-12)) + 1
        return min(k, len(self.ratios))

    def _positions(self) -> np.ndarray:
        n = np.arange(1, len(self.ratios) + 1)
        return n / len(n) if self.x == "fraction" else n

    def _xpos(self, k: int) -> float:
        return k / len(self.ratios) if self.x == "fraction" else k

    def _draw(self, ax: Axes) -> None:
        pos = self._positions()
        cumulative = np.cumsum(self.ratios)
        xlabel = "Fraction of components" if self.x == "fraction" else "Number of components"
        xmax = 1.0 if self.x == "fraction" else len(pos) + 0.5
        width = (pos[1] - pos[0]) * 0.8 if len(pos) > 1 else 0.8

        if self.kind == "scree":
            ax.bar(pos, self.ratios, width=width, color="grey")
            if self.noise_floor is not None:
                ax.axhline(self.noise_floor, linestyle="--", color="red", linewidth=1,
                           label="noise floor")
            ax.set(ylabel="Explained variance ratio")
        else:
            if self.kind == "both":
                scree_ax = ax.twinx()
                scree_ax.bar(pos, self.ratios, width=width, color="lightgrey", zorder=0)
                if self.noise_floor is not None:
                    scree_ax.axhline(self.noise_floor, linestyle=":", color="red", linewidth=1)
                scree_ax.set_ylabel("Explained variance ratio")
                ax.set_zorder(scree_ax.get_zorder() + 1)
                ax.patch.set_visible(False)
            ax.plot(pos, cumulative, color="black", linewidth=1.5)
            for t in self.thresholds:
                k = self.n_components(t)
                color = self.palette[t]
                ax.axhline(t, linestyle="--", color=color, linewidth=1)
                ax.axvline(self._xpos(k), linestyle="--", color=color, linewidth=1,
                           label=f"{t:g}: {k} PCs")
            if self.noise_floor is not None:
                k = int(np.sum(self.ratios > self.noise_floor))
                ax.axvline(self._xpos(k), linestyle=":", color="red", linewidth=1,
                           label=f"noise floor: {k} PCs")
            ax.set(ylim=(0, 1.01), ylabel="Cumulative explained variance")
        ax.set(xlim=(0, xmax), xlabel=xlabel)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right" if self.kind != "scree" else "upper right")
