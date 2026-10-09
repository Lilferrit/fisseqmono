"""`CalibrationPlot`: control-group score distributions with the evidence-point bands."""

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

from .. import _data
from .._base import Plot
from . import _core
from ._inputs import GROUP_LABELS
from ._mixture import Mixture
from ._result import POINT_ORDER, Calibration

GROUP_PALETTE: dict[str, str] = {
    "P/LP": "red",
    "B/LB": "royalblue",
    "gnomAD": "dimgrey",
    "Synonymous": "darkgreen",
}
_GROUP_INDEX = dict(zip(GROUP_LABELS, range(len(GROUP_LABELS)), strict=True))


def _band_color(point: int) -> Any:
    cmap = plt.get_cmap("Reds" if point > 0 else "Blues")
    return cmap(0.25 + 0.09 * abs(point))


class CalibrationPlot(Plot):
    """Score distributions of the calibration's control groups, over shaded bands for the
    score range of each evidence point (as Fig. 4a of Tejura et al.).

    >>> cal = scores.calibrate("auroc_pooled_corrected", gnomad="gnomad_LMNA.csv")
    >>> fb.CalibrationPlot(cal).save("vis/calibration.png")

    Parameters
    ----------
    data : Dataset | pl.DataFrame
        Usually the dataset `Dataset.calibrate` returned: its ``calibration`` is used and
        its ``<prefix>_groups`` column says which group each variant is in.
    score : str | None
        Score column; defaults to the one the calibration was fit on.
    calibration : Calibration | None
        Overrides ``data.calibration`` (required when ``data`` is a DataFrame).
    groups : Sequence[str]
        Groups to draw, from ``"P/LP"``, ``"B/LB"``, ``"gnomAD"``, ``"Synonymous"``. A group
        the calibration dropped as too small is not drawn.
    kind : {"hist", "kde"}
        Observed distributions as density histograms or Gaussian KDEs.
    bins : int
        Histogram bins over the calibration's score range.
    fit : bool
        Overlay each group's fitted mixture density (median across bootstrap fits).
    bands : bool
        Shade each evidence point's score range, labelled with its points.
    palette : mapping | None
        Colors per group (defaults: P/LP red, B/LB blue, gnomAD grey, synonymous green).
    prefix : str
        Prefix of the calibration columns.
    """

    default_figsize = (7.0, 4.0)

    def __init__(
        self,
        data: Any,
        score: str | None = None,
        *,
        calibration: Calibration | None = None,
        groups: Sequence[str] = GROUP_LABELS,
        kind: str = "hist",
        bins: int = 40,
        fit: bool = True,
        bands: bool = True,
        palette: Mapping[str, Any] | None = None,
        prefix: str = "meta_excalibr",
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
    ) -> None:
        calibration = calibration or getattr(data, "calibration", None)
        if calibration is None:
            raise ValueError(
                "No calibration: pass calibration=, or the dataset returned by "
                ".calibrate() / .apply_calibration()"
            )
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        self.calibration = calibration
        self.score = score or calibration.settings.get("score")
        if self.score is None:
            raise ValueError(
                "Pass score=: the calibration does not record a score column"
            )
        self.groups_col = f"{prefix}_groups"
        _data.require_columns(self.data, self.score, self.groups_col)
        unknown = sorted(set(groups) - set(GROUP_LABELS))
        if unknown:
            raise ValueError(
                f"Unknown groups {unknown}; choose from {list(GROUP_LABELS)}"
            )
        if kind not in ("hist", "kde"):
            raise ValueError(f"kind must be 'hist' or 'kde', got {kind!r}")
        self.groups = list(groups)
        self.kind, self.bins, self.fit, self.bands = kind, bins, fit, bands
        self.palette = {**GROUP_PALETTE, **(palette or {})}

    def _group_scores(self, group: str) -> np.ndarray:
        values = self.data.filter(pl.col(self.groups_col).list.contains(group))[
            self.score
        ]
        values = values.cast(pl.Float64).drop_nulls().drop_nans().to_numpy()
        return values

    def _fitted_density(self, group: str, grid: np.ndarray) -> np.ndarray | None:
        cal = self.calibration
        g = _GROUP_INDEX[group]
        curves = []
        for f in cal.fits:
            mixture = Mixture.from_dict(f)
            weights = mixture.weights[g]
            if np.isnan(weights).any():
                continue
            curves.append(np.exp(mixture.logpdf(grid, weights)))
        return np.median(curves, axis=0) if curves else None

    def _draw_bands(self, ax: Axes, lo: float, hi: float) -> None:
        trans = ax.get_xaxis_transform()
        for point in POINT_ORDER:
            for a, b in self.calibration.point_ranges.get(point, []):
                a, b = max(a, lo), min(b, hi)
                if b <= a:
                    continue
                ax.axvspan(a, b, color=_band_color(point), alpha=0.35, lw=0, zorder=0)
                if (b - a) > 0.025 * (hi - lo):
                    ax.text(
                        (a + b) / 2, 0.98, f"{point:+d}", transform=trans,
                        ha="center", va="top", fontsize="x-small",
                    )  # fmt: skip

    def _draw(self, ax: Axes) -> None:
        cal = self.calibration
        grid = np.asarray(cal.grid)
        lo, hi = float(grid[0]), float(grid[-1])
        if self.bands:
            self._draw_bands(ax, lo, hi)
        edges = np.linspace(lo, hi, self.bins + 1)
        used = cal.group_counts
        for group in self.groups:
            values = self._group_scores(group)
            # Skip empty groups and those the fit dropped as too small.
            if len(values) == 0 or used.get(_core.GROUPS[_GROUP_INDEX[group]], 1) == 0:
                continue
            color = self.palette[group]
            label = f"{group} (n={len(values)})"
            if self.kind == "hist":
                ax.hist(values, bins=edges, density=True, histtype="stepfilled",
                        color=color, alpha=0.25, zorder=1)  # fmt: skip
                ax.hist(values, bins=edges, density=True, histtype="step",
                        color=color, label=label, zorder=2)  # fmt: skip
            elif len(values) > 1 and np.ptp(values) > 0:
                ax.plot(
                    grid, gaussian_kde(values)(grid), color=color, label=label, zorder=2
                )
            if self.fit:
                fitted = self._fitted_density(group, grid)
                if fitted is not None:
                    ax.plot(
                        grid, fitted, color=color, linestyle="--", linewidth=1, zorder=3
                    )
        ax.set_xlim(lo, hi)
        ax.set_xlabel(self.score)
        ax.set_ylabel("Density")
        if ax.get_legend_handles_labels()[0]:
            # Below the band labels along the top edge.
            ax.legend(
                frameon=False,
                fontsize="small",
                loc="upper right",
                bbox_to_anchor=(1, 0.94),
            )
