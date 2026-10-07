"""Variant effect maps: one value per single substitution, positions by amino acids.

Like the missense maps of the VIS-seq paper: each column is a protein position, each row
the substituted amino acid (`fisseq.AMINO_ACID_ORDER`), and each cell the variant's value,
e.g. a feature's z-score versus the synonymous variants. Missing variants are gray and the
wild-type residue's (synonymous) cell carries a black dot. Around the heatmap sit the
protein regions (above), the mean per position (below) and each amino acid's values
(right). Long proteins can be wrapped into stacked rows.
"""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from . import _data, _variants, fisseq
from ._base import FigurePlot, config

# Layout sizes, in inches.
_CELL = 0.12  # one heatmap cell (a position by an amino acid)
_REGIONS_H = 0.2
_REGIONS_GAP = 0.04
_MEAN_H = 0.6
_MEAN_GAP = 0.06
_MARGINAL_W = 1.0
_MARGINAL_GAP = 0.08
_ROW_GAP = 0.45  # between wrapped rows, for the tick labels
_CBAR_W = 0.12
_CBAR_GAP = 0.15
_LEFT = 0.55
_RIGHT = 0.7
_TOP = 0.1
_BOTTOM = 0.5

_FONTSIZE = 6.5
_AGGREGATES = ("median", "mean", "min", "max", "first")


@dataclasses.dataclass
class VariantEffectMapAxes:
    """The pieces of a drawn `VariantEffectMap`, one entry per row of the figure.

    Attributes
    ----------
    heatmaps : list[Axes]
    regions, means, marginals : list[Axes]
        The region bars, position-mean lines and per-amino-acid strips; empty when that
        panel is turned off.
    colorbar : Axes
    positions : list[tuple[int, int]]
        Each row's inclusive ``(first, last)`` position range.
    """

    heatmaps: list[Axes]
    regions: list[Axes]
    means: list[Axes]
    marginals: list[Axes]
    colorbar: Axes
    positions: list[tuple[int, int]]


def _text_color(color: Any) -> str:
    r, g, b = to_rgb(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


class VariantEffectMap(FigurePlot):
    """Heatmap of ``value`` per single substitution: protein positions on x, the
    substituted amino acid on y.

    Only single substitutions such as ``"A12V"`` (and synonymous ``"A12A"``) are drawn;
    every other label in ``variant_col`` is ignored, as are nonsense (``"A12X"``,
    ``"A12*"``) and null values.

    >>> fb.VariantEffectMap(profiles, "Mean_Nuclei_AreaShape_FormFactor",
    ...                     positions=(178, 273), vmin=-6, vmax=6).save("vis/map.png")

    Parameters
    ----------
    data : pl.DataFrame
    value : str
        Numeric column to color the cells by.
    variant_col : str, default "meta_aa_changes"
        Variant labels.
    aggregate : {"median", "mean", "min", "max", "first"} | None, default "median"
        How to combine several rows of one variant (e.g. one per batch). ``None`` makes
        duplicates an error.
    positions : tuple[int, int] | None
        Inclusive ``(first, last)`` position range. Default: the substituted positions'
        range.
    positions_per_row : int | None
        Wrap the range into stacked rows of this many positions. Default: one row.
    regions : Mapping[str, tuple[int, int]] | None, default `fisseq.LMNA_DOMAIN_REGIONS`
        Protein regions (name -> inclusive range) drawn as a labelled bar above the
        heatmap and coloring the position means. When ranges overlap, the first one
        listed wins. ``None`` hides the bar.
    region_palette : mapping | str | list | None
    position_mean : bool, default True
        Draw the mean of each position's non-synonymous substitutions below the heatmap.
    marginal : bool, default True
        Draw each amino acid's values (points, black line at the median) to the right of
        the heatmap, over that row's positions.
    cmap, vmin, vmax, center, clip
        Color scale. Missing limits come from the data, symmetric around ``center``
        (default 0) and capped at ``clip``. Values beyond the limits get the end colors
        and the colorbar shows an arrow.
    missing_color : color, default "0.6"
        Color of variants with no value.
    wild_type_marker : bool, default True
        Black dot on each position's wild-type (synonymous) cell.
    cbar_label : str | None
        Default: ``value``.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        value: str,
        *,
        variant_col: str = "meta_aa_changes",
        aggregate: str | None = "median",
        positions: tuple[int, int] | None = None,
        positions_per_row: int | None = None,
        regions: Mapping[str, tuple[int, int]] | None = fisseq.LMNA_DOMAIN_REGIONS,
        region_palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        position_mean: bool = True,
        marginal: bool = True,
        cmap: str | Colormap = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        center: float | None = 0.0,
        clip: float | None = None,
        missing_color: Any = "0.6",
        wild_type_marker: bool = True,
        cbar_label: str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        _data.require_columns(data, variant_col, value)
        if not _data.is_numeric(data, value):
            raise ValueError(f"value column {value!r} must be numeric")
        if aggregate is not None and aggregate not in _AGGREGATES:
            raise ValueError(
                f"aggregate must be one of {_AGGREGATES} or None, got {aggregate!r}"
            )
        if positions_per_row is not None and positions_per_row < 1:
            raise ValueError(
                f"positions_per_row must be positive, got {positions_per_row}"
            )
        self.value, self.variant_col = value, variant_col

        wt, position, mut = _variants.substitution_exprs(variant_col)
        aas = fisseq.AMINO_ACID_ORDER
        cells = data.select(
            wt.alias("wt"),
            position.alias("position"),
            mut.alias("mut"),
            pl.col(value).cast(pl.Float64).alias("value"),
        ).filter(
            pl.col("wt").is_in(aas)
            & pl.col("mut").is_in(aas)
            & pl.col("value").is_not_null()
            & pl.col("value").is_not_nan()
        )
        if cells.is_empty():
            raise ValueError(
                f"No single substitutions with a {value!r} value in {variant_col!r}"
            )
        if positions is None:
            positions = (cells["position"].min(), cells["position"].max())  # type: ignore[assignment]
        first, last = positions  # type: ignore[misc]
        if first > last:
            raise ValueError(f"positions must be (first, last), got {positions}")
        cells = cells.filter(pl.col("position").is_between(first, last))
        keys = ["wt", "position", "mut"]
        if aggregate is None:
            dupes = cells.filter(pl.struct(keys).is_duplicated())
            if not dupes.is_empty():
                raise ValueError(
                    f"{dupes.height} rows share a variant; pass aggregate= to combine them"
                )
        else:
            cells = cells.group_by(keys).agg(getattr(pl.col("value"), aggregate)())
        #: One row per drawn variant: ``wt, position, mut, value``.
        self.cells = cells.sort("position", "mut")
        self.positions = (int(first), int(last))
        self.positions_per_row = positions_per_row

        self.regions = dict(regions) if regions is not None else None
        self.region_palette = (
            _data.resolve_palette(list(self.regions), region_palette)
            if self.regions
            else None
        )
        self.position_mean, self.marginal = position_mean, marginal
        self.cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        self.vmin, self.vmax, self.center, self.clip = vmin, vmax, center, clip
        self.missing_color = missing_color
        self.wild_type_marker = wild_type_marker
        self.cbar_label = value if cbar_label is None else cbar_label

    # ----- data -----------------------------------------------------------------------

    def matrix(self) -> pd.DataFrame:
        """The drawn values: amino acids (`fisseq.AMINO_ACID_ORDER`) by every position in
        range, NaN where a variant is missing."""
        first, last = self.positions
        aas = fisseq.AMINO_ACID_ORDER
        mat = np.full((len(aas), last - first + 1), np.nan)
        rows = self.cells["mut"].replace_strict({aa: i for i, aa in enumerate(aas)})
        mat[rows.to_numpy(), (self.cells["position"] - first).to_numpy()] = self.cells[
            "value"
        ].to_numpy()
        return pd.DataFrame(
            mat,
            index=pd.Index(aas, name="mut"),
            columns=pd.Index(range(first, last + 1), name="position"),
        )

    def wild_type(self) -> dict[int, str]:
        """Wild-type residue per position, read from the substitution labels (the most
        common one if labels disagree)."""
        counts = (
            self.cells.group_by("position", "wt")
            .len()
            .sort("position", "len", "wt", descending=[False, True, False])
            .unique("position", keep="first", maintain_order=True)
        )
        return dict(zip(counts["position"].to_list(), counts["wt"].to_list()))

    def position_means(self) -> pd.Series:
        """Mean of each position's non-synonymous substitutions, NaN where it has none."""
        first, last = self.positions
        means = (
            self.cells.filter(pl.col("wt") != pl.col("mut"))
            .group_by("position")
            .agg(pl.col("value").mean())
        )
        series = pd.Series(
            means["value"].to_numpy(), index=means["position"].to_numpy(), dtype=float
        )
        return series.reindex(range(first, last + 1)).rename_axis("position")

    def _region_of(self, position: int) -> str | None:
        for name, (lo, hi) in (self.regions or {}).items():
            if lo <= position <= hi:
                return name
        return None

    def _chunks(self) -> list[tuple[int, int]]:
        first, last = self.positions
        per_row = self.positions_per_row or last - first + 1
        return [
            (start, min(start + per_row - 1, last))
            for start in range(first, last + 1, per_row)
        ]

    # ----- drawing --------------------------------------------------------------------

    def _draw_figure(self) -> tuple[Figure, VariantEffectMapAxes]:
        mat = self.matrix()
        values = mat.to_numpy()
        norm, extend = _data.color_norm(
            values, vmin=self.vmin, vmax=self.vmax, center=self.center, clip=self.clip
        )
        cmap = self.cmap.with_extremes(bad=self.missing_color)
        chunks = self._chunks()
        per_row = max(hi - lo + 1 for lo, hi in chunks)

        # --- layout, in inches, from the bottom left --------------------------------
        heat_w, heat_h = per_row * _CELL, len(mat.index) * _CELL
        above = _REGIONS_H + _REGIONS_GAP if self.regions else 0.0
        below = _MEAN_H + _MEAN_GAP if self.position_mean else 0.0
        row_h = above + heat_h + below
        width = (
            _LEFT
            + heat_w
            + (_MARGINAL_GAP + _MARGINAL_W if self.marginal else 0.0)
            + _CBAR_GAP
            + _CBAR_W
            + _RIGHT
        )
        height = _TOP + len(chunks) * row_h + (len(chunks) - 1) * _ROW_GAP + _BOTTOM
        fig = plt.figure(
            figsize=self.figsize or (width, height), dpi=self.dpi or config.dpi
        )

        def add(x: float, y: float, w: float, h: float, **kw: Any) -> Axes:
            return fig.add_axes((x / width, y / height, w / width, h / height), **kw)

        handle = VariantEffectMapAxes([], [], [], [], None, chunks)  # type: ignore[arg-type]
        wild_type = self.wild_type() if self.wild_type_marker else {}
        means = self.position_means() if self.position_mean else None
        for i, (lo, hi) in enumerate(chunks):
            top = height - _TOP - i * (row_h + _ROW_GAP)
            heat_y = top - above - heat_h
            ax = add(_LEFT, heat_y, heat_w, heat_h)
            self._draw_heatmap(ax, mat.loc[:, lo:hi], cmap, norm, wild_type, per_row)
            handle.heatmaps.append(ax)
            if self.regions:
                rax = add(_LEFT, heat_y + heat_h + _REGIONS_GAP, heat_w, _REGIONS_H)
                rax.sharex(ax)
                self._draw_regions(rax, lo, hi)
                handle.regions.append(rax)
            if means is not None:
                share = {"sharey": handle.means[0]} if handle.means else {}
                max_ = add(
                    _LEFT, heat_y - _MEAN_GAP - _MEAN_H, heat_w, _MEAN_H, **share
                )
                max_.sharex(ax)
                self._draw_means(max_, means.loc[lo:hi], lo, hi)
                ax.tick_params(labelbottom=False)
                handle.means.append(max_)
            if self.marginal:
                share = {"sharex": handle.marginals[0]} if handle.marginals else {}
                x = _LEFT + heat_w + _MARGINAL_GAP
                max_ = add(x, heat_y, _MARGINAL_W, heat_h, sharey=ax, **share)
                self._draw_marginal(max_, mat.loc[:, lo:hi].to_numpy())
                handle.marginals.append(max_)
        bottom = (handle.means or handle.heatmaps)[-1]
        bottom.set_xlabel("position", fontsize=_FONTSIZE + 1)

        cbar_x = width - _RIGHT - _CBAR_W
        top_y = height - _TOP - above - heat_h
        handle.colorbar = add(cbar_x, top_y, _CBAR_W, heat_h)
        cbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=handle.colorbar,
            extend=extend,
            extendfrac=0.06,
        )
        cbar.set_label(self.cbar_label, fontsize=_FONTSIZE + 1)
        handle.colorbar.tick_params(labelsize=_FONTSIZE)
        return fig, handle

    def _draw_heatmap(
        self,
        ax: Axes,
        mat: pd.DataFrame,
        cmap: Colormap,
        norm: Any,
        wild_type: dict[int, str],
        per_row: int,
    ) -> None:
        lo, hi = int(mat.columns[0]), int(mat.columns[-1])
        n_aa = len(mat.index)
        ax.imshow(
            np.ma.masked_invalid(mat.to_numpy()),
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            extent=(lo - 0.5, hi + 0.5, n_aa - 0.5, -0.5),
        )
        rows = {aa: i for i, aa in enumerate(mat.index)}
        dots = [(p, rows[aa]) for p, aa in wild_type.items() if lo <= p <= hi]
        if dots:
            px, py = zip(*dots)
            ax.scatter(px, py, s=3, color="black", linewidths=0, zorder=3)
        # a short last row keeps the cell width; its unused width stays blank
        ax.set_xlim(lo - 0.5, lo + per_row - 0.5)
        ax.set_ylim(n_aa - 0.5, -0.5)
        for side in ("top", "bottom"):
            ax.spines[side].set_bounds(lo - 0.5, hi + 0.5)
        ax.spines["right"].set_position(("data", hi + 0.5))
        ax.set_yticks(range(n_aa), list(mat.index))
        # every 5 positions, none past the row's last one (shared with the means axes)
        ax.set_xticks(range(lo + (-lo) % 5, hi + 1, 5))
        ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1)
        ax.tick_params(axis="y", length=0)

    def _draw_regions(self, ax: Axes, lo: int, hi: int) -> None:
        assert self.regions is not None and self.region_palette is not None
        ax.set_ylim(0, 1)
        ax.axis("off")
        claimed: set[int] = set()  # overlapping ranges: the first region listed wins
        for name, (start, end) in self.regions.items():
            span = [
                p for p in range(max(start, lo), min(end, hi) + 1) if p not in claimed
            ]
            if not span:
                continue
            claimed.update(span)
            x0, x1 = span[0] - 0.5, span[-1] + 0.5
            color = self.region_palette[name]
            ax.axvspan(x0, x1, color=color, linewidth=0)
            if (x1 - x0) * _CELL >= len(name) * _FONTSIZE * 0.6 / 72:
                ax.text(
                    (x0 + x1) / 2,
                    0.5,
                    name,
                    ha="center",
                    va="center",
                    fontsize=_FONTSIZE,
                    color=_text_color(color),
                )

    def _draw_means(self, ax: Axes, means: pd.Series, lo: int, hi: int) -> None:
        x = means.index.to_numpy()
        y = means.to_numpy()
        ax.plot(x, y, color="black", linewidth=0.7)  # NaNs break the line at gaps
        ok = np.isfinite(y)
        if self.region_palette:
            colors = [
                self.region_palette.get(self._region_of(int(p)), "black")  # type: ignore[arg-type]
                for p in x[ok]
            ]
        else:
            colors = ["black"] * int(ok.sum())
        ax.scatter(x[ok], y[ok], s=5, c=colors, linewidths=0, zorder=3)
        ax.hlines(0, lo - 0.5, hi + 0.5, color="0.5", linestyle="--", linewidth=0.6)
        for side in ("top", "bottom"):
            ax.spines[side].set_bounds(lo - 0.5, hi + 0.5)
        ax.spines["right"].set_position(("data", hi + 0.5))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1)
        ax.set_ylabel("mean", fontsize=_FONTSIZE)

    def _draw_marginal(self, ax: Axes, mat: np.ndarray) -> None:
        rng = np.random.default_rng(0)
        for i, row in enumerate(mat):
            vals = row[np.isfinite(row)]
            if not vals.size:
                continue
            jitter = rng.uniform(-0.3, 0.3, vals.size)
            ax.scatter(vals, i + jitter, s=1, color="black", alpha=0.5, linewidths=0)
            median = float(np.median(vals))
            ax.plot([median, median], [i - 0.4, i + 0.4], color="black", linewidth=1)
        ax.axvline(0, color="0.5", linewidth=0.5)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1, labelleft=False)
        ax.tick_params(axis="y", length=0)

    def _main_ax(self, handle: VariantEffectMapAxes) -> Axes:
        return handle.heatmaps[0]
