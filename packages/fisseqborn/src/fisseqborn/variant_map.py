"""Variant effect maps: one value per single substitution, positions by amino acids.

Like the missense maps of the VIS-seq paper: each column is a protein position, each row
the substituted amino acid (`fisseq.AMINO_ACID_ORDER`), and each cell the variant's value,
e.g. a feature's z-score versus the synonymous variants. Missing variants are gray and the
wild-type residue's (synonymous) cell carries a black dot. Around the heatmap sit the
protein regions (above) and the mean per position (below), plus optionally each amino
acid's values (right). Long proteins can be wrapped into stacked rows, chosen variants
marked with `VariantEffectMap.highlight`, and the whole map turned on its side with
``orientation="portrait"``.
"""

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from . import _data, _variants, fisseq
from ._base import FigurePlot, config

# Layout sizes, in inches. Landscape: positions run left to right and the panels stack
# vertically; portrait swaps the two directions.
_CELL = 0.12  # one heatmap cell (a position by an amino acid)
_REGIONS_H = 0.2
_REGIONS_GAP = 0.04
_MEAN_H = 0.6
_MEAN_GAP = 0.06
_MARGINAL_W = 1.0
_MARGINAL_GAP = 0.08
_ROW_GAP = 0.45  # between wrapped rows (or columns), for the tick labels
_CBAR_W = 0.12
_CBAR_GAP = 0.15
_LEFT = 0.55
_RIGHT = 0.7
_TOP = 0.1
_BOTTOM = 0.5
_PORTRAIT_TOP = 0.4  # amino-acid and mean tick labels sit above the panels
_PORTRAIT_BOTTOM = 0.15

_FONTSIZE = 6.5
_AGGREGATES = ("median", "mean", "min", "max", "first")
_ORIENTATIONS = ("landscape", "portrait")


@dataclasses.dataclass
class VariantEffectMapAxes:
    """The pieces of a drawn `VariantEffectMap`, one entry per row (landscape) or column
    (portrait) of the figure.

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


@dataclasses.dataclass(frozen=True)
class _Highlight:
    where: pl.Expr
    label: str | None
    color: Any
    marker: str
    hue: str | None
    palette: Any
    missing_color: Any
    scatter_kw: dict[str, Any]


def _text_color(color: Any) -> str:
    r, g, b = to_rgb(color)
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"


class VariantEffectMap(FigurePlot):
    """Heatmap of ``value`` per single substitution: protein positions on x, the
    substituted amino acid on y (swapped with ``orientation="portrait"``).

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
        Wrap the range into stacked rows (side-by-side columns in portrait) of this many
        positions. Default: one row.
    orientation : {"landscape", "portrait"}, default "landscape"
        ``"portrait"`` turns the map on its side: positions run down the page, amino
        acids across the top, the region bar on the left and the position means on the
        right.
    regions : Mapping[str, tuple[int, int]] | None, default `fisseq.LMNA_DOMAIN_REGIONS`
        Protein regions (name -> inclusive range) drawn as a labelled bar above the
        heatmap and coloring the position means. When ranges overlap, the first one
        listed wins. ``None`` hides the bar.
    region_palette : mapping | str | list | None
    position_mean : bool, default True
        Draw the mean of each position's non-synonymous substitutions below the heatmap.
    marginal : bool, default False
        Draw each amino acid's values (jittered points, black line at the median) to the
        right of the heatmap (below it in portrait), over that row's positions.
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
        orientation: str = "landscape",
        regions: Mapping[str, tuple[int, int]] | None = fisseq.LMNA_DOMAIN_REGIONS,
        region_palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        position_mean: bool = True,
        marginal: bool = False,
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
        if orientation not in _ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {_ORIENTATIONS}, got {orientation!r}"
            )
        self.value, self.variant_col = value, variant_col

        cells = self._substitutions(data, pl.col(value).cast(pl.Float64).alias("value"))
        cells = cells.filter(
            pl.col("value").is_not_null() & pl.col("value").is_not_nan()
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
        self.orientation = orientation

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
        self._highlights: list[_Highlight] = []

    # ----- layers ---------------------------------------------------------------------

    def highlight(
        self,
        where: pl.Expr,
        *,
        label: str | None = None,
        color: Any = "red",
        marker: str = "o",
        hue: str | None = None,
        palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        missing_color: Any = "white",
        **scatter_kw: Any,
    ) -> Self:
        """Mark the cells of the variants matching ``where`` with a marker, on every row.

        E.g. ``.highlight(pl.col("meta_clinvar_annotation") == fisseq.PATHOGENIC,
        label=fisseq.PATHOGENIC, color="black", marker="^")``. A cell is marked when any of
        its variant's rows matches (rows are matched before ``aggregate``), whether or not
        the variant has a value. Later highlights draw on top of earlier ones.

        Parameters
        ----------
        where : pl.Expr
            Rows to mark.
        label : str | None
            Legend entry for the marks (one entry, even when colored by ``hue``), shown
            below the colorbar.
        color : color
            Marker color, without ``hue``.
        marker : str
        hue : str | None
            Categorical column to color the marks by instead of ``color``. A variant whose
            matching rows disagree takes its first non-null level.
        palette : mapping | str | list | None
            Colors for ``hue``'s levels. Default: the fisseq / seaborn default palette.
        missing_color : color, default "white"
            Color of marks whose ``hue`` level has no color in ``palette``.
        **scatter_kw
            Passed to ``ax.scatter``.
        """
        _data.require_columns(self.data, hue)
        spec = _Highlight(
            where, label, color, marker, hue, palette, missing_color, scatter_kw
        )
        new = copy.copy(self)
        new._highlights = [*self._highlights, spec]
        new._figure = None
        return new

    # ----- data -----------------------------------------------------------------------

    def _substitutions(self, data: pl.DataFrame, *extra: pl.Expr) -> pl.DataFrame:
        """``wt, position, mut`` (plus ``extra``) of ``data``'s single substitutions."""
        wt, position, mut = _variants.substitution_exprs(self.variant_col)
        aas = fisseq.AMINO_ACID_ORDER
        return data.select(
            wt.alias("wt"), position.alias("position"), mut.alias("mut"), *extra
        ).filter(pl.col("wt").is_in(aas) & pl.col("mut").is_in(aas))

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

    def _highlight_cells(self, spec: _Highlight) -> tuple[pl.DataFrame, Any]:
        """The ``position, mut`` cells a highlight marks, and their colors."""
        extra = [] if spec.hue is None else [pl.col(spec.hue)]
        rows = self._substitutions(self.data.filter(spec.where), *extra).filter(
            pl.col("position").is_between(*self.positions)
        )
        if spec.hue is None:
            return rows.unique(["position", "mut"], maintain_order=True), spec.color
        cells = rows.group_by("position", "mut", maintain_order=True).agg(
            pl.col(spec.hue).drop_nulls().first()
        )
        colors = _data.highlight_colors(
            cells, spec.hue, spec.palette, spec.missing_color
        )
        return cells, colors

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

    @property
    def _portrait(self) -> bool:
        return self.orientation == "portrait"

    def _cell_xy(self, positions: Any, rows: Any) -> tuple[Any, Any]:
        """Data coordinates of the cells at ``positions`` and amino-acid ``rows``."""
        return (rows, positions) if self._portrait else (positions, rows)

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
        n = len(chunks)

        # --- layout, in inches from the bottom left. "along" is the position direction
        # (a row's length), "across" the stacking of the panels around one heatmap.
        heat_along, heat_across = per_row * _CELL, len(mat.index) * _CELL
        regions = _REGIONS_H + _REGIONS_GAP if self.regions else 0.0
        means = _MEAN_H + _MEAN_GAP if self.position_mean else 0.0
        strip = regions + heat_across + means
        marginal = _MARGINAL_GAP + _MARGINAL_W if self.marginal else 0.0
        if self._portrait:
            width = (
                _LEFT + n * strip + (n - 1) * _ROW_GAP + _CBAR_GAP + _CBAR_W + _RIGHT
            )
            height = _PORTRAIT_TOP + heat_along + marginal + _PORTRAIT_BOTTOM
        else:
            width = _LEFT + heat_along + marginal + _CBAR_GAP + _CBAR_W + _RIGHT
            height = _TOP + n * strip + (n - 1) * _ROW_GAP + _BOTTOM
        fig = plt.figure(
            figsize=self.figsize or (width, height), dpi=self.dpi or config.dpi
        )

        def add(x: float, y: float, w: float, h: float, **kw: Any) -> Axes:
            return fig.add_axes((x / width, y / height, w / width, h / height), **kw)

        def rect(offset: float, thickness: float, length: float) -> tuple:
            """(x, y, w, h) of the current strip's panel ``offset`` inches in from its
            region-bar edge, ``thickness`` across and ``length`` along the positions."""
            if self._portrait:
                top = height - _PORTRAIT_TOP
                return (x0 + offset, top - length, thickness, length)
            return (_LEFT, y0 - offset - thickness, length, thickness)

        handle = VariantEffectMapAxes([], [], [], [], None, chunks)  # type: ignore[arg-type]
        wild_type = self.wild_type() if self.wild_type_marker else {}
        position_means = self.position_means() if self.position_mean else None
        for i, (lo, hi) in enumerate(chunks):
            x0 = _LEFT + i * (strip + _ROW_GAP)  # portrait: the strip's left edge
            y0 = height - _TOP - i * (strip + _ROW_GAP)  # landscape: its top edge
            ax = add(*rect(regions, heat_across, heat_along))
            self._draw_heatmap(ax, mat.loc[:, lo:hi], cmap, norm, wild_type, per_row)
            handle.heatmaps.append(ax)
            if self.regions:
                rax = add(*rect(0.0, _REGIONS_H, heat_along))
                (rax.sharey if self._portrait else rax.sharex)(ax)
                self._draw_regions(rax, lo, hi)
                handle.regions.append(rax)
            if position_means is not None:
                share_ax = handle.means[0] if handle.means else None
                share = {"sharex" if self._portrait else "sharey": share_ax}
                max_ = add(
                    *rect(regions + heat_across + _MEAN_GAP, _MEAN_H, heat_along),
                    **(share if share_ax else {}),
                )
                (max_.sharey if self._portrait else max_.sharex)(ax)
                self._draw_means(max_, position_means.loc[lo:hi], lo, hi)
                if not self._portrait:
                    ax.tick_params(labelbottom=False)
                handle.means.append(max_)
            if self.marginal:
                share_ax = handle.marginals[0] if handle.marginals else None
                if self._portrait:
                    top = height - _PORTRAIT_TOP - heat_along - _MARGINAL_GAP
                    box = (x0 + regions, top - _MARGINAL_W, heat_across, _MARGINAL_W)
                    kw = {"sharex": ax, **({"sharey": share_ax} if share_ax else {})}
                else:
                    heat_y = y0 - regions - heat_across
                    box = (
                        _LEFT + heat_along + _MARGINAL_GAP,
                        heat_y,
                        _MARGINAL_W,
                        heat_across,
                    )
                    kw = {"sharey": ax, **({"sharex": share_ax} if share_ax else {})}
                max_ = add(*box, **kw)
                self._draw_marginal(max_, mat.loc[:, lo:hi].to_numpy())
                handle.marginals.append(max_)
            self._draw_highlights(ax, lo, hi)
        if self._portrait:
            label_ax = (handle.regions or handle.heatmaps)[0]
            label_ax.set_ylabel("position", fontsize=_FONTSIZE + 1)
        else:
            bottom = (handle.means or handle.heatmaps)[-1]
            bottom.set_xlabel("position", fontsize=_FONTSIZE + 1)

        # colorbar: right of the figure, top-aligned with the (first) heatmap
        cbar_x = width - _RIGHT - _CBAR_W
        if self._portrait:
            cbar_h = min(heat_along, heat_across)
            cbar_y = height - _PORTRAIT_TOP - cbar_h
        else:
            cbar_h = heat_across
            cbar_y = height - _TOP - regions - heat_across
        handle.colorbar = add(cbar_x, cbar_y, _CBAR_W, cbar_h)
        cbar = fig.colorbar(
            ScalarMappable(norm=norm, cmap=cmap),
            cax=handle.colorbar,
            extend=extend,
            extendfrac=0.06,
        )
        cbar.set_label(self.cbar_label, fontsize=_FONTSIZE + 1)
        handle.colorbar.tick_params(labelsize=_FONTSIZE)
        self._draw_legend(fig, (cbar_x / width, (cbar_y - 0.15) / height))
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
        image = np.ma.masked_invalid(mat.to_numpy())
        if self._portrait:
            image, extent = image.T, (-0.5, n_aa - 0.5, hi + 0.5, lo - 0.5)
        else:
            extent = (lo - 0.5, hi + 0.5, n_aa - 0.5, -0.5)
        ax.imshow(
            image,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        rows = {aa: i for i, aa in enumerate(mat.index)}
        dots = [(p, rows[aa]) for p, aa in wild_type.items() if lo <= p <= hi]
        if dots:
            px, py = zip(*dots)
            ax.scatter(
                *self._cell_xy(px, py), s=3, color="black", linewidths=0, zorder=3
            )
        # a short last row keeps the cell width; its unused length stays blank
        along, end = (lo - 0.5, lo + per_row - 0.5), hi + 0.5
        aa_lim = (n_aa - 0.5, -0.5)
        # every 5 positions, none past the row's last one (shared with the side panels)
        ticks = range(lo + (-lo) % 5, hi + 1, 5)
        if self._portrait:
            ax.set_xlim(-0.5, n_aa - 0.5)
            ax.set_ylim(along[1], along[0])
            for side in ("left", "right"):
                ax.spines[side].set_bounds(lo - 0.5, end)
            ax.spines["bottom"].set_position(("data", end))
            ax.set_xticks(range(n_aa), list(mat.index))
            ax.set_yticks(ticks)
            ax.xaxis.tick_top()
            ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1)
            ax.tick_params(axis="x", length=0)
            if self.regions:
                ax.tick_params(axis="y", labelleft=False, length=0)
        else:
            ax.set_xlim(*along)
            ax.set_ylim(*aa_lim)
            for side in ("top", "bottom"):
                ax.spines[side].set_bounds(lo - 0.5, end)
            ax.spines["right"].set_position(("data", end))
            ax.set_yticks(range(n_aa), list(mat.index))
            ax.set_xticks(ticks)
            ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1)
            ax.tick_params(axis="y", length=0)

    def _draw_regions(self, ax: Axes, lo: int, hi: int) -> None:
        assert self.regions is not None and self.region_palette is not None
        if self._portrait:
            # keep the y axis: it carries the position ticks, outside the bar
            ax.set_xlim(0, 1)
            ax.xaxis.set_visible(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1)
        else:
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
            p0, p1 = span[0] - 0.5, span[-1] + 0.5
            color = self.region_palette[name]
            (ax.axhspan if self._portrait else ax.axvspan)(
                p0, p1, color=color, linewidth=0
            )
            if (p1 - p0) * _CELL >= len(name) * _FONTSIZE * 0.6 / 72:
                ax.text(
                    *self._cell_xy((p0 + p1) / 2, 0.5),
                    name,
                    ha="center",
                    va="center",
                    rotation=90 if self._portrait else 0,
                    fontsize=_FONTSIZE,
                    color=_text_color(color),
                )

    def _draw_means(self, ax: Axes, means: pd.Series, lo: int, hi: int) -> None:
        x = means.index.to_numpy()
        y = means.to_numpy()
        ok = np.isfinite(y)
        if self.region_palette:
            colors = [
                self.region_palette.get(self._region_of(int(p)), "black")  # type: ignore[arg-type]
                for p in x[ok]
            ]
        else:
            colors = ["black"] * int(ok.sum())
        end = hi + 0.5
        if self._portrait:
            ax.plot(y, x, color="black", linewidth=0.7)  # NaNs break the line at gaps
            ax.scatter(y[ok], x[ok], s=5, c=colors, linewidths=0, zorder=3)
            ax.vlines(0, lo - 0.5, end, color="0.5", linestyle="--", linewidth=0.6)
            for side in ("left", "right"):
                ax.spines[side].set_bounds(lo - 0.5, end)
            ax.spines["bottom"].set_position(("data", end))
            ax.xaxis.set_major_locator(MaxNLocator(2))  # a narrow panel
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.tick_params(axis="y", labelleft=False, length=0)
            ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1)
            ax.set_xlabel("mean", fontsize=_FONTSIZE)
        else:
            ax.plot(x, y, color="black", linewidth=0.7)  # NaNs break the line at gaps
            ax.scatter(x[ok], y[ok], s=5, c=colors, linewidths=0, zorder=3)
            ax.hlines(0, lo - 0.5, end, color="0.5", linestyle="--", linewidth=0.6)
            for side in ("top", "bottom"):
                ax.spines[side].set_bounds(lo - 0.5, end)
            ax.spines["right"].set_position(("data", end))
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
            median = float(np.median(vals))
            if self._portrait:
                ax.scatter(
                    i + jitter, vals, s=1, color="black", alpha=0.5, linewidths=0
                )
                ax.plot(
                    [i - 0.4, i + 0.4], [median, median], color="black", linewidth=1
                )
            else:
                ax.scatter(
                    vals, i + jitter, s=1, color="black", alpha=0.5, linewidths=0
                )
                ax.plot(
                    [median, median], [i - 0.4, i + 0.4], color="black", linewidth=1
                )
        values_axis, aa_axis = ("y", "x") if self._portrait else ("x", "y")
        (ax.axhline if self._portrait else ax.axvline)(0, color="0.5", linewidth=0.5)
        getattr(ax, f"{values_axis}axis").set_major_locator(MaxNLocator(3))
        ax.tick_params(labelsize=_FONTSIZE, length=2, pad=1)
        ax.tick_params(axis=aa_axis, length=0)
        if self._portrait:
            ax.tick_params(labelbottom=False, labeltop=False)
        else:
            ax.tick_params(labelleft=False)

    def _draw_highlights(self, ax: Axes, lo: int, hi: int) -> None:
        rows = {aa: i for i, aa in enumerate(fisseq.AMINO_ACID_ORDER)}
        for i, spec in enumerate(self._highlights):
            cells, colors = self._highlight_cells(spec)
            keep = cells["position"].is_between(lo, hi).to_numpy()
            if not keep.any():
                continue
            positions = cells["position"].to_numpy()[keep]
            aa_rows = np.array([rows[aa] for aa in cells["mut"].to_list()])[keep]
            if spec.hue is not None:
                colors = [c for c, k in zip(colors, keep) if k]
            ax.scatter(
                *self._cell_xy(positions, aa_rows),
                color=colors,
                marker=spec.marker,
                **{
                    "s": 10,
                    "edgecolors": "black",
                    "linewidths": 0.3,
                    "zorder": 4 + i,  # later highlights draw on top
                    **spec.scatter_kw,
                },
            )

    def _draw_legend(self, fig: Figure, anchor: tuple[float, float]) -> None:
        handles, labels = [], []
        for spec in self._highlights:
            if spec.label is None:
                continue
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="",
                    marker=spec.marker,
                    markerfacecolor="lightgrey" if spec.hue else spec.color,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    markersize=4,
                )
            )
            labels.append(spec.label)
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=anchor,
                frameon=False,
                fontsize=_FONTSIZE,
                handletextpad=0.2,
                borderaxespad=0,
            )

    def _main_ax(self, handle: VariantEffectMapAxes) -> Axes:
        return handle.heatmaps[0]
