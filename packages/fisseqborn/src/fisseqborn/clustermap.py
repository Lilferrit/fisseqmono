"""Clustered heatmaps built from groups of features, each with its own color scale.

The DataFrame rows (e.g. one per cluster) are ordered by hierarchical clustering on the
features of the groups marked ``cluster=True``; the other groups are drawn alongside in the
same order but never influence it. With ``orientation="horizontal"`` each DataFrame row is
a heatmap row and the groups sit side by side; ``"vertical"`` transposes that, so each
DataFrame row is a heatmap column and the groups are stacked. For example, a per-cluster summary can order its rows
by landmark-feature z-scores and variant-class proportions while showing the median
distinguishability score next to them.
"""

import dataclasses
import logging
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist

from . import _data
from ._base import FigurePlot, config

logger = logging.getLogger(__name__)

# Layout sizes, in inches.
_DENDRO_W = 1.2
_ROW_COLORS_W = 0.25
_GAP_W = 0.2
_MIN_GROUP_W = 0.9  # so a narrow group's title has room
_CBAR_H = 0.12
_CBAR_TICKS_H = 0.3  # colorbar tick labels, between the title and the colorbar
_CBAR_GAP_H = 0.1
_COL_COLORS_H = 0.2
_LABEL_MARGIN = 1.6  # rotated feature / row labels
_TITLE_W = 1.4  # vertical layout: the title column left of the colorbars
_CBAR_TICKS_W = 0.4  # vertical layout: colorbar tick labels, left of the colorbar
_MINI_CBAR = 0.7  # per-feature colorbars fill this fraction of their cell's length
_EDGE_MARGIN = 0.1

# Group titles, in points.
_TITLE_FONTSIZES = (10.0, 8.0, 7.0)  # tried in turn until every word fits the block
_TITLE_LINE = 1.25  # line height, as a multiple of the font size
_CHAR_W = 0.6  # average character width, as a multiple of the font size


Limit = float | Mapping[str, float] | None


@dataclasses.dataclass(frozen=True)
class FeatureGroup:
    """A block of feature columns in a `ClusterMap`, drawn with one shared color scale,
    or with one color scale and colorbar per feature.

    A group is drawn *per feature* when ``palette`` is set, ``cmap`` is a mapping, or any
    limit is a mapping (see `per_feature`). Each feature then gets a small colorbar next
    to its own heatmap column (row, when vertical).

    Parameters
    ----------
    name : str
        Title shown above the block's colorbar (left of it when vertical).
    features : str | Sequence[str] | polars selector
        The group's columns.
    cluster : bool, default True
        Use these features when clustering the rows. Groups with ``cluster=False`` are
        only displayed, in the row order the other groups determine.
    cluster_features : bool, default False
        Reorder the features within this block by clustering them; otherwise they keep
        the given order.
    cmap : str | Colormap | Mapping[str, str | Colormap] | None
        Default: ``"RdBu_r"`` when ``center`` is set, else ``"viridis"``. A mapping gives
        each listed feature its own colormap (and its own colorbar); unlisted features use
        ``palette`` or the default.
    vmin, vmax, center, clip : float | Mapping[str, float] | None
        Color limits. With ``center``, missing limits are symmetric around it, with the
        half-width capped at ``clip``. A mapping sets them per feature (giving each feature
        its own scale); in a per-feature group a scalar applies to every feature, and a
        missing limit comes from that feature's own values.
    labels : bool | Mapping[str, str] | None
        Show feature names under the block (right of it when vertical). A mapping gives
        display names (e.g. `fisseq.LMNA_LANDMARK_FEATURES`). Default: shown when the
        block has <= 40 features.
    annot : bool
        Write each cell's value on it, formatted with ``fmt``.
    palette : Mapping[str, color] | str | Sequence | bool | None
        Give each feature its own sequential white -> color scale and colorbar, so each
        feature (e.g. each domain or variant class) is drawn in its own hue. A mapping
        gives the feature colors, a string or list a seaborn palette (e.g. ``"tab10"``),
        and ``True`` the default: fisseq colors when every feature is a known level
        (e.g. ``"Synonymous"``), else seaborn's.
    """

    name: str
    features: Any
    cluster: bool = True
    cluster_features: bool = False
    cmap: str | Colormap | Mapping[str, str | Colormap] | None = None
    vmin: Limit = None
    vmax: Limit = None
    center: Limit = None
    clip: Limit = None
    labels: bool | Mapping[str, str] | None = None
    annot: bool = False
    fmt: str = ".2f"
    palette: Mapping[str, Any] | str | Sequence[Any] | bool | None = None

    @property
    def per_feature(self) -> bool:
        """Whether each feature gets its own color scale and colorbar."""
        return (
            (self.palette is not None and self.palette is not False)
            or isinstance(self.cmap, Mapping)
            or any(isinstance(v, Mapping) for v in (self.vmin, self.vmax, self.center, self.clip))
        )


def _limit(value: Limit, feature: str) -> float | None:
    return value.get(feature) if isinstance(value, Mapping) else value


def _as_cmap(cmap: str | Colormap) -> Colormap:
    return (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(bad="lightgrey")


def _color_scales(
    group: FeatureGroup, feats: list[str], mat: np.ndarray
) -> list[tuple[Colormap, Normalize, str]]:
    """``(cmap, norm, colorbar extend)`` per feature column of ``mat``; the same scale
    for every feature unless the group is `per_feature`."""
    def default_cmap(center: float | None) -> str | Colormap:
        if group.cmap is not None and not isinstance(group.cmap, Mapping):
            return group.cmap
        return "RdBu_r" if center is not None else "viridis"

    if not group.per_feature:
        center = _limit(group.center, "")
        norm, extend = _data.color_norm(
            mat, vmin=_limit(group.vmin, ""), vmax=_limit(group.vmax, ""),
            center=center, clip=_limit(group.clip, ""),
        )
        return [(_as_cmap(default_cmap(center)), norm, extend)] * len(feats)

    palette: dict[str, Any] = {}
    if group.palette is not None and group.palette is not False:
        palette = _data.resolve_palette(feats, None if group.palette is True else group.palette)
    cmaps = group.cmap if isinstance(group.cmap, Mapping) else {}
    scales = []
    for j, f in enumerate(feats):
        center = _limit(group.center, f)
        if f in cmaps:
            cmap: str | Colormap = cmaps[f]
        elif f in palette:
            cmap = LinearSegmentedColormap.from_list(f, ["white", palette[f]])
        else:
            cmap = default_cmap(center)
        norm, extend = _data.color_norm(
            mat[:, j], vmin=_limit(group.vmin, f), vmax=_limit(group.vmax, f),
            center=center, clip=_limit(group.clip, f),
        )
        scales.append((_as_cmap(cmap), norm, extend))
    return scales


def _rgba(mat: np.ndarray, scales: list[tuple[Colormap, Normalize, str]]) -> np.ndarray:
    """Colors of ``mat`` (rows x features), each column through its own scale."""
    rgba = np.empty((*mat.shape, 4))
    for j, (cmap, norm, _) in enumerate(scales):
        rgba[:, j] = cmap(norm(np.ma.masked_invalid(mat[:, j])))
    return rgba


@dataclasses.dataclass
class ClusterMapAxes:
    """The pieces of a drawn `ClusterMap`.

    "Row" always means a DataFrame row, whichever way the heatmap is drawn.

    Attributes
    ----------
    heatmap_axes, colorbar_axes : dict[str, Axes]
        Per group name.
    title_axes : dict[str, Axes]
        Per group with a name: the (axis-less) axes holding its title.
    feature_colorbar_axes : dict[str, dict[str, Axes]]
        Per `FeatureGroup.per_feature` group, each feature's own colorbar. Those groups
        have no entry in ``colorbar_axes``.
    row_dendrogram_ax, row_colors_ax : Axes | None
        The DataFrame rows' dendrogram and color strip: left of the heatmap when
        horizontal, above it when vertical.
    row_order : list[int]
        Positions of the input rows in plotting order: top to bottom when horizontal,
        left to right when vertical.
    feature_order : dict[str, list[str]]
        Per group name, the features left to right (horizontal) or top to bottom
        (vertical).
    row_linkage : np.ndarray | None
        The scipy linkage used to order the rows (``None`` if no group clusters).
    orientation : {"horizontal", "vertical"}
    """

    heatmap_axes: dict[str, Axes]
    colorbar_axes: dict[str, Axes]
    title_axes: dict[str, Axes]
    feature_colorbar_axes: dict[str, dict[str, Axes]]
    row_dendrogram_ax: Axes | None
    row_colors_ax: Axes | None
    row_order: list[int]
    feature_order: dict[str, list[str]]
    row_linkage: np.ndarray | None
    orientation: str = "horizontal"


def _resolve_features(data: pl.DataFrame, features: Any) -> list[str]:
    if isinstance(features, str):
        features = [features]
    if isinstance(features, Sequence):
        _data.require_columns(data, *features)
        return list(features)
    return data.select(features).columns


def _text_width(text: str, fontsize: float) -> float:
    """Estimated width of ``text`` in inches."""
    return len(text) * _CHAR_W * fontsize / 72


def _longest_word_width(title: str) -> float:
    """Width a block needs so every word of ``title`` fits at the smallest title font."""
    return max((_text_width(w, _TITLE_FONTSIZES[-1]) for w in title.split()), default=0.0)


def _wrap_title(title: str, width_in: float) -> tuple[list[str], float]:
    """Wrap ``title`` to a block ``width_in`` inches wide, shrinking the font until every
    word fits on a line. Returns the lines and the font size."""
    for fontsize in _TITLE_FONTSIZES:
        chars = max(1, int(width_in * 72 / (_CHAR_W * fontsize)))
        if all(len(word) <= chars for word in title.split()):
            break
    return textwrap.wrap(title, chars, break_long_words=False) or [""], fontsize


def _title_height(lines: list[str], fontsize: float) -> float:
    return len(lines) * fontsize * _TITLE_LINE / 72


def _finite_columns(data: pl.DataFrame, columns: list[str]) -> list[bool]:
    return list(
        data.select(
            pl.col(c).cast(pl.Float64).is_finite().fill_null(False).all() for c in columns
        ).row(0)
    )


class ClusterMap(FigurePlot):
    """Hierarchically clustered heatmap of the DataFrame rows (e.g. one per cluster).

    Pass ``groups`` to split the features into blocks with their own color scales and
    to choose which blocks drive the row clustering. Without ``groups`` every feature
    column forms one clustered block (``features``, ``cmap``, ``vmin``, ``vmax`` and
    ``center`` configure it).

    Throughout, "row" means a DataFrame row (``row_colors``, ``row_labels``,
    `row_order`, ...). With ``orientation="vertical"`` those are drawn as heatmap
    columns.

    Parameters
    ----------
    data : pl.DataFrame
    groups : Sequence[FeatureGroup] | None
        Feature blocks, left to right (horizontal) or top to bottom (vertical).
    features : polars selector | Sequence[str] | None
        Without ``groups``: the matrix columns. Default: every column not starting with
        ``meta_``.
    orientation : {"horizontal", "vertical"}, default "horizontal"
        ``"horizontal"``: one heatmap row per DataFrame row, groups side by side, the
        dendrogram on the left and row labels on the right. ``"vertical"``: one heatmap
        column per DataFrame row, groups stacked with their features as heatmap rows
        (labelled on the right), the dendrogram along the top, row labels under the
        bottom group, and each group's title and colorbar on its left.
    row_colors : str | None
        Categorical column drawn as a color strip along the DataFrame rows (beside the
        heatmap rows, or above the heatmap columns when vertical), with a legend.
    row_palette : mapping | str | list | None
    row_labels : str | None
        Column whose values label the DataFrame rows (y tick labels on the right, or x
        tick labels at the bottom when vertical). Default: no row labels.
    col_colors : mapping | None
        Maps feature name -> category, drawn as a strip along the features (above them,
        or left of them when vertical).
    col_palette : mapping | str | list | None
    method, metric : str
        `scipy.cluster.hierarchy.linkage` method and `scipy.spatial.distance.pdist`
        metric, used for rows and for ``cluster_features``.
    standardize : bool, default False
        z-score each clustering feature across rows before computing distances, so
        groups on different scales (z-scores vs. proportions) weigh in comparably.
        Zero-variance features are then dropped from the clustering input. Only the
        clustering is affected, not the colors.
    drop_nonfinite : bool, default True
        Drop clustering features (``cluster=True`` or ``cluster_features=True``) that
        contain null / NaN / inf, which clustering can't handle. Display-only features
        keep missing values, drawn in light grey.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        groups: Sequence[FeatureGroup] | None = None,
        features: Any = None,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        row_colors: str | None = None,
        row_palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        row_labels: str | None = None,
        col_colors: Mapping[str, Any] | None = None,
        col_palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        method: str = "average",
        metric: str = "euclidean",
        standardize: bool = False,
        drop_nonfinite: bool = True,
        cmap: str | Colormap = "RdBu_r",
        vmin: float | None = -3,
        vmax: float | None = 3,
        center: float | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        if groups is None:
            groups = [
                FeatureGroup(
                    "",
                    ~cs.starts_with("meta_") if features is None else features,
                    cluster_features=True,
                    cmap=cmap, vmin=vmin, vmax=vmax, center=center, labels=False,
                )
            ]
        elif features is not None:
            raise ValueError("Pass either `features` or `groups`, not both")
        if orientation not in ("horizontal", "vertical"):
            raise ValueError(
                f"orientation must be 'horizontal' or 'vertical', got {orientation!r}"
            )
        self.orientation = orientation
        _data.require_columns(data, row_colors, row_labels)

        names = [g.name for g in groups]
        if len(set(names)) != len(names):
            raise ValueError(f"Feature group names must be unique, got {names}")
        owner: dict[str, str] = {}
        self.groups: list[tuple[FeatureGroup, list[str]]] = []
        for group in groups:
            cols = _resolve_features(data, group.features)
            for c in cols:
                if c in owner:
                    raise ValueError(
                        f"Feature {c!r} is in both groups {owner[c]!r} and {group.name!r}"
                    )
                owner[c] = group.name
            non_numeric = [c for c in cols if not _data.is_numeric(data, c)]
            if non_numeric:
                raise ValueError(f"Non-numeric features in group {group.name!r}: {non_numeric}")
            if drop_nonfinite and (group.cluster or group.cluster_features) and cols:
                ok = _finite_columns(data, cols)
                n_dropped = ok.count(False)
                if n_dropped:
                    logger.warning(
                        "Dropping %d feature column(s) with null/NaN/inf values from group %r",
                        n_dropped, group.name,
                    )
                cols = [c for c, keep in zip(cols, ok) if keep]
            if not cols:
                raise ValueError(f"Feature group {group.name!r} has no usable feature columns")
            self.groups.append((group, cols))

        self.features = [c for _, cols in self.groups for c in cols]
        self.row_colors, self.row_labels = row_colors, row_labels
        self.row_levels = _data.resolve_order(data, row_colors) if row_colors else None
        self.row_palette = (
            _data.resolve_palette(self.row_levels, row_palette) if self.row_levels else None
        )
        self.col_colors = dict(col_colors) if col_colors else None
        self.col_palette = (
            _data.resolve_palette(
                sorted(set(self.col_colors.values()), key=_data._natural_key), col_palette
            )
            if self.col_colors
            else None
        )
        self.method, self.metric = method, metric
        self.standardize = standardize

    # ----- clustering -----------------------------------------------------------------

    def _matrix(self, columns: list[str]) -> np.ndarray:
        return self.data.select(pl.col(columns).cast(pl.Float64)).to_numpy()

    def row_linkage(self) -> np.ndarray | None:
        """Linkage over the ``cluster=True`` groups' features, or ``None`` if there are
        none (the rows then keep their input order)."""
        cols = [c for group, gcols in self.groups if group.cluster for c in gcols]
        if not cols or self.data.height < 2:
            return None
        x = self._matrix(cols)
        if not np.isfinite(x).all():
            raise ValueError(
                "Clustering features contain null/NaN/inf values; use drop_nonfinite=True "
                "or move those features to a group with cluster=False"
            )
        if self.standardize:
            std = x.std(axis=0)
            keep = std > 0
            if not keep.all():
                logger.warning(
                    "Ignoring %d zero-variance feature(s) when clustering rows",
                    int((~keep).sum()),
                )
            if not keep.any():
                return None
            x = (x[:, keep] - x[:, keep].mean(axis=0)) / std[keep]
        return linkage(pdist(x, metric=self.metric), method=self.method)

    def row_order(self) -> list[int]:
        """Positions of the input rows in plotting order, top to bottom."""
        z = self.row_linkage()
        return list(range(self.data.height)) if z is None else leaves_list(z).tolist()

    def _feature_order(self, group: FeatureGroup, cols: list[str]) -> list[str]:
        if not group.cluster_features or len(cols) < 2:
            return cols
        z = linkage(pdist(self._matrix(cols).T, metric=self.metric), method=self.method)
        return [cols[i] for i in leaves_list(z)]

    # ----- drawing --------------------------------------------------------------------

    @staticmethod
    def _show_labels(group: FeatureGroup, n: int) -> bool:
        return bool(group.labels) if group.labels is not None else n <= 40

    def _draw_figure(self) -> tuple[Figure, ClusterMapAxes]:
        z = self.row_linkage()
        order = list(range(self.data.height)) if z is None else leaves_list(z).tolist()
        n_rows = self.data.height
        vertical = self.orientation == "vertical"
        feature_orders = {g.name: self._feature_order(g, cols) for g, cols in self.groups}

        # --- layout, in inches -------------------------------------------------------
        # "Outer" tracks run along the groups: the dendrogram, the row colors and the
        # groups themselves (x when horizontal, y when vertical). "Inner" tracks run
        # across them: title, colorbar ticks, colorbar, gap, col colors and the heatmap.
        if vertical:
            units = [len(cols) for _, cols in self.groups]
            span = min(max(4.0, 0.3 * sum(units)), 14.0)
            titles = [_wrap_title(g.name, _TITLE_W) if g.name else ([], 10.0)
                      for g, _ in self.groups]
            group_sizes = [
                max(span * u / sum(units), _title_height(*t) + 0.1)
                for u, t in zip(units, titles)
            ]
        else:
            units = [max(len(cols), 2) for _, cols in self.groups]
            span = min(max(6.0, 0.3 * sum(units)), 16.0)
            group_sizes = [
                max(span * u / sum(units), _MIN_GROUP_W, _longest_word_width(g.name))
                for u, (g, _) in zip(units, self.groups)
            ]
            titles = [_wrap_title(g.name, w) if g.name else ([], 10.0)
                      for (g, _), w in zip(self.groups, group_sizes)]

        outer: list[float] = []
        dendro_i = rc_i = None
        if z is not None:
            dendro_i = len(outer)
            outer.append(_DENDRO_W)
        if self.row_colors is not None:
            rc_i = len(outer)
            outer.append(_ROW_COLORS_W)
        group_is = []
        for i, size in enumerate(group_sizes):
            if i:
                outer.append(_GAP_W)
            group_is.append(len(outer))
            outer.append(size)

        if vertical:
            heat_size = min(max(6.0, 0.45 * n_rows), 16.0)
            title_size = _TITLE_W if any(g.name for g, _ in self.groups) else 0.0
            ticks_size = _CBAR_TICKS_W
        else:
            heat_size = min(max(4.0, 0.2 * n_rows), 12.0) if self.row_labels else 8.0
            title_size = max(_title_height(*t) for t in titles)
            ticks_size = _CBAR_TICKS_H
        inner: list[float] = []
        title_j = None
        if title_size:
            title_j = len(inner)
            inner.append(title_size)
        inner += [ticks_size, _CBAR_H, _CBAR_GAP_H]
        cbar_j = len(inner) - 2
        cc_j = None
        if self.col_colors is not None:
            cc_j = len(inner)
            inner.append(_COL_COLORS_H)
        heat_j = len(inner)
        inner.append(heat_size)

        any_labels = any(self._show_labels(g, len(c)) for g, c in self.groups)
        if vertical:
            widths, heights = inner, outer
            left, top = _EDGE_MARGIN, _EDGE_MARGIN
            right = _LABEL_MARGIN if any_labels else _EDGE_MARGIN
            bottom = _LABEL_MARGIN if self.row_labels else _EDGE_MARGIN
        else:
            widths, heights = outer, inner
            left, top = _EDGE_MARGIN, _EDGE_MARGIN
            right = _LABEL_MARGIN if self.row_labels else _EDGE_MARGIN
            bottom = _LABEL_MARGIN if (any_labels or self.row_colors) else _EDGE_MARGIN
        fig_w = sum(widths) + left + right
        fig_h = sum(heights) + top + bottom
        if self.figsize is not None:
            fig_w, fig_h = self.figsize
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=self.dpi or config.dpi)
        gs = fig.add_gridspec(
            len(heights), len(widths), width_ratios=widths, height_ratios=heights,
            left=left / fig_w, right=1 - right / fig_w,
            bottom=bottom / fig_h, top=1 - top / fig_h, wspace=0, hspace=0,
        )

        def cell(i: int, j: int) -> Any:
            return gs[i, j] if vertical else gs[j, i]

        # --- row dendrogram and row colors --------------------------------------------
        dendro_ax = rc_ax = None
        if z is not None and dendro_i is not None:
            dendro_ax = fig.add_subplot(cell(dendro_i, heat_j))
            dendrogram(z, orientation="top" if vertical else "left", ax=dendro_ax,
                       no_labels=True, color_threshold=0, above_threshold_color="black")
            for coll in dendro_ax.collections:
                coll.set_linewidth(0.75 if n_rows <= 100 else 0.4)
            if vertical:
                dendro_ax.set_xlim(0, 10 * n_rows)
            else:
                dendro_ax.set_ylim(10 * n_rows, 0)
            dendro_ax.axis("off")
        if self.row_colors is not None and rc_i is not None:
            rc_ax = fig.add_subplot(cell(rc_i, heat_j))
            values = self.data.get_column(self.row_colors).to_list()
            rgb = np.array([[to_rgb(self.row_palette.get(values[i], "white"))]  # type: ignore[union-attr]
                            for i in order])
            if vertical:
                rc_ax.imshow(rgb.transpose(1, 0, 2), aspect="auto", interpolation="nearest")
                rc_ax.set_yticks([0], [self.row_colors])
                rc_ax.set_xticks([])
            else:
                rc_ax.imshow(rgb, aspect="auto", interpolation="nearest")
                rc_ax.set_xticks([0], [self.row_colors], rotation=90)
                rc_ax.set_yticks([])

        # --- one heatmap block per group ----------------------------------------------
        heatmap_axes: dict[str, Axes] = {}
        cbar_axes: dict[str, Axes] = {}
        title_axes: dict[str, Axes] = {}
        feature_cbar_axes: dict[str, dict[str, Axes]] = {}
        last_name = self.groups[-1][0].name
        row_labels = (
            [str(v) for v in self.data.get_column(self.row_labels).to_list()]
            if self.row_labels is not None
            else None
        )
        for (group, _), gi, (lines, fontsize) in zip(self.groups, group_is, titles):
            feats = feature_orders[group.name]
            mat = self._matrix(feats)[order]
            shown = mat.T if vertical else mat
            ax = fig.add_subplot(cell(gi, heat_j))
            scales = _color_scales(group, feats, mat)
            rgba = _rgba(mat, scales)
            if vertical:
                rgba = rgba.transpose(1, 0, 2)
            ax.imshow(rgba, aspect="auto", interpolation="nearest")
            if group.annot:
                self._annotate(ax, shown, rgba, group.fmt)

            feature_ticks: tuple[Any, ...] = ([],)
            if self._show_labels(group, len(feats)):
                names = group.labels if isinstance(group.labels, Mapping) else {}
                feature_ticks = (range(len(feats)), [names.get(f, f) for f in feats])
            row_ticks: tuple[Any, ...] = ([],)
            if row_labels is not None and group.name == last_name:
                row_ticks = (range(n_rows), [row_labels[i] for i in order])
            if vertical:
                ax.yaxis.tick_right()
                ax.set_yticks(*feature_ticks)
                ax.set_xticks(*row_ticks, **({"rotation": 90} if len(row_ticks) > 1 else {}))
            else:
                ax.set_xticks(*feature_ticks, **({"rotation": 90} if len(feature_ticks) > 1 else {}))
                ax.yaxis.tick_right()
                ax.set_yticks(*row_ticks)

            if group.per_feature:
                feature_cbar_axes[group.name] = self._feature_colorbars(
                    fig, cell(gi, cbar_j), feats, scales, vertical
                )
            else:
                cmap, norm, extend = scales[0]
                cax = fig.add_subplot(cell(gi, cbar_j))
                fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                             orientation="vertical" if vertical else "horizontal", extend=extend)
                axis = cax.yaxis if vertical else cax.xaxis
                axis.set_ticks_position("left" if vertical else "top")
                axis.set_major_locator(MaxNLocator(3))
                cax.tick_params(labelsize=8)
                cbar_axes[group.name] = cax
            if group.name and title_j is not None:
                tax = fig.add_subplot(cell(gi, title_j))
                tax.axis("off")
                text_kw = (
                    {"x": 1.0, "y": 0.5, "ha": "right", "va": "center", "multialignment": "right"}
                    if vertical
                    else {"x": 0.5, "y": 0.0, "ha": "center", "va": "bottom"}
                )
                tax.text(s="\n".join(lines), fontsize=fontsize, linespacing=_TITLE_LINE / 1.2,
                         transform=tax.transAxes, **text_kw)
                title_axes[group.name] = tax

            if self.col_colors is not None and cc_j is not None:
                cc_ax = fig.add_subplot(cell(gi, cc_j))
                rgb = np.array([[to_rgb(self.col_palette.get(self.col_colors.get(f), "white"))  # type: ignore[union-attr]
                                 for f in feats]])
                cc_ax.imshow(rgb.transpose(1, 0, 2) if vertical else rgb, aspect="auto",
                             interpolation="nearest")
                cc_ax.set_xticks([])
                cc_ax.set_yticks([])
            heatmap_axes[group.name] = ax

        self._add_legends(fig)
        result = ClusterMapAxes(
            heatmap_axes=heatmap_axes, colorbar_axes=cbar_axes, title_axes=title_axes,
            feature_colorbar_axes=feature_cbar_axes,
            row_dendrogram_ax=dendro_ax, row_colors_ax=rc_ax, row_order=order,
            feature_order=feature_orders, row_linkage=z, orientation=self.orientation,
        )
        return fig, result

    @staticmethod
    def _feature_colorbars(
        fig: Figure,
        spec: Any,
        feats: list[str],
        scales: list[tuple[Colormap, Normalize, str]],
        vertical: bool,
    ) -> dict[str, Axes]:
        """One small colorbar per feature, each centered on its own heatmap cell.

        The colorbar cell is split into three slots per feature with no spacing, so each
        feature's middle slot is centered on its heatmap column (row, when vertical) and
        the colorbars stay aligned with the cells whatever the figure size.
        """
        pad = (1 - _MINI_CBAR) / 2
        ratios = [pad, _MINI_CBAR, pad] * len(feats)
        if vertical:
            sub = spec.subgridspec(3 * len(feats), 1, height_ratios=ratios, hspace=0)
        else:
            sub = spec.subgridspec(1, 3 * len(feats), width_ratios=ratios, wspace=0)
        axes: dict[str, Axes] = {}
        for j, (f, (cmap, norm, extend)) in enumerate(zip(feats, scales)):
            cax = fig.add_subplot(sub[3 * j + 1, 0] if vertical else sub[0, 3 * j + 1])
            cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                                orientation="vertical" if vertical else "horizontal",
                                extend=extend, extendfrac=0.15)
            axis = cax.yaxis if vertical else cax.xaxis
            axis.set_ticks_position("left" if vertical else "top")
            if norm.vmin is not None and norm.vmax is not None:
                cbar.set_ticks([norm.vmin, norm.vmax], labels=[
                    format(norm.vmin, ".2g"), format(norm.vmax, ".2g")
                ])
            # rotated when horizontal, so neighbouring features' labels don't collide
            cax.tick_params(labelsize=6, length=2, pad=1, labelrotation=0 if vertical else 90)
            axes[f] = cax
        return axes

    @staticmethod
    def _annotate(ax: Axes, mat: np.ndarray, rgba: np.ndarray, fmt: str) -> None:
        for (i, j), v in np.ndenumerate(mat):
            if not np.isfinite(v):
                continue
            r, g, b, _ = rgba[i, j]
            color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6.5, color=color)

    def _add_legends(self, fig: Figure) -> None:
        legends: list[tuple[str | None, dict[Any, Any]]] = []
        if self.row_palette:
            legends.append((self.row_colors, self.row_palette))
        if self.col_palette:
            legends.append((None, self.col_palette))
        y = 1.0
        for title, palette in legends:
            handles = [Patch(facecolor=c, label=str(k)) for k, c in palette.items()]
            fig.legend(handles=handles, title=title, loc="upper left",
                       bbox_to_anchor=(1.0, y), frameon=False)
            y -= 0.03 * (len(handles) + 2)

    def _main_ax(self, handle: ClusterMapAxes) -> Axes:
        return next(iter(handle.heatmap_axes.values()))
