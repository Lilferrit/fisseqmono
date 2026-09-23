"""Clustered heatmaps built from groups of features, each with its own color scale.

The rows (one per DataFrame row) are ordered by hierarchical clustering on the features
of the groups marked ``cluster=True``; the other groups are drawn alongside in the same
row order but never influence it. For example, a per-cluster summary can order its rows
by landmark-feature z-scores and variant-class proportions while showing the median
distinguishability score next to them.
"""

import dataclasses
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize, to_rgb
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
_CBAR_H = 0.12
_CBAR_GAP_H = 0.1
_COL_COLORS_H = 0.2
_TOP_MARGIN = 0.75  # colorbar ticks + group titles
_LABEL_MARGIN = 1.6  # rotated feature / row labels
_EDGE_MARGIN = 0.1


@dataclasses.dataclass(frozen=True)
class FeatureGroup:
    """A block of feature columns in a `ClusterMap`, drawn with one shared color scale.

    Parameters
    ----------
    name : str
        Title shown above the block's colorbar.
    features : str | Sequence[str] | polars selector
        The group's columns.
    cluster : bool, default True
        Use these features when clustering the rows. Groups with ``cluster=False`` are
        only displayed, in the row order the other groups determine.
    cluster_features : bool, default False
        Reorder the features within this block by clustering them; otherwise they keep
        the given order.
    cmap : str | Colormap | None
        Default: ``"RdBu_r"`` when ``center`` is set, else ``"viridis"``.
    vmin, vmax, center, clip
        Color limits. With ``center``, missing limits are symmetric around it, with the
        half-width capped at ``clip``.
    labels : bool | Mapping[str, str] | None
        Show feature names under the block. A mapping gives display names (e.g.
        `fisseq.LMNA_LANDMARK_FEATURES`). Default: shown when the block has <= 40 features.
    annot : bool
        Write each cell's value on it, formatted with ``fmt``.
    """

    name: str
    features: Any
    cluster: bool = True
    cluster_features: bool = False
    cmap: str | Colormap | None = None
    vmin: float | None = None
    vmax: float | None = None
    center: float | None = None
    clip: float | None = None
    labels: bool | Mapping[str, str] | None = None
    annot: bool = False
    fmt: str = ".2f"


@dataclasses.dataclass
class ClusterMapAxes:
    """The pieces of a drawn `ClusterMap`.

    Attributes
    ----------
    heatmap_axes, colorbar_axes : dict[str, Axes]
        Per group name.
    row_dendrogram_ax, row_colors_ax : Axes | None
    row_order : list[int]
        Positions of the input rows, top to bottom.
    feature_order : dict[str, list[str]]
        Per group name, the features left to right.
    row_linkage : np.ndarray | None
        The scipy linkage used to order the rows (``None`` if no group clusters).
    """

    heatmap_axes: dict[str, Axes]
    colorbar_axes: dict[str, Axes]
    row_dendrogram_ax: Axes | None
    row_colors_ax: Axes | None
    row_order: list[int]
    feature_order: dict[str, list[str]]
    row_linkage: np.ndarray | None


def _resolve_features(data: pl.DataFrame, features: Any) -> list[str]:
    if isinstance(features, str):
        features = [features]
    if isinstance(features, Sequence):
        _data.require_columns(data, *features)
        return list(features)
    return data.select(features).columns


def _finite_columns(data: pl.DataFrame, columns: list[str]) -> list[bool]:
    return list(
        data.select(
            pl.col(c).cast(pl.Float64).is_finite().fill_null(False).all() for c in columns
        ).row(0)
    )


class ClusterMap(FigurePlot):
    """Hierarchically clustered heatmap, one row per DataFrame row.

    Pass ``groups`` to split the features into blocks with their own color scales and
    to choose which blocks drive the row clustering. Without ``groups`` every feature
    column forms one clustered block (``features``, ``cmap``, ``vmin``, ``vmax`` and
    ``center`` configure it).

    Parameters
    ----------
    data : pl.DataFrame
    groups : Sequence[FeatureGroup] | None
        Feature blocks, left to right.
    features : polars selector | Sequence[str] | None
        Without ``groups``: the matrix columns. Default: every column not starting with
        ``meta_``.
    row_colors : str | None
        Categorical column drawn as a color strip beside the rows, with a legend.
    row_palette : mapping | str | list | None
    row_labels : str | None
        Column whose values label the rows (on the right). Default: no row labels.
    col_colors : mapping | None
        Maps feature name -> category, drawn as a strip above the features.
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
        feature_orders = {g.name: self._feature_order(g, cols) for g, cols in self.groups}

        # --- layout, in inches -------------------------------------------------------
        units = [max(len(cols), 2) for _, cols in self.groups]
        heat_w = min(max(6.0, 0.3 * sum(units)), 16.0)
        widths: list[float] = []
        dendro_col = rc_col = None
        if z is not None:
            dendro_col = len(widths)
            widths.append(_DENDRO_W)
        if self.row_colors is not None:
            rc_col = len(widths)
            widths.append(_ROW_COLORS_W)
        group_cols = []
        for i, u in enumerate(units):
            if i:
                widths.append(_GAP_W)
            group_cols.append(len(widths))
            widths.append(heat_w * u / sum(units))

        heat_h = min(max(4.0, 0.2 * n_rows), 12.0) if self.row_labels else 8.0
        heights = [_CBAR_H, _CBAR_GAP_H]
        cc_row = None
        if self.col_colors is not None:
            cc_row = len(heights)
            heights.append(_COL_COLORS_H)
        heat_row = len(heights)
        heights.append(heat_h)

        any_labels = any(self._show_labels(g, len(c)) for g, c in self.groups)
        left, right = _EDGE_MARGIN, _LABEL_MARGIN if self.row_labels else _EDGE_MARGIN
        top = _TOP_MARGIN
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

        # --- row dendrogram and row colors --------------------------------------------
        dendro_ax = rc_ax = None
        if z is not None and dendro_col is not None:
            dendro_ax = fig.add_subplot(gs[heat_row, dendro_col])
            dendrogram(z, orientation="left", ax=dendro_ax, no_labels=True,
                       color_threshold=0, above_threshold_color="black")
            for coll in dendro_ax.collections:
                coll.set_linewidth(0.75 if n_rows <= 100 else 0.4)
            dendro_ax.set_ylim(10 * n_rows, 0)
            dendro_ax.axis("off")
        if self.row_colors is not None and rc_col is not None:
            rc_ax = fig.add_subplot(gs[heat_row, rc_col])
            values = self.data.get_column(self.row_colors).to_list()
            rgb = np.array([[to_rgb(self.row_palette.get(values[i], "white"))]  # type: ignore[union-attr]
                            for i in order])
            rc_ax.imshow(rgb, aspect="auto", interpolation="nearest")
            rc_ax.set_xticks([0], [self.row_colors], rotation=90)
            rc_ax.set_yticks([])

        # --- one heatmap block per group ----------------------------------------------
        heatmap_axes: dict[str, Axes] = {}
        cbar_axes: dict[str, Axes] = {}
        last_name = self.groups[-1][0].name
        for (group, _), col in zip(self.groups, group_cols):
            feats = feature_orders[group.name]
            mat = self._matrix(feats)[order]
            ax = fig.add_subplot(gs[heat_row, col])
            norm, extend = _data.color_norm(
                mat, vmin=group.vmin, vmax=group.vmax, center=group.center, clip=group.clip
            )
            cmap = group.cmap or ("RdBu_r" if group.center is not None else "viridis")
            cmap = (mpl.colormaps[cmap] if isinstance(cmap, str) else cmap).with_extremes(
                bad="lightgrey"
            )
            im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto",
                           interpolation="nearest", cmap=cmap, norm=norm)
            if group.annot:
                self._annotate(ax, mat, cmap, norm, group.fmt)

            if self._show_labels(group, len(feats)):
                names = group.labels if isinstance(group.labels, Mapping) else {}
                ax.set_xticks(range(len(feats)), [names.get(f, f) for f in feats], rotation=90)
            else:
                ax.set_xticks([])
            if self.row_labels is not None and group.name == last_name:
                labels = self.data.get_column(self.row_labels).to_list()
                ax.yaxis.tick_right()
                ax.set_yticks(range(n_rows), [str(labels[i]) for i in order])
            else:
                ax.set_yticks([])

            cax = fig.add_subplot(gs[0, col])
            fig.colorbar(im, cax=cax, orientation="horizontal", extend=extend)
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_major_locator(MaxNLocator(3))
            cax.tick_params(labelsize=8)
            if group.name:
                cax.set_title(group.name, fontsize=10, pad=14)

            if self.col_colors is not None and cc_row is not None:
                cc_ax = fig.add_subplot(gs[cc_row, col])
                rgb = np.array([[to_rgb(self.col_palette.get(self.col_colors.get(f), "white"))  # type: ignore[union-attr]
                                 for f in feats]])
                cc_ax.imshow(rgb, aspect="auto", interpolation="nearest")
                cc_ax.set_xticks([])
                cc_ax.set_yticks([])
            heatmap_axes[group.name] = ax
            cbar_axes[group.name] = cax

        self._add_legends(fig)
        result = ClusterMapAxes(
            heatmap_axes=heatmap_axes, colorbar_axes=cbar_axes,
            row_dendrogram_ax=dendro_ax, row_colors_ax=rc_ax, row_order=order,
            feature_order=feature_orders, row_linkage=z,
        )
        return fig, result

    @staticmethod
    def _annotate(ax: Axes, mat: np.ndarray, cmap: Colormap, norm: Normalize, fmt: str) -> None:
        for (i, j), v in np.ndenumerate(mat):
            if not np.isfinite(v):
                continue
            r, g, b, _ = cmap(norm(v))
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
