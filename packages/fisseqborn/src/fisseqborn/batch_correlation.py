"""Batch-vs-batch correlation matrices computed from long-form scores."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from . import _data
from ._base import FigurePlot, config
from .heatmap import Heatmap, with_cbar_label
from .metrics import Metric, MetricMatrix


def _scale(
    metric: Metric, matrix: MetricMatrix, vmin: float | None, vmax: float | None
) -> tuple[float, float]:
    """Color limits: 0 to 1 for a similarity, 0 to the largest value for a distance,
    unless given."""
    if metric.kind == "similarity":
        return (0.0 if vmin is None else vmin, 1.0 if vmax is None else vmax)
    finite = matrix.values[np.isfinite(matrix.values)]
    top = float(finite.max()) if finite.size else 1.0
    return (0.0 if vmin is None else vmin, top if vmax is None else vmax)


def check_unique(
    data: pl.DataFrame, batch: str, label: str, aggregate: str | None
) -> None:
    """Raise if ``(batch, label)`` repeats and no ``aggregate`` is given."""
    if aggregate is None and data.select(batch, label).is_duplicated().any():
        raise ValueError(
            f"Duplicate ({batch!r}, {label!r}) rows; pass aggregate='mean' (or similar)"
        )


def pivot_by_batch(
    data: pl.DataFrame, batch: str, label: str, col: str, aggregate: str | None
) -> pl.DataFrame:
    """``col`` with one column per batch (as strings) and one row per ``label``; null
    where a batch lacks the label."""
    return data.select(
        pl.col(batch).cast(pl.String), label, pl.col(col).cast(pl.Float64)
    ).pivot(on=batch, index=label, values=col, aggregate_function=aggregate)  # type: ignore[arg-type]


class _BatchScores:
    """The long-form scores pivoted to one column per batch, and the metric between
    every pair of batches; shared by `BatchCorrelationHeatmap` and `BatchCorrelationGrid`."""

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        batch: str,
        label: str,
        score: str,
        y_score: str | None,
        metric: Metric,
        min_shared: int,
        aggregate: str | None,
    ) -> None:
        _data.require_columns(data, batch, label, score, y_score)
        check_unique(data, batch, label, aggregate)
        self.y_score = None if y_score == score else y_score
        self.metric, self.min_shared = metric, min_shared

        def by_batch(col: str) -> pl.DataFrame:
            return pivot_by_batch(data, batch, label, col, aggregate)

        #: one column per batch, one row per label; null where a batch lacks the label
        scores = by_batch(score)
        #: the label of each row of `scores`
        self.labels = scores.get_column(label)
        self.scores = scores.drop(label)
        self.y_scores = (
            by_batch(self.y_score).drop(label) if self.y_score else self.scores
        )
        self.batches: list[str] = self.scores.columns
        #: ``(n_batches, n_batches)`` in `batches` order: cell ``(a, b)`` compares batch
        #: a's ``y_score`` with batch b's ``score``
        self.matrix = (
            metric.pairwise(self.scores, min_shared=min_shared)
            if self.y_score is None
            else metric.cross(self.y_scores, self.scores, min_shared=min_shared)
        )

    @property
    def pairs(self) -> pl.DataFrame:
        return self.matrix.pairs("batch_a", "batch_b")


class BatchCorrelationHeatmap(Heatmap):
    """Heatmap of a metric (by default the Spearman correlation) of ``score`` between every
    pair of batches, with one point per ``label`` the two batches share (e.g. replicate-vs-replicate OVWT AUROCs, one point
    per variant).

    Pairs of batches with fewer than ``min_shared`` shared labels, such as replicates from
    different tiles, are left missing and drawn in ``nan_color`` (black by default).

    With ``y_score``, the rows (y axis) correlate ``y_score`` and the columns (x axis)
    ``score``: cell ``(a, b)`` is the correlation of batch a's ``y_score`` with batch b's
    ``score`` (e.g. OVWT AUROC against a single-missense score). The matrix is then not
    symmetric, and its diagonal is each batch's within-batch correlation of the two scores
    rather than 1.

    See `BatchCorrelationGrid` for the same layout with each pair's scatter plot.

    Parameters
    ----------
    data : pl.DataFrame
        Long form: one row per ``(batch, label)``.
    batch : str
        Column whose values become the matrix rows and columns.
    label : str
        Column matched between batches, e.g. the variant.
    score : str
        Numeric column that is correlated (on both axes, or the x axis with ``y_score``).
    y_score : str | None
        Numeric column for the rows (y axis). ``None`` (or ``score``) correlates ``score``
        with itself.
    metric : str | Metric
        Any `fisseqborn.metrics` metric, e.g. ``"spearman"``, ``"pearson_squared"``,
        ``"cosine"`` (of the uncentred scores) or ``"euclidean"``.
    min_shared : int
        Minimum number of shared labels (with finite scores in both batches) for a pair to
        get a value.
    aggregate : {"mean", "median", "first", ...} | None
        polars aggregation for duplicate ``(batch, label)`` rows, which are otherwise an error.
    order : Sequence | None
        Batch order for both rows and columns. Default: natural sort.
    **kw
        Passed to `Heatmap`. Defaults: ``annot=True, nan_color="black", linewidths=0.5,
        linecolor="black"``; ``cmap="Blues", vmin=0, vmax=1`` for a similarity, or
        ``cmap="Blues_r"`` from 0 to the largest value for a distance; and a colorbar
        labeled with the metric, e.g. ``"Spearman ρ"``. With ``y_score`` the axes are labeled
        ``score`` and ``y_score``; override with ``.set(xlabel=..., ylabel=...)``.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        batch: str,
        label: str,
        score: str,
        y_score: str | None = None,
        metric: str | Metric = "spearman",
        min_shared: int = 10,
        aggregate: str | None = None,
        order: Sequence[Any] | None = None,
        figsize: tuple[float, float] | None = None,
        **kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        self.metric = Metric.get(metric)
        self._batch_scores = _BatchScores(
            data,
            batch=batch,
            label=label,
            score=score,
            y_score=y_score,
            metric=self.metric,
            min_shared=min_shared,
            aggregate=aggregate,
        )
        self.batch, self.label, self.score = batch, label, score
        self.y_score = self._batch_scores.y_score
        self.min_shared = min_shared
        self.batch_aggregate = aggregate

        batches = self._batch_scores.batches
        #: The batch-by-batch `MetricMatrix` drawn.
        self.metric_matrix = self._batch_scores.matrix
        matrix = self.metric_matrix.to_frame("batch")
        vmin, vmax = _scale(
            self.metric, self.metric_matrix, kw.pop("vmin", None), kw.pop("vmax", None)
        )

        if figsize is None:
            n = len(batches)
            figsize = (max(6.0, 0.65 * n + 1.0), max(5.0, 0.6 * n + 0.6))
        kw = with_cbar_label(
            {
                "cmap": "Blues" if self.metric.kind == "similarity" else "Blues_r",
                "vmin": vmin,
                "vmax": vmax,
                "annot": True,
                "nan_color": "black",
                "linewidths": 0.5,
                "linecolor": "black",
                "row_order": order,
                "col_order": order,
                **kw,
            },
            self.metric.label,
        )
        super().__init__(
            matrix, index="batch", columns=list(batches), figsize=figsize, **kw
        )

    def pairs(self) -> pl.DataFrame:
        """One row per unordered batch pair (including each batch with itself):
        ``batch_a, batch_b, value, n_shared``. With ``y_score``, one row per ordered
        pair: ``batch_a`` is the row batch (``y_score``), ``batch_b`` the column batch
        (``score``)."""
        return self._batch_scores.pairs

    def _draw(self, ax: Axes) -> None:
        super()._draw(ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        if self.y_score is None:
            ax.set(xlabel="", ylabel="")
        else:
            ax.set(xlabel=self.score, ylabel=self.y_score)


class BatchCorrelationGrid(FigurePlot):
    """`BatchCorrelationHeatmap` with each cell replaced by the scatter plot behind its
    value: panel ``(a, b)`` has batch b's ``score`` on x and batch a's ``score`` (or
    ``y_score``) on y, one point per shared ``label``.

    Without ``y_score`` the diagonal panels are histograms of each batch's scores (one per
    ``hue`` level, overlaid, when ``hue`` is given). Pairs with
    fewer than ``min_shared`` shared labels get no points and a ``nan_color`` background.
    Columns share x limits and rows share y limits, so each batch has one scale; every
    panel has x tick labels, the left column y tick labels.

    `plot` returns ``(fig, axes)``, ``axes`` a 2-D array; layers are applied to the
    top-left panel.

    Parameters
    ----------
    data, batch, label, score, y_score, metric, min_shared, aggregate, order
        As for `BatchCorrelationHeatmap`.
    hue : str | None
        Column to color points (and split the diagonal histograms) by. It describes the label,
        e.g. the variant type, so it must have one value per ``label`` across batches. The
        correlations still use every shared label, including those whose hue level isn't
        drawn.
    hue_order, palette
        Level order and colors; by default the fisseq order and palette for known levels
        (e.g. variant types), otherwise natural order. The first level is drawn on top. A
        legend goes below the grid.
    tint : bool
        Color each panel's background by its value on ``cmap``/``vmin``/``vmax``, with
        a shared colorbar.
    annot : bool
        Write the value and the number of shared labels in each panel's corner.
    cmap, vmin, vmax
        Background color scale for ``tint``. Default as the heatmap: ``"Blues"`` from 0
        to 1 for a similarity, ``"Blues_r"`` from 0 to the largest value for a distance.
    tint_alpha : float
        Opacity of the tint (and the colorbar), so points stay visible on dark panels.
    nan_color : color
        Background of panels without a correlation.
    identity : bool
        Draw the dashed ``y = x`` line in the scatter panels.
    lims : tuple[float, float] | None
        Limits applied to both axes of every panel (and the histograms' range), e.g.
        ``(0, 1)`` for AUROCs. Default: each batch's data range.
    label_rotation : float
        Angle of the batch names above the columns, in degrees; the row names on the left
        turn by ``90 - label_rotation``, so ``45`` slants both and ``90`` makes the column
        names vertical and the row names horizontal. For long batch names.
    marginals : {"hist", "kde"} | None
        Draw the distribution of each scatter panel's shared points in strips along its top
        (x) and right (y) edges: histograms on the same bins as the batch's diagonal
        histogram, or KDEs scaled by each hue level's share of the points. Split by ``hue``.
    marginal_size : float
        Depth of the marginal strips as a fraction of the panel.
    height : float
        Size of each panel in inches; the figure size follows from it.
    scatter_kws, hist_kws : mapping | None
        Passed to ``ax.scatter`` (defaults ``s=8, alpha=0.7, linewidths=0, color="black"``)
        and ``ax.hist`` (defaults ``bins=20, color="gray"``; with ``hue``, overlaid
        ``histtype="stepfilled", alpha=0.4`` histograms on shared bins), for the diagonal
        and the marginal histograms.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        batch: str,
        label: str,
        score: str,
        y_score: str | None = None,
        metric: str | Metric = "spearman",
        min_shared: int = 10,
        aggregate: str | None = None,
        order: Sequence[Any] | None = None,
        hue: str | None = None,
        hue_order: Sequence[Any] | None = None,
        palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        tint: bool = True,
        annot: bool = True,
        cmap: str | mpl.colors.Colormap | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        tint_alpha: float = 0.6,
        nan_color: Any = "black",
        identity: bool = False,
        lims: tuple[float, float] | None = None,
        label_rotation: float = 0.0,
        marginals: Literal["hist", "kde"] | None = None,
        marginal_size: float = 0.25,
        height: float = 1.6,
        scatter_kws: Mapping[str, Any] | None = None,
        hist_kws: Mapping[str, Any] | None = None,
        title: str | None = None,
        dpi: int | None = None,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, dpi=dpi)
        self.metric = Metric.get(metric)
        self._batch_scores = _BatchScores(
            data,
            batch=batch,
            label=label,
            score=score,
            y_score=y_score,
            metric=self.metric,
            min_shared=min_shared,
            aggregate=aggregate,
        )
        self.batch, self.label, self.score = batch, label, score
        self.y_score = self._batch_scores.y_score
        self.min_shared = min_shared
        #: The batch-by-batch `MetricMatrix` behind the panels.
        self.metric_matrix = self._batch_scores.matrix
        batches = self._batch_scores.batches
        if order is None:
            self.order = sorted(batches, key=_data._natural_key)
        else:
            self.order = [str(b) for b in order]
            unknown = set(self.order) - set(batches)
            if unknown:
                raise ValueError(f"order has unknown batches: {sorted(unknown)}")
        _data.require_columns(data, hue)
        self.hue = hue
        self.hue_order: list[Any] | None = None
        self.palette: dict[Any, Any] | None = None
        self._hue: np.ndarray | None = None
        if hue is not None:
            per_label = data.select(label, hue).unique()
            if per_label.get_column(label).is_duplicated().any():
                raise ValueError(
                    f"hue {hue!r} must have one value per {label!r}; it differs between "
                    "batches for some labels"
                )
            self.hue_order = _data.resolve_order(data, hue, hue_order)
            self.palette = _data.resolve_palette(self.hue_order, palette)
            mapping = dict(per_label.iter_rows())
            self._hue = np.array(
                [mapping.get(v) for v in self._batch_scores.labels], dtype=object
            )
        self.tint, self.annot, self.identity = tint, annot, identity
        if cmap is None:
            cmap = "Blues" if self.metric.kind == "similarity" else "Blues_r"
        self.cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        self.norm = Normalize(*_scale(self.metric, self.metric_matrix, vmin, vmax))
        self.tint_alpha = tint_alpha
        self.nan_color, self.height = nan_color, height
        self.lims, self.label_rotation = lims, label_rotation
        if marginals not in (None, "hist", "kde"):
            raise ValueError(
                f"marginals must be 'hist', 'kde' or None, got {marginals!r}"
            )
        self.marginals, self.marginal_size = marginals, marginal_size
        self.scatter_kws = {
            "s": 8,
            "alpha": 0.7,
            "linewidths": 0,
            "color": "black",
            **(scatter_kws or {}),
        }
        self.hist_kws = {"bins": 20, **(hist_kws or {})}

    def pairs(self) -> pl.DataFrame:
        """The numbers behind the panels, as `BatchCorrelationHeatmap.pairs`."""
        return self._batch_scores.pairs

    def _draw_figure(self) -> tuple[Figure, np.ndarray]:
        bs = self._batch_scores
        n = len(self.order)
        pos = {b: i for i, b in enumerate(bs.batches)}
        mat = bs.matrix.values
        n_shared = bs.matrix.n_shared
        assert n_shared is not None
        x_all = bs.scores.fill_null(np.nan).to_numpy()
        y_all = bs.y_scores.fill_null(np.nan).to_numpy()
        symbol = self.metric.symbol
        #: (panel, x column, y row, shared-point mask) of every scatter panel
        scattered: list[tuple[Axes, int, int, np.ndarray]] = []

        # each cell holds a panel plus, with marginals, its strips
        cell = self.height * (1 + (self.marginal_size if self.marginals else 0))
        fig, axes = plt.subplots(
            n,
            n,
            figsize=(cell * n + (1.0 if self.tint else 0.2), cell * n),
            dpi=self.dpi or config.dpi,
            sharex="col",
            sharey="row",
            squeeze=False,
        )
        for i, a in enumerate(self.order):
            for j, b in enumerate(self.order):
                ax = axes[i, j]
                ia, jb = pos[a], pos[b]
                r = mat[ia, jb]
                if i == j and self.y_score is None:
                    # the histogram goes on a twin so the panel keeps its row's y scale
                    xs = x_all[:, jb]
                    xs = xs[np.isfinite(xs)]
                    ax.update_datalim(np.column_stack([xs, xs]))
                    hist_ax = ax.twinx()
                    self._hist(hist_ax, x_all[:, jb], self._edges(x_all[:, jb]))
                    hist_ax.set_yticks([])
                    continue
                if n_shared[ia, jb] < self.min_shared:
                    ax.set_facecolor(self.nan_color)
                else:
                    xs, ys = x_all[:, jb], y_all[:, ia]
                    ok = np.isfinite(xs) & np.isfinite(ys)
                    self._scatter(ax, xs, ys, ok)
                    scattered.append((ax, jb, ia, ok))
                    if self.tint and np.isfinite(r):
                        ax.set_facecolor(self.cmap(self.norm(r), alpha=self.tint_alpha))
                    if self.identity:
                        ax.axline(
                            (0, 0), slope=1, linestyle="--", color="black", linewidth=1
                        )
                    if self.annot:
                        text = (
                            f"{symbol} = {r:.2f}"
                            if np.isfinite(r)
                            else f"{symbol} = NA"
                        )
                        ax.text(
                            0.95,
                            0.05,
                            f"{text}\nn = {n_shared[ia, jb]}",
                            transform=ax.transAxes,
                            ha="right",
                            va="bottom",
                            fontsize=7,
                            bbox={
                                "boxstyle": "round,pad=0.2",
                                "facecolor": "white",
                                "alpha": 0.8,
                                "edgecolor": "none",
                            },
                            zorder=10,
                        )
        for (i, j), ax in np.ndenumerate(axes):
            ax.autoscale_view()
            ax.tick_params(labelsize=7, labelbottom=True, labelleft=j == 0)
        if self.lims is not None:
            for k in range(n):  # rows and columns share their limits
                axes[0, k].set_xlim(self.lims)
                axes[k, 0].set_ylim(self.lims)
        if self.marginals is not None:
            for ax in axes.flat:  # freeze, so the strips' artists don't rescale panels
                ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
            for ax, jb, ia, ok in scattered:
                self._draw_marginals(ax, x_all[:, jb], y_all[:, ia], ok)
        rot = self.label_rotation
        title_pad = 4.0
        if self.marginals is not None:
            title_pad += self.marginal_size * self.height * 72
        for j, b in enumerate(self.order):
            axes[0, j].set_title(
                b,
                fontsize=9,
                pad=title_pad,
                rotation=rot,
                ha="left" if rot else "center",
                rotation_mode="anchor",
            )
        for i, a in enumerate(self.order):
            axes[i, 0].set_ylabel(
                a,
                fontsize=9,
                rotation=90 - rot,
                ha="right" if rot else "center",
                va="center" if rot else "bottom",
                rotation_mode="anchor",
            )
        x_label = self.score
        y_label = self.y_score or self.score
        fig.supxlabel(x_label, fontsize=10)
        fig.supylabel(y_label, fontsize=10)
        fig.tight_layout(rect=(0, 0, 0.88, 1) if self.tint else None)
        if self.tint:
            sm = mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
            cbar = fig.colorbar(
                sm,
                ax=axes.ravel().tolist(),
                fraction=0.04,
                # clear the last column's right-hand marginal strips
                pad=0.02 + (self.marginal_size / n if self.marginals else 0),
                label=self.metric.label,
            )
            cbar.solids.set_alpha(self.tint_alpha)
        if self.hue_order:
            assert self.palette is not None
            handles = [
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    color=self.palette[level],
                    label=str(level),
                )
                for level in self.hue_order
            ]
            fig.legend(
                handles=handles,
                title=self.hue,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=min(len(handles), 5),
                frameon=False,
            )
        return fig, axes

    def _levels(self, ok: np.ndarray) -> list[tuple[Any, np.ndarray]]:
        """``(color, mask)`` per hue level in drawing order (one uncolored entry without
        ``hue``)."""
        if self._hue is None or self.hue_order is None or self.palette is None:
            return [(None, ok)]
        # reversed: the first level is drawn last, on top
        return [
            (self.palette[level], ok & (self._hue == level))
            for level in reversed(self.hue_order)
        ]

    def _scatter(
        self, ax: Axes, xs: np.ndarray, ys: np.ndarray, ok: np.ndarray
    ) -> None:
        for color, mask in self._levels(ok):
            kws = (
                self.scatter_kws
                if color is None
                else {**self.scatter_kws, "color": color}
            )
            ax.scatter(xs[mask], ys[mask], **kws)

    def _edges(self, xs: np.ndarray) -> np.ndarray:
        """Histogram bin edges of one batch's scores, shared by its diagonal and marginals."""
        ok = np.isfinite(xs)
        return np.histogram_bin_edges(
            xs[ok],
            bins=self.hist_kws["bins"],
            range=self.hist_kws.get("range", self.lims),
        )

    def _hist(
        self,
        ax: Axes,
        xs: np.ndarray,
        edges: np.ndarray,
        ok: np.ndarray | None = None,
        orientation: str = "vertical",
    ) -> None:
        ok = np.isfinite(xs) if ok is None else ok
        hist_kws = {
            **self.hist_kws,
            "range": None,
            "bins": edges,
            "orientation": orientation,
        }
        if self._hue is None:
            ax.hist(xs[ok], **{"color": "gray", **hist_kws})
            return
        for color, mask in self._levels(ok):
            ax.hist(
                xs[mask],
                **{"histtype": "stepfilled", "alpha": 0.4, **hist_kws, "color": color},
            )

    def _kde(
        self, ax: Axes, xs: np.ndarray, ok: np.ndarray, grid: np.ndarray, vertical: bool
    ) -> None:
        total = ok.sum()
        for color, mask in self._levels(ok):
            values = xs[mask]
            if len(values) < 2 or np.ptp(values) == 0:
                continue
            density = gaussian_kde(values)(grid) * len(values) / total
            color = "gray" if color is None else color
            if vertical:
                ax.plot(grid, density, color=color, linewidth=1)
                ax.fill_between(grid, density, color=color, alpha=0.25, linewidth=0)
            else:
                ax.plot(density, grid, color=color, linewidth=1)
                ax.fill_betweenx(grid, density, color=color, alpha=0.25, linewidth=0)

    def _draw_marginals(
        self, ax: Axes, xs: np.ndarray, ys: np.ndarray, ok: np.ndarray
    ) -> None:
        """Distribution strips of a scatter panel's shared points above (x) and to the
        right (y) of it."""
        size = self.marginal_size
        top = ax.inset_axes((0, 1, 1, size), sharex=ax)
        right = ax.inset_axes((1, 0, size, 1), sharey=ax)
        for strip, values, vertical in ((top, xs, True), (right, ys, False)):
            if self.marginals == "hist":
                self._hist(
                    strip,
                    values,
                    self._edges(values),
                    ok,
                    "vertical" if vertical else "horizontal",
                )
            else:
                lim = ax.get_xlim() if vertical else ax.get_ylim()
                self._kde(strip, values, ok, np.linspace(*lim, 200), vertical)
            strip.patch.set_visible(False)
            strip.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            for name, spine in strip.spines.items():
                spine.set_visible(name == ("bottom" if vertical else "left"))
            if vertical:
                strip.set_ylim(bottom=0)
            else:
                strip.set_xlim(left=0)

    def _main_ax(self, handle: np.ndarray) -> Axes:
        return handle[0, 0]
