"""Volcano plots of per-feature effect sizes against -log10 p-values."""

import copy
import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Self

import numpy as np
import polars as pl
import polars.selectors as cs
from adjustText import adjust_text
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from scipy import stats

from . import _data, fisseq
from ._base import Plot


@dataclasses.dataclass(frozen=True)
class _PointLayer:
    where: pl.Expr
    label: str | None
    color: Any
    scatter_kw: Mapping[str, Any]


_Stat = Callable[[np.ndarray, np.ndarray], np.ndarray]

#: p-value tests for `VolcanoPlot.from_labels`: name -> (y column, two-sided p per column)
_TESTS: dict[str, tuple[str, _Stat]] = {
    "ks": (
        "KSnegLogP",
        lambda a, b: stats.ks_2samp(a, b, axis=0, nan_policy="omit").pvalue,
    ),
    "welch": (
        "WelchNegLogP",
        lambda a, b: (
            stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit").pvalue
        ),
    ),
    "mannwhitney": (
        "MWUnegLogP",
        lambda a, b: stats.mannwhitneyu(a, b, axis=0, nan_policy="omit").pvalue,
    ),
}


def _cohens_d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na, nb = np.isfinite(a).sum(axis=0), np.isfinite(b).sum(axis=0)
    pooled = (
        (na - 1) * np.nanvar(a, axis=0, ddof=1)
        + (nb - 1) * np.nanvar(b, axis=0, ddof=1)
    ) / (na + nb - 2)
    return (np.nanmean(a, axis=0) - np.nanmean(b, axis=0)) / np.sqrt(pooled)


#: effect sizes for `VolcanoPlot.from_labels`: name -> (x column, positive minus negative)
_EFFECTS: dict[str, tuple[str, _Stat]] = {
    "median": (
        "median_diff",
        lambda a, b: np.nanmedian(a, axis=0) - np.nanmedian(b, axis=0),
    ),
    "mean": ("mean_diff", lambda a, b: np.nanmean(a, axis=0) - np.nanmean(b, axis=0)),
    "cohens_d": ("cohens_d", _cohens_d),
}


def _as_list(value: str | Sequence[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


def _lookup(options: dict[str, tuple[str, _Stat]], name: str, kind: str):
    if name not in options:
        raise ValueError(f"Unknown {kind} {name!r}; choose from {sorted(options)}")
    return options[name]


def _label_mask(df: pl.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    """``(keep, positive)`` boolean arrays for a binary label column; nulls are dropped."""
    col = df.get_column(label)
    if col.dtype != pl.Boolean:
        values = set(col.drop_nulls().unique().to_list())
        if not col.dtype.is_integer() or not values <= {0, 1}:
            raise ValueError(
                f"Label column {label!r} must be boolean or 0/1, got {col.dtype}"
                + (
                    f" with values {sorted(values)[:5]}"
                    if col.dtype.is_integer()
                    else ""
                )
            )
        col = col.cast(pl.Boolean)
    keep = col.is_not_null().to_numpy()
    positive = col.fill_null(False).to_numpy()
    if not positive[keep].any() or positive[keep].all():
        raise ValueError(f"Label column {label!r} needs both true and false rows")
    return keep, positive


def _label_statistics(
    df: pl.DataFrame, label: str, features: list[str], test: str, effect: str
) -> pl.DataFrame:
    """One row per feature: positive vs. negative rows of ``label``."""
    y, test_fn = _lookup(_TESTS, test, "test")
    x, effect_fn = _lookup(_EFFECTS, effect, "effect")
    keep, positive = _label_mask(df, label)
    matrix = df.select(pl.col(features).cast(pl.Float64)).to_numpy()
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    pos, neg = matrix[keep & positive], matrix[keep & ~positive]
    # features with too few finite values in a group come out NaN (and are not drawn)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        p = np.asarray(test_fn(pos, neg), dtype=float)
        x_values = np.asarray(effect_fn(pos, neg), dtype=float)
    return pl.DataFrame(
        {
            "label": label,
            "feature": features,
            "n_positive": np.isfinite(pos).sum(axis=0),
            "n_negative": np.isfinite(neg).sum(axis=0),
            x: x_values,
            # p-values that underflow to 0 are capped at -log10(tiny) ~ 307.7
            y: -np.log10(np.maximum(p, np.finfo(float).tiny)),
        }
    )


class VolcanoPlot(Plot):
    """Volcano plot of long-form ``(variant, feature, effect, -log10 p)`` rows.

    Points are added in groups with chained `layer` calls, each selecting its rows with a
    polars expression over the metadata columns. Layers are drawn in the order they are
    added, so add the largest group first and the controls last to keep them visible.
    Without any layers, every row is drawn as a single grey group. Rows with a missing or
    non-finite ``x`` or ``y`` are not drawn or counted.

    Use `from_wide` for one row per variant with ``<feature>_median`` /
    ``<feature>_KSnegLogP`` column pairs, or `from_labels` to run the tests on a frame of
    binary label and feature columns.

    >>> (fb.VolcanoPlot(long_df, x="median", y="KSnegLogP")
    ...    .layer(pl.col("meta_variant_type") == "Single Missense", label="Single Missense")
    ...    .layer(pl.col("meta_variant_type") == "Synonymous", label="Synonymous")
    ...    .save("vis/volcano.png"))

    Parameters
    ----------
    data : pl.DataFrame
        One row per (variant, feature) pair, plus any metadata columns the layers filter on.
    x : str
        Effect-size column, e.g. the median normalized feature value.
    y : str
        ``-log10(p)`` column.
    alpha : float | None, default 0.05
        Significance level drawn as a dashed horizontal line and listed in the legend.
        ``None`` draws no line.
    bonferroni : bool, default True
        Divide ``alpha`` by the number of plotted points (the tests across all layers).
    effect_cutoff : float | None, default None
        Effect-size cutoff drawn as dashed vertical lines at ``±effect_cutoff`` (e.g. 0.2
        for ``cohens_d``). It also restricts which points `label_top` may label.
    label_top : int | None, default None
        Annotate the ``label_top`` top-ranked points (see ``label_rank``) among the
        significant ones: above the significance line and beyond ``effect_cutoff``, where
        set. The x-axis widens to keep them in view. adjustText moves the labels apart,
        inside the axes, with leader lines to their points, and the legend moves outside
        the axes to the right.
    label_rank : {"y", "x"}, default "y"
        Rank points for `label_top` by ``y`` (the most significant) or by ``|x|`` (the
        largest effect).
    label_by : str, default "feature"
        Column whose values annotate the `label_top` points.
    x_quantiles : tuple[float, float] | None, default (0.001, 0.999)
        Limit the x-axis to these quantiles of the plotted points, so a few extreme
        values don't squash the rest. ``None`` shows every point.
    **scatter_kw
        Defaults for every layer's ``ax.scatter`` call; `layer` kwargs override them.
        Point size ``s`` and transparency ``alpha`` default to values that scale with
        the number of plotted points: large and opaque for a few hundred variants, tiny
        and faint for millions.
    """

    default_figsize = (7.0, 6.0)

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        x: str,
        y: str,
        alpha: float | None = 0.05,
        bonferroni: bool = True,
        effect_cutoff: float | None = None,
        label_top: int | None = None,
        label_rank: Literal["y", "x"] = "y",
        label_by: str = "feature",
        x_quantiles: tuple[float, float] | None = (0.001, 0.999),
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **scatter_kw: Any,
    ) -> None:
        data = _data.as_frame(data)
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        _data.require_columns(data, x, y, None if label_top is None else label_by)
        self.x, self.y = x, y
        self.alpha, self.bonferroni = alpha, bonferroni
        self.effect_cutoff = None if effect_cutoff is None else abs(effect_cutoff)
        if label_rank not in ("x", "y"):
            raise ValueError(f"label_rank must be 'x' or 'y', got {label_rank!r}")
        self.label_top, self.label_rank, self.label_by = label_top, label_rank, label_by
        self.x_quantiles = x_quantiles
        self.scatter_kw = scatter_kw
        self._point_layers: list[_PointLayer] = []

    @classmethod
    def from_wide(
        cls,
        data: pl.DataFrame,
        *,
        x_suffix: str = "_median",
        y_suffix: str = "_KSnegLogP",
        metadata: cs.Selector | None = None,
        **kw: Any,
    ) -> Self:
        """Volcano plot of a wide frame with one row per variant.

        Every feature with both a ``<feature><x_suffix>`` and a ``<feature><y_suffix>``
        column becomes one point per variant. The long-form data has a ``feature`` column
        (the feature name without its suffix), the ``metadata`` columns for `layer` to
        filter on, and ``x``/``y`` columns named after the suffixes without their leading
        underscore (by default ``"median"`` and ``"KSnegLogP"``). ``metadata`` defaults to
        every ``meta_`` column. ``kw`` is passed to the constructor.
        """
        data = _data.as_frame(data)
        x, y = x_suffix.lstrip("_"), y_suffix.lstrip("_")
        features = [
            c.removesuffix(x_suffix)
            for c in data.columns
            if c.endswith(x_suffix)
            and c.removesuffix(x_suffix) + y_suffix in data.columns
        ]
        if not features:
            raise ValueError(
                f"No feature has both a {x_suffix!r} and a {y_suffix!r} column"
            )
        index = data.select(
            cs.starts_with("meta_") if metadata is None else metadata
        ).columns

        def unpivot(suffix: str, name: str) -> pl.DataFrame:
            return data.unpivot(
                on=[f + suffix for f in features],
                index=index,
                variable_name="feature",
                value_name=name,
            )

        # Both unpivots walk the features in the same order, so their rows line up
        long = unpivot(x_suffix, x).with_columns(
            pl.col("feature").str.strip_suffix(x_suffix),
            unpivot(y_suffix, y).get_column(y),
        )
        return cls(long, x=x, y=y, **kw)

    @classmethod
    def from_labels(
        cls,
        data: pl.DataFrame,
        labels: str | Sequence[str],
        *,
        features: cs.Selector | Sequence[str] | None = None,
        test: Literal["ks", "welch", "mannwhitney"] = "ks",
        effect: Literal["median", "mean", "cohens_d"] = "median",
        **kw: Any,
    ) -> Self:
        """Volcano plot of per-feature tests between the two groups of binary labels.

        ``data`` has one row per variant or cell. Each label column (boolean or 0/1) is
        one comparison of its true rows against its false rows; rows where it is null are
        left out. Every (label, feature) pair becomes one point. The p-value test and the
        effect size are chosen independently:

        - ``test``: ``"ks"`` (two-sample Kolmogorov-Smirnov, y column ``KSnegLogP``),
          ``"welch"`` (Welch's t-test, ``WelchNegLogP``) or ``"mannwhitney"``
          (Mann-Whitney U, ``MWUnegLogP``), all two-sided;
        - ``effect``, positive minus negative group: ``"median"`` (``median_diff``),
          ``"mean"`` (``mean_diff``) or ``"cohens_d"`` (mean difference over the pooled
          standard deviation, ``cohens_d``).

        Missing and non-finite feature values are ignored. A p-value that underflows to 0
        is drawn at ``-log10`` of the smallest positive float (~307.7). The long-form
        data has ``label`` (the label column's name), ``feature``, ``n_positive`` and
        ``n_negative`` (finite values per group) and the x and y columns, so layers can
        select on ``label`` or ``feature``. ``features`` defaults to every numeric column
        that is neither a ``meta_`` column nor a label. ``kw`` is passed to the
        constructor.

        >>> fb.VolcanoPlot.from_labels(profiles, "is_frameshift", test="ks", effect="median")
        """
        data = _data.as_frame(data)
        labels = _as_list(labels)
        _data.require_columns(data, *labels)
        if features is None:
            features = cs.numeric() - cs.starts_with("meta_") - cs.by_name(labels)
        if isinstance(features, cs.Selector):
            features = data.select(features).columns
        else:
            features = list(features)
            _data.require_columns(data, *features)
        if not features:
            raise ValueError("No feature columns selected")
        long = pl.concat(
            [_label_statistics(data, lbl, features, test, effect) for lbl in labels]
        )
        x, y = _EFFECTS[effect][0], _TESTS[test][0]
        return cls(long, x=x, y=y, **kw)

    # ----- layers ---------------------------------------------------------------------

    def layer(
        self,
        where: pl.Expr,
        *,
        label: str | None = None,
        color: Any = None,
        **scatter_kw: Any,
    ) -> Self:
        """Add a group of points: the rows matching ``where``, drawn above earlier layers.

        ``color`` defaults to the fisseq palette entry for ``label`` (e.g.
        ``"Synonymous"`` is dark green), else grey. The legend shows ``label`` with the
        number of points drawn.
        """
        if color is None:
            color = fisseq.PALETTE.get(label, "grey")
        new = copy.copy(self)
        new._point_layers = [
            *self._point_layers,
            _PointLayer(where, label, color, scatter_kw),
        ]
        new._figure = None
        return new

    # ----- drawing --------------------------------------------------------------------

    def significance_threshold(self) -> float | None:
        """The ``-log10(p)`` height of the significance line, or ``None`` without one."""
        return self._threshold(sum(rows.height for _, rows in self._points()))

    def _threshold(self, n_tests: int) -> float | None:
        if self.alpha is None:
            return None
        alpha = self.alpha / max(n_tests, 1) if self.bonferroni else self.alpha
        return float(-np.log10(alpha))

    def _points(self) -> list[tuple[_PointLayer, pl.DataFrame]]:
        """Each layer with its drawable rows (the ``x``, ``y`` and label columns)."""
        layers = self._point_layers or [_PointLayer(pl.lit(True), None, "grey", {})]
        columns = [self.x, self.y]
        if self.label_top is not None and self.label_by not in columns:
            columns.append(self.label_by)
        finite = self.data.filter(
            pl.col(self.x).cast(pl.Float64).is_finite(),
            pl.col(self.y).cast(pl.Float64).is_finite(),
        )
        return [(lyr, finite.filter(lyr.where).select(*columns)) for lyr in layers]

    @staticmethod
    def _point_style(n_points: int) -> dict[str, float]:
        """Default marker size and transparency: ``s`` 20 and ``alpha`` 0.8 up to about a
        thousand points, shrinking to ``s`` 1 and ``alpha`` 0.1 by a million."""
        n = max(n_points, 1)
        return {
            "s": float(np.clip(np.sqrt(4e5 / n), 1, 20)),
            "alpha": float(np.clip(np.sqrt(1e4 / n), 0.1, 0.8)),
        }

    def _top_points(
        self, points: list[tuple[_PointLayer, pl.DataFrame]], threshold: float | None
    ) -> pl.DataFrame:
        """The `label_top` significant rows with the largest ``y`` (or ``|x|``)."""
        rows = pl.concat([r for _, r in points])
        if threshold is not None:
            rows = rows.filter(pl.col(self.y) >= threshold)
        if self.effect_cutoff is not None:
            rows = rows.filter(pl.col(self.x).abs() >= self.effect_cutoff)
        rank = pl.col(self.y) if self.label_rank == "y" else pl.col(self.x).abs()
        return rows.sort(rank, descending=True).head(self.label_top)

    def _annotate(self, ax: Axes, top: pl.DataFrame, zorder: int) -> None:
        """Label ``top``, letting adjustText spread the labels apart inside the axes."""
        texts = [
            ax.text(x, y, str(text), fontsize="x-small", zorder=zorder)
            for x, y, text in top.select(self.x, self.y, self.label_by).iter_rows()
        ]
        adjust_text(
            texts,
            x=top[self.x].to_numpy(),
            y=top[self.y].to_numpy(),
            ax=ax,
            arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.5},
        )

    def _draw(self, ax: Axes) -> None:
        points = self._points()
        n_points = sum(rows.height for _, rows in points)
        style = self._point_style(n_points)
        for zorder, (lyr, rows) in enumerate(points, start=1):
            kw = {
                **style,
                "linewidths": 0,
                "rasterized": True,
                **self.scatter_kw,
                **lyr.scatter_kw,
            }
            ax.scatter(
                rows.get_column(self.x).to_numpy(),
                rows.get_column(self.y).to_numpy(),
                color=lyr.color,
                label=None if lyr.label is None else f"{lyr.label} (n={rows.height:,})",
                zorder=zorder,
                **kw,
            )

        threshold = self._threshold(n_points)
        if threshold is not None:
            name = "Bonferroni p" if self.bonferroni else "p"
            ax.axhline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=1,
                zorder=len(points) + 1,
                label=f"{name} = {self.alpha:g}",
            )

        if self.effect_cutoff is not None:
            for sign in (-1, 1):
                ax.axvline(
                    sign * self.effect_cutoff,
                    color="black",
                    linestyle=":",
                    linewidth=1,
                    zorder=len(points) + 1,
                    label=f"|{self.x}| = {self.effect_cutoff:g}" if sign > 0 else None,
                )

        top = None if self.label_top is None else self._top_points(points, threshold)
        if self.x_quantiles is not None:
            xs = pl.concat([rows.get_column(self.x) for _, rows in points])
            if xs.len():
                lo, hi = self.x_quantiles
                lo, hi = xs.quantile(lo), xs.quantile(hi)
                if top is not None and top.height:
                    lo = min(lo, top[self.x].min())
                    hi = max(hi, top[self.x].max())
                pad = 0.05 * (hi - lo)  # keep points at the limits whole
                ax.set_xlim(lo - pad, hi + pad)

        ax.set(xlabel=self.x, ylabel=self.y)
        ax.spines[["top", "right"]].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            # labelled points are often in the top corners, so move the legend out of them
            outside = {} if top is None else {"bbox_to_anchor": (1.02, 1)}
            legend = ax.legend(frameon=False, loc="upper left", **outside)
            # legend markers are readable and opaque whatever the points' size and alpha
            for handle in legend.legend_handles:
                handle.set_alpha(1)
                if isinstance(handle, PathCollection):
                    handle.set_sizes([20])
        if top is not None:
            self._annotate(ax, top, zorder=len(points) + 2)
