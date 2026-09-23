"""ROC curves for separating a positive class from a negative (control) class by a score."""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
from matplotlib.axes import Axes
from sklearn.metrics import roc_auc_score, roc_curve

from . import _data
from ._base import Plot


def _as_list(value: str | Sequence[str]) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


class RocPlot(Plot):
    """ROC curves: how well ``score`` separates ``positive`` from ``negative`` rows of
    ``label``.

    Parameters
    ----------
    data : pl.DataFrame
    label : str
        Class column, e.g. ``"meta_variant_type"`` or ``"meta_clinvar_annotation"``.
    positive, negative : str | Sequence[str]
        Level(s) of ``label`` treated as positive / negative. Rows with any other label
        are dropped.
    score : str | Sequence[str]
        Score column(s). Several columns give one curve each (e.g. one per variance ratio).
    group : str | None
        One curve per level of this column (e.g. ``"meta_experiment"``). Only allowed with
        a single ``score``.
    combined : bool
        With ``group``, add a black curve over all groups pooled.
    names : mapping | None
        Display names for score columns / group levels in the legend.
    palette : mapping | str | list | None
        Colors per curve (keyed by score column or group level).
    **line_kw
        Passed to ``ax.plot`` for every curve.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        *,
        label: str,
        positive: str | Sequence[str],
        score: str | Sequence[str],
        negative: str | Sequence[str] = "Synonymous",
        group: str | None = None,
        combined: bool = False,
        names: Mapping[Any, str] | None = None,
        palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
        group_order: Sequence[Any] | None = None,
        title: str | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        **line_kw: Any,
    ) -> None:
        super().__init__(data, title=title, figsize=figsize, dpi=dpi)
        self.scores = _as_list(score)
        _data.require_columns(data, label, group, *self.scores)
        if group is not None and len(self.scores) > 1:
            raise ValueError("Pass either several score columns or a group column, not both")
        self.label, self.group, self.combined = label, group, combined
        self.positive, self.negative = _as_list(positive), _as_list(negative)
        overlap = set(self.positive) & set(self.negative)
        if overlap:
            raise ValueError(f"Levels are both positive and negative: {sorted(overlap)}")
        self.names = dict(names or {})
        self.line_kw = line_kw
        self.group_order = (
            _data.resolve_order(data, group, group_order) if group is not None else None
        )
        keys = self.group_order if group is not None else self.scores
        self.palette = _data.resolve_palette(keys, palette)  # type: ignore[arg-type]

    def _labelled(self) -> pl.DataFrame:
        return self.data.filter(
            pl.col(self.label).is_in(self.positive + self.negative)
        ).with_columns(pl.col(self.label).is_in(self.positive).alias("__is_positive"))

    def _curves(self) -> list[tuple[Any, str, pl.DataFrame]]:
        """``(palette key, score column, rows)`` for every curve."""
        df = self._labelled()
        if self.group is None:
            return [(s, s, df) for s in self.scores]
        score = self.scores[0]
        curves = [
            (lvl, score, df.filter(pl.col(self.group) == lvl)) for lvl in self.group_order or []
        ]
        if self.combined:
            curves.append(("Combined", score, df))
        return curves

    @staticmethod
    def _roc_inputs(rows: pl.DataFrame, score: str):
        rows = rows.filter(pl.col(score).cast(pl.Float64).is_finite())
        return rows.get_column("__is_positive").to_numpy(), rows.get_column(score).to_numpy()

    def aucs(self) -> pl.DataFrame:
        """One row per curve: ``curve``, ``auc``, ``n_positive``, ``n_negative``."""
        rows = []
        for key, score, curve_df in self._curves():
            truth, values = self._roc_inputs(curve_df, score)
            n_pos = int(truth.sum())
            n_neg = len(truth) - n_pos
            auc = roc_auc_score(truth, values) if n_pos and n_neg else float("nan")
            rows.append({"curve": str(key), "auc": float(auc),
                         "n_positive": n_pos, "n_negative": n_neg})
        return pl.DataFrame(rows, schema={"curve": pl.String, "auc": pl.Float64,
                                          "n_positive": pl.Int64, "n_negative": pl.Int64})

    def _draw(self, ax: Axes) -> None:
        for key, score, curve_df in self._curves():
            truth, values = self._roc_inputs(curve_df, score)
            if truth.all() or not truth.any():
                continue  # ROC undefined without both classes
            fpr, tpr, _ = roc_curve(truth, values)
            auc = roc_auc_score(truth, values)
            name = self.names.get(key, str(key))
            color = "black" if key == "Combined" and self.group else self.palette.get(key)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color,
                    **{"linewidth": 2, **self.line_kw})
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
               ylabel="True positive rate")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right")
