"""Column validation, polars -> pandas conversion and order/palette resolution."""

import difflib
import re
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import Normalize

from . import fisseq


def as_frame(data: Any) -> pl.DataFrame:
    """The polars DataFrame behind ``data``: a `Dataset` is collected, a DataFrame passes
    through, and anything else raises `TypeError`."""
    from .dataset import Dataset

    if isinstance(data, Dataset):
        return data.df
    if not isinstance(data, pl.DataFrame):
        raise TypeError(f"data must be a polars DataFrame, got {type(data).__name__}")
    return data


def require_columns(df: pl.DataFrame, *columns: str | None) -> None:
    """Raise a helpful `ValueError` if any (non-``None``) column is missing from ``df``."""
    missing = [c for c in columns if c is not None and c not in df.columns]
    if not missing:
        return
    lines = []
    for col in missing:
        close = difflib.get_close_matches(col, df.columns, n=3)
        hint = f" (did you mean {', '.join(map(repr, close))}?)" if close else ""
        lines.append(f"  {col!r}{hint}")
    raise ValueError("Column(s) not found in DataFrame:\n" + "\n".join(lines))


def to_pandas(df: pl.DataFrame, *columns: str | None) -> pd.DataFrame:
    """Convert only the columns a plot uses (deduplicated, ``None`` skipped) to pandas."""
    cols = list(dict.fromkeys(c for c in columns if c is not None))
    return df.select(cols).to_pandas()


def is_numeric(df: pl.DataFrame, column: str) -> bool:
    return df.schema[column].is_numeric()


def _natural_key(value: Hashable) -> tuple:
    """Sort key that orders ``"2" < "10"`` and ``"T2_R1" < "T10_R1"``."""
    parts = re.split(r"(\d+)", str(value))
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)


def resolve_order(
    df: pl.DataFrame, column: str, order: Sequence[Any] | None = None
) -> list[Any]:
    """Levels of ``column`` in plotting order.

    An explicit ``order`` wins. Otherwise levels known to the fisseq theme come first in
    their canonical order, followed by the remaining levels sorted naturally.
    """
    if order is not None:
        return list(order)
    levels = df.get_column(column).drop_nulls().unique().to_list()
    level_set = set(levels)
    known: list[Any] = []
    for canonical in fisseq.ORDERS:
        known.extend(v for v in canonical if v in level_set and v not in known)
    rest = sorted((v for v in levels if v not in known), key=_natural_key)
    return known + rest


def resolve_palette(
    levels: Iterable[Any],
    palette: Mapping[Any, Any] | str | Sequence[Any] | None = None,
) -> dict[Any, Any]:
    """Map each level to a color.

    An explicit ``palette`` wins (a mapping, a seaborn palette name, or a list of colors).
    Otherwise the fisseq palette is used if it covers every level, else seaborn's default
    (``tab20`` for 11-20 levels and ``husl`` beyond that, so colors never repeat).
    """
    levels = list(levels)
    if isinstance(palette, Mapping):
        return dict(palette)
    if palette is None:
        if levels and all(lvl in fisseq.PALETTE for lvl in levels):
            return {lvl: fisseq.PALETTE[lvl] for lvl in levels}
        if len(levels) > 20:
            palette = "husl"
        elif len(levels) > 10:
            palette = "tab20"
    colors = sns.color_palette(palette, n_colors=len(levels))
    return dict(zip(levels, colors))


def highlight_colors(
    subset: pl.DataFrame,
    hue: str,
    palette: Mapping[Any, Any] | str | Sequence[Any] | None,
    missing_color: Any,
    base_palette: Mapping[Any, Any] | None = None,
) -> list[Any]:
    """One color per row of ``subset`` from its categorical ``hue`` level, for highlights.

    An explicit ``palette`` wins. Otherwise ``base_palette`` (the plot's own colors) is used
    when it has a color for every level (levels are also matched as strings, so integer
    cluster ids match string ones), else the default palette for ``subset``'s levels. Rows
    without a color get ``missing_color``.
    """
    if is_numeric(subset, hue) and not subset.schema[hue].is_integer():
        raise TypeError(
            f"highlight hue must be categorical, got numeric column {hue!r}"
        )
    values = subset.get_column(hue).to_list()
    levels = [v for v in dict.fromkeys(values) if v is not None]

    def lookup(colors: Mapping[Any, Any], value: Any) -> Any:
        if value in colors:
            return colors[value]
        return colors.get(str(value))

    if palette is not None:
        colors = resolve_palette(levels, palette)
    elif base_palette is not None and all(
        lookup(base_palette, v) is not None for v in levels
    ):
        colors = dict(base_palette)
    else:
        colors = resolve_palette(resolve_order(subset, hue), None)
    return [
        missing_color if v is None or lookup(colors, v) is None else lookup(colors, v)
        for v in values
    ]


def color_norm(
    values: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    clip: float | None = None,
) -> tuple[Normalize, str]:
    """Color normalization for ``values`` and the matching colorbar ``extend``.

    Explicit ``vmin``/``vmax`` win; otherwise the finite data range is used. With
    ``center``, missing limits are symmetric around it with a half-width of the largest
    absolute deviation, capped at ``clip``.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if center is not None and (vmin is None or vmax is None):
        half = float(np.max(np.abs(finite - center))) if finite.size else 1.0
        if clip is not None:
            half = min(half, clip)
        vmin = center - half if vmin is None else vmin
        vmax = center + half if vmax is None else vmax
    if finite.size:
        vmin = float(finite.min()) if vmin is None else vmin
        vmax = float(finite.max()) if vmax is None else vmax
        below, above = bool(finite.min() < vmin), bool(finite.max() > vmax)
    else:
        below = above = False
    extend = {
        (False, False): "neither",
        (True, False): "min",
        (False, True): "max",
        (True, True): "both",
    }[(below, above)]
    return Normalize(vmin=vmin, vmax=vmax), extend
