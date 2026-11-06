"""An interactive variant effect map: substituted amino acid by structure residue.

The same layout and colors as `fisseqborn.VariantEffectMap` in the variant-effect notebooks:
rows in `fisseqborn.fisseq.AMINO_ACID_ORDER`, one column per position, colored on the chosen
colormap (Blues from 0.5 to 1.0 for distinguishability), gray for variants without a score and a black dot on each position's wild-type cell. Below it,
the per-position aggregate that colors the structure.

Instead of wrapping long proteins into rows, the map is drawn at its full length with a fixed
cell width, inside a box the browser scrolls sideways; zooming changes the cell width. The
amino-acid labels (`labels_figure`) and the color bars (`colorbar_figure`) are separate figures
on either side of that box, so they don't scroll away. The three share `rows` and `MARGIN`
so their rows line up. Region bars (`regions.Track`, domains or tiles) sit on top. Nothing pans or zooms inside the figures.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import polars as pl

from fisseqborn.fisseq import AMINO_ACID_ORDER

from .aggregate import RESIDUE, VALUE
from .colors import COLORSCALE, MISSING_GRAY
from .data import (
    CLINVAR_CLASS,
    CLINVAR_CLASSES,
    CONDITIONS,
    N_EXPERIMENTS,
    POSITION,
    SCORE,
    SIGNIFICANCE,
    STARS,
    VARIANT,
    VARIANT_TYPE,
)
from .regions import Track
from .structure import Structure

#: Top and bottom figure margins in pixels (the x tick labels sit in the bottom one).
MARGIN = dict(t=8, b=40)

#: Row heights and gaps in pixels: each region bar, the map, the aggregate track.
REGION_PX = 16
REGION_GAP_PX = 3
MAP_PX = 360
TRACK_PX = 80
GAP_PX = 22

#: Cell width in pixels: the default, and the zoom slider's range.
CELL_PX = 8
CELL_PX_RANGE = (2, 24)

#: Widths of the labels and color bar figures.
LABELS_WIDTH = 72
COLORBAR_WIDTH = 120

#: Per-theme figure colors (the page's are in ``assets/structview.css``).
THEMES = {
    "light": dict(text="#222222", panel="#ffffff", line="#555555", grid="#dddddd"),
    "dark": dict(text="#dddddd", panel="#1c1e24", line="#bbbbbb", grid="#3a3d45"),
}

#: Variant classes drawn on the map: single substitutions, and the synonymous ones in the
#: wild-type row.
MAP_TYPES = ("Single Missense", "Synonymous")

#: Column of `map_cells` holding the substituted amino acid.
AA = "aa"


def _theme(theme: str | None) -> dict[str, str]:
    return THEMES.get(theme or "light", THEMES["light"])


def map_cells(
    variants: pl.DataFrame, offset: int = 0, clinvar: pl.DataFrame | None = None
) -> pl.DataFrame:
    """The map's cells: one row per single substitution (missense or synonymous) with a
    score, at its structure residue.

    Parameters
    ----------
    variants : pl.DataFrame
        A table of `data.RunScores.tables`.
    offset : int
        Structure residue = score position + ``offset``.
    clinvar : pl.DataFrame | None
        `data.ClinvarCalls.table`, joined on by variant.

    Returns
    -------
    pl.DataFrame
        `aggregate.RESIDUE`, `AA`, and the variants' `data.VARIANT`, `data.SCORE` and
        `data.N_EXPERIMENTS`; with ``clinvar``, also its class, significance, stars and
        conditions (null for variants without a call).
    """
    cells = (
        variants.filter(
            pl.col(VARIANT_TYPE).is_in(MAP_TYPES)
            & pl.col(POSITION).is_not_null()
            & pl.col(SCORE).is_not_null()
        )
        .select(
            (pl.col(POSITION) + int(offset)).alias(RESIDUE),
            pl.col(VARIANT).str.extract(r"^[A-Z]\d+([A-Z])$", 1).alias(AA),
            VARIANT,
            SCORE,
            N_EXPERIMENTS,
        )
        .filter(pl.col(AA).is_in(AMINO_ACID_ORDER))
    )
    if clinvar is not None:
        cells = cells.join(clinvar, on=VARIANT, how="left", maintain_order="left")
    return cells


def _span(structure: Structure) -> list[int]:
    """Every residue number from the structure chain's first to its last (gaps included)."""
    numbers = list(structure.residues)
    return list(range(min(numbers), max(numbers) + 1)) if numbers else []


@dataclass(frozen=True)
class Rows:
    """The vertical layout shared by the map, labels and color bar figures: the figure
    height and each row's y domain (fractions of the plot area)."""

    height: int
    map: tuple[float, float]
    track: tuple[float, float]
    regions: tuple[tuple[float, float], ...]


def rows(n_regions: int = 0) -> Rows:
    """Region bars on top (one per track), then the map, then the aggregate track."""
    blocks = [(REGION_PX, REGION_GAP_PX)] * n_regions
    if blocks:
        blocks[-1] = (REGION_PX, GAP_PX // 2)
    blocks += [(MAP_PX, GAP_PX), (TRACK_PX, 0)]
    plot = sum(h + g for h, g in blocks)
    domains = []
    top = plot
    for h, g in blocks:
        domains.append((round((top - h) / plot, 6), round(top / plot, 6)))
        top -= h + g
    return Rows(
        height=plot + MARGIN["t"] + MARGIN["b"],
        map=domains[-2],
        track=domains[-1],
        regions=tuple(domains[:-2]),
    )


def _track_yrange(per_pos: pl.DataFrame) -> list[float]:
    """The track's y range, fixed so the labels figure can show the same axis."""
    values = [
        v for v in (per_pos[VALUE].to_list() if per_pos.height else []) if v is not None
    ]
    if not values:
        return [0.0, 1.0]
    low, high = min(values), max(values)
    pad = (high - low) * 0.08 or 0.05
    return [low - pad, high + pad]


def map_width(structure: Structure, cell_px: float = CELL_PX) -> int:
    """The map figure's width in pixels: one ``cell_px`` column per residue of the span."""
    return max(10, round(len(_span(structure)) * cell_px))


def geometry(structure: Structure, cell_px: float = CELL_PX) -> dict[str, float]:
    """Where residues sit in the map figure, for the browser to scroll to one: residue ``r``'s
    column is centered ``(r - first + 0.5) * cell`` pixels from the figure's left edge."""
    span = _span(structure)
    return {"first": span[0] if span else 0, "cell": float(cell_px)}


def _region_axis(i: int) -> str:
    """The y axis name of region bar ``i`` (the map is ``y``, the track ``y2``)."""
    return f"yaxis{i + 3}"


def _axes(fig: go.Figure, layout: Rows, tracks: Sequence[Track]) -> None:
    """One shared x axis, under the track; a y axis per row. None pans or zooms."""
    fig.update_layout(
        xaxis=dict(anchor="y2", domain=[0, 1]),
        yaxis=dict(domain=list(layout.map), anchor="x"),
        yaxis2=dict(domain=list(layout.track), anchor="x"),
        **{
            _region_axis(i): dict(
                domain=list(layout.regions[i]),
                anchor="x",
                range=[t.n_lanes - 0.5, -0.5],
                showticklabels=False,
            )
            for i, t in enumerate(tracks)
        },
    )
    fig.update_xaxes(fixedrange=True, showgrid=False, zeroline=False)
    fig.update_yaxes(fixedrange=True, showgrid=False, zeroline=False)


def _layout(
    fig: go.Figure, width: int, height: int, colors: dict[str, str], **margin
) -> None:
    fig.update_layout(
        width=width,
        height=height,
        autosize=False,
        margin=dict(MARGIN, pad=0, **margin),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"], size=11),
        showlegend=False,
        dragmode=False,
        hovermode="closest",
        clickmode="event",
    )
    fig.update_xaxes(fixedrange=True, showgrid=False, zeroline=False)
    fig.update_yaxes(fixedrange=True, showgrid=False, zeroline=False)


def _luminance(color: str) -> float:
    r, g, b = (int(color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _region_traces(
    fig: go.Figure, tracks: Sequence[Track], span: list[int], cell_px: float
) -> None:
    """Each track as a heatmap of region index over (lane, residue), with names written on
    the regions wide enough for them."""
    if not span:
        return
    col = {r: i for i, r in enumerate(span)}
    for i, t in enumerate(tracks):
        names = list(t.regions)
        k = len(names)
        z = np.full((t.n_lanes, len(span)), np.nan)
        text = np.full(z.shape, "", dtype=object)
        for j, name in enumerate(names):
            first, last = t.regions[name]
            lane = t.lane[name]
            for r in range(max(first, span[0]), min(last, span[-1]) + 1):
                z[lane, col[r]] = j
                text[lane, col[r]] = (
                    f"{t.label + ' ' if t.label else ''}{name}: {first}–{last}"
                )
        scale = []
        for j, name in enumerate(names):
            scale += [[j / k, t.colors[name]], [(j + 1) / k, t.colors[name]]]
        axis = f"y{i + 3}"
        fig.add_trace(
            go.Heatmap(
                x=span,
                y=list(range(t.n_lanes)),
                z=z,
                text=text,
                customdata=np.full(z.shape, "region", dtype=object),
                hoverinfo="text",
                colorscale=scale,
                zmin=-0.5,
                zmax=k - 0.5,
                showscale=False,
                xgap=0,
                ygap=1 if t.n_lanes > 1 else 0,
                xaxis="x",
                yaxis=axis,
                name=t.label or f"regions {i}",
            )
        )
        for name in names:
            first, last = t.regions[name]
            first, last = max(first, span[0]), min(last, span[-1])
            if first > last or (last - first + 1) * cell_px < len(name) * 6 + 6:
                continue
            fig.add_annotation(
                x=(first + last) / 2,
                y=t.lane[name],
                xref="x",
                yref=axis,
                text=name,
                showarrow=False,
                font=dict(
                    size=9 if t.n_lanes == 1 else 8,
                    color="white" if _luminance(t.colors[name]) < 0.5 else "black",
                ),
            )


def _hover(row: dict, value_label: str | None = None) -> str:
    value = f"{value_label} = {row[SCORE]:.3f}" if value_label else f"{row[SCORE]:.3f}"
    text = f"{row[VARIANT]}: {value}<br>{row[N_EXPERIMENTS]} experiments"
    if row.get(SIGNIFICANCE):
        stars = row.get(STARS)
        text += f"<br>ClinVar: {row[SIGNIFICANCE]}"
        if stars is not None:
            text += f" ({'★' * int(stars) or '0 stars'})"
        if row.get(CONDITIONS):
            text += f"<br>{row[CONDITIONS]}"
    return text


def effect_map_figure(
    cells: pl.DataFrame,
    per_pos: pl.DataFrame,
    structure: Structure,
    *,
    zrange: tuple[float, float],
    track_range: tuple[float, float],
    track_label: str,
    selected: int | None = None,
    cell_px: float = CELL_PX,
    theme: str | None = None,
    tracks: Sequence[Track] = (),
    clinvar_classes: Collection[str] = (),
    colorscale: str = COLORSCALE,
    value_label: str | None = None,
) -> go.Figure:
    """The effect map at full length: region bars on top, the per-position aggregate below.

    Parameters
    ----------
    cells : pl.DataFrame
        From `map_cells` (with ClinVar columns to mark annotated variants).
    per_pos : pl.DataFrame
        From `aggregate.per_position`, restricted to the structure's residues.
    structure : Structure
        For the wild-type dots and the residue span.
    zrange : tuple[float, float]
        The map's color range (the notebooks' ``vmin``, ``vmax``).
    track_range : tuple[float, float]
        The color range of the aggregate track (the structure's).
    track_label : str
        The track's hover label.
    selected : int | None
        A residue to outline.
    cell_px : float
        Column width in pixels.
    theme : str | None
        ``"light"`` or ``"dark"`` (`THEMES`).
    tracks : Sequence[Track]
        Region bars, top to bottom.
    clinvar_classes : Collection[str]
        `data.CLINVAR_CLASSES` keys whose variants get a triangle.
    colorscale : str
        The plotly colorscale of the map and the track (`colors.COLORSCALES`).
    value_label : str | None
        Names the value in a cell's hover (e.g. ``"z"``); ``None``: the bare value.
    """
    colors = _theme(theme)
    layout = rows(len(tracks))
    # Like the structure, the map shows only the chain's residues.
    cells = cells.filter(pl.col(RESIDUE).is_in(list(structure.residues)))
    span = _span(structure)
    col = {r: i for i, r in enumerate(span)}
    row = {aa: i for i, aa in enumerate(AMINO_ACID_ORDER)}
    z = np.full((len(AMINO_ACID_ORDER), len(span)), np.nan)
    text = np.full(z.shape, "", dtype=object)
    for cell in cells.iter_rows(named=True):
        r, aa = cell[RESIDUE], cell[AA]
        if r in col:
            z[row[aa], col[r]] = cell[SCORE]
            text[row[aa], col[r]] = _hover(cell, value_label)
    for j, r in enumerate(span):
        wt = structure.residues.get(r)
        for aa, i in row.items():
            if not text[i, j]:
                label = f"{wt}{r}{aa}" if wt else f"{r}{aa}"
                text[i, j] = f"{label}: no score"

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=span,
            y=list(AMINO_ACID_ORDER),
            z=z,
            text=text,
            hoverinfo="text",
            colorscale=colorscale,
            zmin=zrange[0],
            zmax=zrange[1],
            showscale=False,
            xgap=0.5 if cell_px >= 4 else 0,
            ygap=0.5,
            name="map",
            xaxis="x",
            yaxis="y",
        )
    )
    wt = [(r, aa) for r, aa in structure.residues.items() if aa in row and r in col]
    if wt:
        fig.add_trace(
            go.Scatter(
                x=[r for r, _ in wt],
                y=[aa for _, aa in wt],
                mode="markers",
                marker=dict(color="black", size=max(2, min(5, cell_px / 2.5))),
                hoverinfo="skip",
                name="wild type",
                xaxis="x",
                yaxis="y",
            )
        )
    if per_pos.height:
        values = per_pos[VALUE].to_list()
        fig.add_trace(
            go.Scatter(
                x=per_pos[RESIDUE].to_list(),
                y=values,
                mode="lines+markers",
                line=dict(color=colors["line"], width=1),
                marker=dict(
                    color=values,
                    colorscale=colorscale,
                    cmin=track_range[0],
                    cmax=track_range[1],
                    size=max(3, min(6, cell_px * 0.6)),
                    line=dict(width=0.5, color=colors["line"]),
                ),
                hovertemplate=f"%{{x}}<br>{track_label}: %{{y:.3f}}<extra></extra>",
                name="aggregate",
                xaxis="x",
                yaxis="y2",
            )
        )
    if CLINVAR_CLASS in cells.columns:
        # Least severe first, so pathogenic triangles draw on top.
        for klass in reversed(list(CLINVAR_CLASSES)):
            if klass not in clinvar_classes:
                continue
            marked = cells.filter(pl.col(CLINVAR_CLASS) == klass)
            if not marked.height:
                continue
            fig.add_trace(
                go.Scatter(
                    x=marked[RESIDUE].to_list(),
                    y=marked[AA].to_list(),
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        color=CLINVAR_CLASSES[klass][1],
                        size=max(4, min(9, cell_px * 0.8)),
                        line=dict(width=0.5, color="white"),
                    ),
                    hoverinfo="skip",
                    name=f"ClinVar {klass}",
                    xaxis="x",
                    yaxis="y",
                )
            )
    _region_traces(fig, tracks, span, cell_px)
    if selected is not None:
        for yref in ("y", "y2"):
            fig.add_shape(
                type="rect",
                xref="x",
                yref=f"{yref} domain",
                x0=selected - 0.5,
                x1=selected + 0.5,
                y0=0,
                y1=1,
                line=dict(color=colors["text"], width=1.5),
                fillcolor="rgba(0,0,0,0)",
            )
    _layout(fig, map_width(structure, cell_px), layout.height, colors, l=0, r=0)
    _axes(fig, layout, tracks)
    fig.update_layout(plot_bgcolor=MISSING_GRAY)
    xrange = [span[0] - 0.5, span[-1] + 0.5] if span else [0, 1]
    # About one tick per 50 pixels, on a round step.
    step = next(
        (s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200) if s * cell_px >= 45), 500
    )
    fig.update_xaxes(
        range=xrange,
        dtick=step,
        tick0=0,
    )
    fig.update_yaxes(showticklabels=False)
    # The rows exactly, top to bottom (no gray padding above A or below R).
    fig.update_layout(
        yaxis_range=[len(AMINO_ACID_ORDER) - 0.5, -0.5],
        yaxis2=dict(
            range=_track_yrange(per_pos), showgrid=True, gridcolor=colors["grid"]
        ),
    )
    # Only the heatmap sits on the gray: the other rows get the panel's background.
    for yref in ["y2", *(f"y{i + 3}" for i in range(len(tracks)))]:
        fig.add_shape(
            type="rect",
            xref="x domain",
            yref=f"{yref} domain",
            x0=0,
            x1=1,
            y0=0,
            y1=1,
            fillcolor=colors["panel"],
            line=dict(width=0),
            layer="below",
        )
    return fig


def labels_figure(
    per_pos: pl.DataFrame,
    track_label: str,
    theme: str | None = None,
    tracks: Sequence[Track] = (),
) -> go.Figure:
    """The amino-acid row labels, the track's y axis and the region bars' labels, to sit
    left of the scrolling map."""
    colors = _theme(theme)
    layout = rows(len(tracks))
    fig = go.Figure()
    # Invisible points give each axis something to draw ticks for.
    fig.add_trace(
        go.Scatter(
            x=[0], y=[AMINO_ACID_ORDER[0]], mode="markers", opacity=0, hoverinfo="skip"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0], mode="markers", opacity=0, hoverinfo="skip", yaxis="y2"
        )
    )
    _layout(fig, LABELS_WIDTH, layout.height, colors, l=LABELS_WIDTH - 2, r=0)
    _axes(fig, layout, tracks)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False)
    fig.update_layout(
        yaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=list(AMINO_ACID_ORDER),
            range=[len(AMINO_ACID_ORDER) - 0.5, -0.5],
            tickmode="array",
            tickvals=list(AMINO_ACID_ORDER),
            tickfont=dict(size=9),
        ),
        yaxis2=dict(
            range=_track_yrange(per_pos),
            nticks=3,
            tickfont=dict(size=9),
            title_text=track_label,
            title_font=dict(size=10),
            title_standoff=2,
        ),
    )
    for i, t in enumerate(tracks):
        lo, hi = layout.regions[i]
        fig.add_annotation(
            x=1,
            y=(lo + hi) / 2,
            xref="paper",
            yref="paper",
            xanchor="right",
            xshift=-4,
            text=t.label or "",
            showarrow=False,
            font=dict(size=9, color=colors["text"]),
        )
    return fig


def colorbar_figure(
    zrange: tuple[float, float],
    zlabel: str,
    track_range: tuple[float, float],
    track_label: str,
    theme: str | None = None,
    n_regions: int = 0,
    colorscale: str = COLORSCALE,
) -> go.Figure:
    """The color bars of the map and of the track, beside their rows, on ``colorscale``."""
    colors = _theme(theme)
    layout = rows(n_regions)
    fig = go.Figure()
    for (low, high), label, domain in (
        (zrange, zlabel, layout.map),
        (track_range, track_label, layout.track),
    ):
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, 0],
                mode="markers",
                hoverinfo="skip",
                marker=dict(
                    color=[low, high],
                    colorscale=colorscale,
                    cmin=low,
                    cmax=high,
                    opacity=0,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text=label, side="right", font=dict(size=10)),
                        thickness=10,
                        x=0,
                        xanchor="left",
                        xpad=4,
                        y=domain[1],
                        yanchor="top",
                        len=domain[1] - domain[0],
                        lenmode="fraction",
                        ypad=0,
                        outlinewidth=0,
                        tickfont=dict(size=9),
                        nticks=6 if domain is layout.map else 3,
                    ),
                ),
            )
        )
    _layout(fig, COLORBAR_WIDTH, layout.height, colors, l=0, r=0)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(visible=False, domain=[0, 0.01])
    fig.update_yaxes(visible=False)
    return fig


def residue_of_click(click_data: Mapping | None) -> int | None:
    """The residue of a map cell or a track point a user clicked (``None`` for a region
    bar)."""
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    if point.get("customdata") == "region":
        return None
    x = point.get("x")
    return int(x) if x is not None else None
