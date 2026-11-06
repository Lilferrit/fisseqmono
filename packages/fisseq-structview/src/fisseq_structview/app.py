"""The Dash app: a dash-molstar structure colored by per-residue scores, linked to an
interactive variant effect map.

`create_app` builds the layout from what was loaded: without a run's scores (`RunScores` is
``None``) the score controls and map are left out and the structure is drawn gray. The work of
each callback is a plain function (`compute_view`, `map_figures`, `link_residue`) so it can be
tested without a browser.

Layout: a nav bar with the controls that apply to both panels (score, corrected or raw,
numbering offset), the info popup and the light/dark toggle; then the structure and the map,
each with its own controls below it. Warnings are a modal shown when the page loads (and when a
control change can't be applied).

Linking: clicking a residue in the structure (the viewer's ``focus``) outlines it on the map and
scrolls the map to it; clicking a map cell also selects the residue in the structure and
centers the camera on it. Hovering the map highlights the residue in the structure. Hovering the
structure doesn't reach the map: dash-molstar doesn't report hover back to Dash.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import dash
import dash_molstar
import plotly.graph_objects as go
import polars as pl
from dash import Input, Output, State, dcc, html

from . import viewer
from .aggregate import (
    DEFAULT_METHOD,
    DEFAULT_THRESHOLD,
    DEFAULT_TYPES,
    METHODS,
    RESIDUE,
    VARIANT_TYPES,
    apply_offset,
    filter_types,
    per_position,
)
from .colors import DEFAULT_RANGE, FRACTION_RANGE, residue_colors
from .data import CLINVAR_CLASS, CLINVAR_CLASSES, ClinvarCalls, RunScores, default_score
from .effect_map import (
    CELL_PX,
    CELL_PX_RANGE,
    COLORBAR_WIDTH,
    LABELS_WIDTH,
    THEMES,
    colorbar_figure,
    effect_map_figure,
    geometry,
    labels_figure,
    map_cells,
    map_width,
    residue_of_click,
    rows,
)
from .info import STRUCTURE_COLUMNS, DataInfo, data_info
from .mapping import MappingError, match_positions
from .regions import Track
from .structure import Structure

logger = logging.getLogger(__name__)

#: Delay before framing the structure after the viewer loads (see the "frame" interval).
FRAME_DELAY_MS = 1500

#: pLDDT below which residues are faded, by default (AlphaFold's "low confidence").
DEFAULT_PLDDT_CUTOFF = 70.0

#: Height of the structure viewer, in pixels.
VIEWER_HEIGHT = 560

#: The Mol* canvas background per theme (the panel color of ``assets/structview.css``).
VIEWER_BACKGROUND = {
    theme: int(colors["panel"].lstrip("#"), 16) for theme, colors in THEMES.items()
}

#: Figure config of the map's three figures: no mode bar, no zooming or panning inside them.
STATIC_GRAPH = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


@dataclass(frozen=True)
class ViewSettings:
    """Everything the UI controls choose."""

    score: str | None = None
    corrected: bool = True
    method: str = DEFAULT_METHOD
    types: tuple[str, ...] = DEFAULT_TYPES
    #: The structure's color range; ``None``: `structure_range`'s default.
    structure_vmin: float | None = None
    structure_vmax: float | None = None
    #: The map's color range (the notebooks' ``vmin``/``vmax``); ``None``: `DEFAULT_RANGE`.
    map_vmin: float | None = None
    map_vmax: float | None = None
    threshold: float = DEFAULT_THRESHOLD
    offset: int = 0
    fade: bool = False
    plddt_cutoff: float = DEFAULT_PLDDT_CUTOFF
    check_wt: bool = True
    #: `data.CLINVAR_CLASSES` keys marked on the map and on the structure.
    map_clinvar: tuple[str, ...] = tuple(CLINVAR_CLASSES)
    structure_clinvar: tuple[str, ...] = ()

    @property
    def map_range(self) -> tuple[float, float]:
        """The color range of the map's scores."""
        return _range(self.map_vmin, self.map_vmax, DEFAULT_RANGE)

    @property
    def default_structure_range(self) -> tuple[float, float]:
        """`FRACTION_RANGE` for ``frac_deleterious`` (a fraction), else `DEFAULT_RANGE`."""
        return FRACTION_RANGE if self.method == "frac_deleterious" else DEFAULT_RANGE

    @property
    def structure_range(self) -> tuple[float, float]:
        """The color range of the per-position aggregate, on the structure and the map's
        track."""
        return _range(
            self.structure_vmin, self.structure_vmax, self.default_structure_range
        )


def _range(
    low: float | None, high: float | None, default: tuple[float, float]
) -> tuple[float, float]:
    return (default[0] if low is None else low, default[1] if high is None else high)


@dataclass(frozen=True)
class View:
    """What the viewer and map show for one `ViewSettings`."""

    viewer_data: dict[str, Any]
    #: Per-residue rows of the structure's residues that have a value (`per_position`).
    table: pl.DataFrame
    #: The effect map's cells (`effect_map.map_cells`).
    cells: pl.DataFrame
    #: Warnings about the numbering (residues outside the structure, wild-type mismatches).
    messages: tuple[str, ...] = field(default=())


def coloring_key(viewer_data: dict[str, Any]) -> str:
    """A digest of the viewer components (colors and fading), to tell whether the structure
    needs re-sending."""
    blob = json.dumps(viewer_data.get("component"), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def faded_residues(structure: Structure, settings: ViewSettings) -> set[int]:
    """Residues whose pLDDT is below the cutoff, if fading is on."""
    if not settings.fade:
        return set()
    return {r for r, b in structure.bfactors.items() if b < settings.plddt_cutoff}


def clinvar_sites(cells: pl.DataFrame, classes: Sequence[str]) -> dict[int, str]:
    """Residue -> marker color of the most severe of ``classes`` among its variants' ClinVar
    calls (``cells`` from `effect_map.map_cells`, with ClinVar)."""
    if CLINVAR_CLASS not in cells.columns:
        return {}
    sites: dict[int, str] = {}
    # Least severe first, so the most severe class wins.
    for klass in reversed(list(CLINVAR_CLASSES)):
        if klass in classes:
            residues = cells.filter(pl.col(CLINVAR_CLASS) == klass)[RESIDUE].to_list()
            sites.update(dict.fromkeys(residues, CLINVAR_CLASSES[klass][1]))
    return sites


def compute_view(
    run: RunScores | None,
    structure: Structure,
    settings: ViewSettings,
    clinvar: ClinvarCalls | None = None,
) -> View:
    """Aggregate, map and color for ``settings``.

    Raises
    ------
    MappingError
        If the positions don't line up with the structure (see `match_positions`).
    """
    faded = faded_residues(structure, settings)
    if run is None or settings.score is None:
        return View(
            viewer.viewer_data(structure, {}, faded), pl.DataFrame(), pl.DataFrame()
        )
    variants = run.table(settings.score, settings.corrected)
    table = per_position(
        apply_offset(filter_types(variants, settings.types), settings.offset),
        settings.method,
        settings.threshold,
    )
    match = match_positions(
        table,
        structure.residues,
        offset=settings.offset,
        check_wt=settings.check_wt,
    )
    messages = []
    if match.outside:
        messages.append(
            f"{len(match.outside)} scored positions are outside the structure chain "
            f"(residues {min(match.outside)}-{max(match.outside)} after offset "
            f"{settings.offset}) and are not shown."
        )
    if match.wt_mismatches:
        messages.append(
            f"{len(match.wt_mismatches)} scored residues have a wild type that differs "
            "from the structure."
        )
    colors = residue_colors(match.values, *settings.structure_range)
    cells = map_cells(
        variants, settings.offset, None if clinvar is None else clinvar.table
    )
    sites = clinvar_sites(cells, settings.structure_clinvar)
    return View(
        viewer.viewer_data(structure, colors, faded, markers=sites),
        table.filter(pl.col(RESIDUE).is_in(list(match.values))),
        cells,
        tuple(messages),
    )


def map_figures(
    view: View,
    settings: ViewSettings,
    structure: Structure,
    selected: int | None = None,
    cell_px: float = CELL_PX,
    theme: str | None = None,
    tracks: Sequence[Track] = (),
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """The effect map of ``view`` with ``selected`` outlined and ``tracks`` on top, and its
    labels and color bar figures."""
    method = METHODS.get(settings.method, settings.method)
    score_label = f"{settings.score}" + (" (corrected)" if settings.corrected else "")
    return (
        effect_map_figure(
            view.cells,
            view.table,
            structure,
            zrange=settings.map_range,
            track_range=settings.structure_range,
            track_label=method,
            selected=selected,
            cell_px=cell_px,
            theme=theme,
            tracks=tracks,
            clinvar_classes=settings.map_clinvar,
        ),
        labels_figure(view.table, method, theme, tracks),
        colorbar_figure(
            settings.map_range,
            score_label,
            settings.structure_range,
            method,
            theme,
            len(tracks),
        ),
    )


def map_title(structure: Structure, selected: int | None) -> str:
    if selected is None:
        return "Variant effect map · click a cell or a residue to link the views"
    return f"Variant effect map · residue {structure.residues.get(selected, '?')}{selected}"


def link_residue(
    trigger: str | None,
    focus_data: dict[str, Any] | None,
    click_data: dict[str, Any] | None,
    structure: Structure,
) -> tuple[int | None, bool]:
    """``(residue, from_map)``: the residue just clicked in the viewer (``trigger ==
    "viewer"``) or on the map (``"plot"``), and whether the viewer should zoom to it."""
    if trigger == "plot":
        residue = residue_of_click(click_data)
        return (residue, True) if residue is not None else (None, False)
    if trigger == "viewer":
        return viewer.focused_residue(focus_data, structure.chain), False
    return None, False


def _number(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _settings(
    base: ViewSettings,
    score: str | None,
    corrected: str,
    offset: int | None,
    method: str,
    types: Sequence[str] | None,
    threshold: float | None,
    structure_vmin: float | None,
    structure_vmax: float | None,
    fade: Sequence[str] | None,
    cutoff: float | None,
    map_vmin: float | None,
    map_vmax: float | None,
    map_clinvar: Sequence[str] | None = None,
    structure_clinvar: Sequence[str] | None = None,
) -> ViewSettings:
    """`ViewSettings` from the control values, in the order of `create_app`'s inputs."""
    return ViewSettings(
        score=score,
        corrected=corrected == "corrected",
        method=method or DEFAULT_METHOD,
        types=tuple(types or ()),
        structure_vmin=_number(structure_vmin),
        structure_vmax=_number(structure_vmax),
        map_vmin=_number(map_vmin),
        map_vmax=_number(map_vmax),
        threshold=DEFAULT_THRESHOLD if threshold is None else float(threshold),
        offset=base.offset if offset is None else int(offset),
        fade=bool(fade),
        plddt_cutoff=DEFAULT_PLDDT_CUTOFF if cutoff is None else float(cutoff),
        check_wt=base.check_wt,
        map_clinvar=tuple(map_clinvar or ()),
        structure_clinvar=tuple(structure_clinvar or ()),
    )


# --- layout pieces -------------------------------------------------------------------------


def _control(label: str, component: Any, wide: bool = False) -> html.Div:
    return html.Div(
        [html.Label(label), component],
        className="sv-control" + (" sv-control-wide" if wide else ""),
    )


def _nav_item(label: str, component: Any) -> html.Div:
    return html.Div([html.Label(label), component], className="sv-nav-item")


def _range_inputs(prefix: str, value: tuple, placeholder: tuple) -> html.Div:
    return html.Div(
        [
            dcc.Input(
                id=f"{prefix}-{end}",
                type="number",
                value=v,
                placeholder=f"{p}",
                step=0.01,
                debounce=True,
            )
            for end, v, p in zip(("vmin", "vmax"), value, placeholder)
        ],
        className="sv-pair",
    )


def modal(name: str, title: str, body: Any, is_open: bool = False) -> html.Div:
    """A dialog over the page: ``{name}-modal`` (class ``open`` shows it), closed by
    ``{name}-close``, ``{name}-backdrop`` or Esc (``assets/structview.js``)."""
    return html.Div(
        [
            html.Div(id=f"{name}-backdrop", className="sv-modal-backdrop", n_clicks=0),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(title),
                            html.Button(
                                "×",
                                id=f"{name}-close",
                                className="sv-modal-close",
                                n_clicks=0,
                                title="Close",
                            ),
                        ],
                        className="sv-modal-head",
                    ),
                    html.Div(body, id=f"{name}-body", className="sv-modal-body"),
                ],
                className="sv-modal-card",
                role="dialog",
            ),
        ],
        id=f"{name}-modal",
        className="sv-modal" + (" open" if is_open else ""),
    )


def _warnings(messages: Sequence[str]) -> list[html.P]:
    return [html.P(m, className="sv-warning") for m in messages]


def _info_body(info: DataInfo, messages: Sequence[str]) -> html.Div:
    def table(section):
        return html.Table(
            html.Tbody([html.Tr([html.Th(k), html.Td(v)]) for k, v in section])
        )

    body: list[Any] = []
    for title, section in info.sections.items():
        body += [html.H4(title), table(section)]
        if title == "Run":
            body += [
                html.H4(f"Structures ({len(info.structures)})"),
                html.Table(
                    [
                        html.Thead(html.Tr([html.Th(c) for c in STRUCTURE_COLUMNS])),
                        html.Tbody(
                            [
                                html.Tr([html.Td(v) for v in row])
                                for row in info.structures
                            ]
                        ),
                    ]
                ),
            ]
    if info.batches:
        body += [
            html.H4(f"Batches ({len(info.batches)})"),
            html.Table(
                [
                    html.Thead(
                        html.Tr(
                            [html.Th("Batch"), html.Th("OvWT rows", className="num")]
                        )
                    ),
                    html.Tbody(
                        [
                            html.Tr(
                                [
                                    html.Td(batch),
                                    html.Td(
                                        "—" if n is None else f"{n:,}", className="num"
                                    ),
                                ]
                            )
                            for batch, n in info.batches
                        ]
                    ),
                ]
            ),
        ]
    body += [
        html.H4("Warnings"),
        html.Div(_warnings(messages) or "None.", id="info-warnings"),
    ]
    return html.Div(body, className="sv-info")


def _viewer(data: dict[str, Any] | None, theme: str) -> html.Div:
    """The Mol* viewer in a wrapper whose id names the theme. dash-molstar reads its layout
    (and so the background) only when it mounts; a new wrapper id makes React mount a new
    viewer."""
    return html.Div(
        dash_molstar.MolstarViewer(
            id="viewer",
            data=data,
            # Keep the camera when a recolor re-sends the structure; let clicks pick faded
            # (low-pLDDT) residues too.
            layout={
                "manualReset": True,
                "pickingAlphaThreshold": viewer.FADED_ALPHA / 2,
                "backgroundColor": VIEWER_BACKGROUND[theme],
            },
            style={
                "width": "100%",
                "height": f"{VIEWER_HEIGHT}px",
                "position": "relative",
            },
        ),
        id=f"viewer-{theme}",
    )


# --- browser-side callbacks ----------------------------------------------------------------

# Set the theme before the page renders, so a dark page doesn't flash light (the "theme"
# store is localStorage["theme"], JSON-encoded).
_EARLY_THEME = """<script>
try {
    const stored = JSON.parse(localStorage.getItem("theme"));
    const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = stored || (dark ? "dark" : "light");
} catch (e) {}
</script>"""

_THEME_JS = """
function(n, stored) {
    let theme = stored;
    if (!theme) {
        const dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        theme = dark ? "dark" : "light";
    }
    if (n) theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    return [theme, theme === "dark" ? "☀" : "☾"];
}
"""

# Scroll the map: to a residue clicked in the structure, or, after a zoom, back to the residue
# that was in the middle. Re-applied after a moment, as the figure may not have its new width
# yet.
_SCROLL_JS = """
function(geometry, selected, prev) {
    const box = document.getElementById("map-scroll");
    if (!box || !geometry) return geometry;
    const fired = dash_clientside.callback_context.triggered.map(t => t.prop_id);
    let target = null;
    if (fired.some(p => p.startsWith("selected.")) && selected && selected.scroll) {
        target = selected.selected;
    } else if (fired.some(p => p.startsWith("geometry.")) && prev && prev.cell !== geometry.cell) {
        target = prev.first + (box.scrollLeft + box.clientWidth / 2) / prev.cell - 0.5;
    }
    if (target === null) return geometry;
    const go = () => {
        box.scrollLeft = (target - geometry.first + 0.5) * geometry.cell - box.clientWidth / 2;
    };
    go();
    setTimeout(go, 60);
    setTimeout(go, 400);
    return geometry;
}
"""


def _clinvar_options() -> list[dict[str, Any]]:
    return [
        {
            "label": html.Span(
                [html.Span("▲ ", style={"color": color}), name],
                style={"marginLeft": "4px"},
            ),
            "value": key,
        }
        for key, (name, color) in CLINVAR_CLASSES.items()
    ]


def _fade_options(plddt: bool) -> list[dict[str, Any]]:
    return [{"label": " fade pLDDT below", "value": "fade", "disabled": not plddt}]


def create_app(
    run: RunScores | None,
    structures: Structure | Sequence[Structure],
    settings: ViewSettings | None = None,
    *,
    plddt: bool | None = None,
    notes: Sequence[str] = (),
    labels: Sequence[str] | None = None,
    offsets: Sequence[int | None] | None = None,
    tracks: Sequence[Track] = (),
    clinvar: ClinvarCalls | None = None,
    params_source: Any = None,
) -> dash.Dash:
    """Build the app.

    Parameters
    ----------
    run : RunScores | None
        The run's scores; ``None`` leaves the score controls and map out.
    structures : Structure | Sequence[Structure]
        One structure, or several to choose from in the nav bar.
    settings : ViewSettings | None
        Initial control values (the score defaults to `data.default_score`).
    plddt : bool | None
        Whether the B-factors are pLDDT, enabling the fade toggle; ``None`` uses each
        structure's `Structure.has_plddt`.
    notes : Sequence[str]
        Warnings for the load-time modal (e.g. what was missing from the run).
    labels : Sequence[str] | None
        The structures' names in the dropdown (default: their file names).
    offsets : Sequence[int | None] | None
        Per structure numbering offset; ``None`` (or a ``None`` entry) uses
        ``settings.offset``.
    tracks : Sequence[Track]
        Region bars drawn over the map.
    clinvar : ClinvarCalls | None
        ClinVar calls to mark on the map and, if toggled, the structure.
    params_source : path | None
        The parameters file, for the info popup.

    Raises
    ------
    MappingError
        If no structure lines up with the run's positions (the first structure's error).
    """
    if isinstance(structures, Structure):
        structures = [structures]
    structures = list(structures)
    if not structures:
        raise ValueError("create_app needs at least one structure")
    settings = settings or ViewSettings()
    if run is not None and settings.score is None:
        settings = ViewSettings(
            **{**settings.__dict__, "score": default_score(run.scores)}
        )
    labels = list(labels or [s.path.name for s in structures])
    offsets = [
        settings.offset if o is None else o
        for o in (offsets or [None] * len(structures))
    ]
    plddt_forced = plddt is not None
    plddts = [s.has_plddt if plddt is None else plddt for s in structures]

    # Each structure's initial view; one that doesn't line up stays selectable, with a
    # warning, and the first one that does is shown first.
    views: dict[int, View] = {}
    errors: dict[int, MappingError] = {}
    for i, structure in enumerate(structures):
        try:
            views[i] = compute_view(
                run,
                structure,
                ViewSettings(**{**settings.__dict__, "offset": offsets[i]}),
                clinvar,
            )
        except MappingError as err:
            errors[i] = err
    if not views:
        raise errors[0]
    current = min(views)
    settings = ViewSettings(**{**settings.__dict__, "offset": offsets[current]})
    initial = views[current]
    structure = structures[current]
    plddt0 = plddts[current]
    load_warnings = [
        *notes,
        *[
            f"{labels[i]}: its numbering doesn't match the run ({err})"
            for i, err in errors.items()
        ],
        *initial.messages,
    ]
    info = data_info(
        run,
        structures,
        plddt=plddts,
        plddt_forced=plddt_forced,
        labels=labels,
        offsets=offsets,
        params_source=params_source,
        clinvar=clinvar,
        tracks=tracks,
    )
    layout_rows = rows(len(tracks))

    app = dash.Dash(__name__, title="fisseq-structview")
    app.index_string = app.index_string.replace("{%metas%}", _EARLY_THEME + "{%metas%}")

    # Nav bar: what applies to both panels, the info popup and the theme.
    def brand(s: Structure) -> str:
        return f"{s.path.name} · chain {s.chain} · {len(s.residues)} residues"

    nav: list[Any] = [
        html.Div(
            [html.B("fisseq-structview"), html.Span(brand(structure), id="brand")],
            className="sv-brand",
        ),
        html.Div(
            _nav_item(
                "Structure",
                dcc.Dropdown(
                    [
                        {
                            "label": label
                            + (" (numbering doesn't match)" if i in errors else ""),
                            "value": i,
                        }
                        for i, label in enumerate(labels)
                    ],
                    current,
                    id="structure",
                    clearable=False,
                    style={"width": "240px"},
                ),
            ),
            style={} if len(structures) > 1 else {"display": "none"},
        ),
    ]
    if run is not None:
        nav += [
            _nav_item(
                "Score",
                dcc.Dropdown(
                    run.scores,
                    settings.score,
                    id="score",
                    clearable=False,
                    style={"width": "220px"},
                ),
            ),
            _nav_item(
                "Batches",
                dcc.RadioItems(
                    [
                        {"label": " corrected", "value": "corrected"},
                        {"label": " raw", "value": "raw"},
                    ],
                    "corrected" if settings.corrected else "raw",
                    id="corrected",
                    inline=True,
                ),
            ),
            _nav_item(
                "Offset",
                html.Div(
                    dcc.Input(
                        id="offset",
                        type="number",
                        value=settings.offset,
                        step=1,
                        debounce=True,
                        style={"width": "64px"},
                    ),
                    title="Structure residue = score position + offset",
                ),
            ),
        ]
    nav += [
        html.Div(className="sv-nav-spacer"),
        html.Button(
            "ⓘ",
            id="info-open",
            className="sv-icon-button",
            title="About the data",
            n_clicks=0,
        ),
        html.Button(
            "☾", id="theme-toggle", className="sv-icon-button", title="Light / dark"
        ),
    ]
    hidden = {"display": "none"}
    clinvar_style = {} if clinvar is not None else hidden

    # Structure panel and its controls.
    structure_controls: list[Any] = []
    if run is not None:
        structure_controls += [
            _control(
                "Aggregate per position",
                dcc.Dropdown(
                    [{"label": v, "value": k} for k, v in METHODS.items()],
                    settings.method,
                    id="method",
                    clearable=False,
                    style={"width": "200px"},
                ),
            ),
            _control(
                "Variant classes",
                dcc.Checklist(list(VARIANT_TYPES), list(settings.types), id="types"),
            ),
            _control(
                "Deleterious threshold (score >)",
                dcc.Input(
                    id="threshold",
                    type="number",
                    value=settings.threshold,
                    step=0.01,
                    debounce=True,
                    style={"width": "80px"},
                ),
            ),
            _control(
                "Color range (Blues)",
                _range_inputs(
                    "structure",
                    (settings.structure_vmin, settings.structure_vmax),
                    settings.default_structure_range,
                ),
            ),
            html.Div(
                _control(
                    "ClinVar sites (ball-and-stick)",
                    dcc.Checklist(
                        _clinvar_options(),
                        list(settings.structure_clinvar),
                        id="structure-clinvar",
                    ),
                ),
                style=clinvar_style,
            ),
        ]
    structure_controls += [
        _control(
            "Confidence",
            html.Div(
                [
                    dcc.Checklist(
                        _fade_options(plddt0),
                        ["fade"] if settings.fade and plddt0 else [],
                        id="fade",
                    ),
                    dcc.Input(
                        id="cutoff",
                        type="number",
                        value=settings.plddt_cutoff,
                        min=0,
                        max=100,
                        step=1,
                        debounce=True,
                        disabled=not plddt0,
                        style={"width": "64px"},
                    ),
                ],
                className="sv-pair",
                style={"alignItems": "center"},
            ),
        ),
        html.Div(
            "The B-factors don't look like pLDDT, so fading is off.",
            id="plddt-hint",
            className="sv-hint",
            style={} if not plddt0 else hidden,
        ),
    ]
    panels: list[Any] = [
        html.Div(
            [
                html.Div(
                    "Structure · click a residue to find it on the map"
                    if run is not None
                    else "Structure",
                    className="sv-card-title",
                ),
                # Empty until the theme is known (see `_viewer_theme`): replacing a viewer
                # that is still loading its structure breaks dash-molstar.
                html.Div(_viewer(None, "light"), id="viewer-box"),
                html.Div(structure_controls, className="sv-controls"),
            ],
            className="sv-card",
        )
    ]

    # Map panel and its controls.
    if run is not None:
        fig, labels_fig, colorbar = map_figures(
            initial, settings, structure, tracks=tracks
        )
        height = f"{layout_rows.height}px"
        panels.append(
            html.Div(
                [
                    html.Div(
                        map_title(structure, None),
                        id="map-title",
                        className="sv-card-title",
                    ),
                    html.Div(
                        [
                            dcc.Graph(
                                id="map-labels",
                                figure=labels_fig,
                                config=STATIC_GRAPH,
                                style={"width": f"{LABELS_WIDTH}px", "height": height},
                            ),
                            html.Div(
                                dcc.Graph(
                                    id="plot",
                                    figure=fig,
                                    config=STATIC_GRAPH,
                                    clear_on_unhover=True,
                                    style=_plot_style(structure, CELL_PX, len(tracks)),
                                ),
                                id="map-scroll",
                                className="sv-map-scroll",
                            ),
                            dcc.Graph(
                                id="map-colorbar",
                                figure=colorbar,
                                config=STATIC_GRAPH,
                                style={
                                    "width": f"{COLORBAR_WIDTH}px",
                                    "height": height,
                                },
                            ),
                        ],
                        className="sv-map-row",
                    ),
                    html.Div(
                        [
                            _control(
                                "Zoom (cell width, px)",
                                dcc.Slider(
                                    *CELL_PX_RANGE,
                                    step=1,
                                    value=CELL_PX,
                                    id="zoom",
                                    marks={v: str(v) for v in (2, 4, 8, 12, 16, 24)},
                                    updatemode="mouseup",
                                ),
                                wide=True,
                            ),
                            _control(
                                "Color range (Blues)",
                                _range_inputs(
                                    "map",
                                    (settings.map_vmin, settings.map_vmax),
                                    DEFAULT_RANGE,
                                ),
                            ),
                            html.Div(
                                _control(
                                    "ClinVar",
                                    dcc.Checklist(
                                        _clinvar_options(),
                                        list(settings.map_clinvar),
                                        id="map-clinvar",
                                    ),
                                ),
                                style=clinvar_style,
                            ),
                            html.Div(
                                "Scroll sideways (shift + wheel, or the scrollbar). Gray: no "
                                "score; dots: wild type. Below: the per-position value that "
                                "colors the structure.",
                                className="sv-hint",
                                style={"flex": "1 1 100%"},
                            ),
                        ],
                        className="sv-controls",
                    ),
                ],
                className="sv-card",
            )
        )

    app.layout = html.Div(
        [
            html.Div(nav, className="sv-nav"),
            html.Div(panels, className="sv-panels"),
            modal(
                "warnings", "Warnings", _warnings(load_warnings), bool(load_warnings)
            ),
            modal("info", "About the data", _info_body(info, load_warnings)),
            # manualReset also skips framing the first load, and dash-molstar loads its data
            # asynchronously without reporting when it's done: frame the chain shortly after
            # a viewer mounts (or the structure changes).
            dcc.Interval(
                id="frame", interval=FRAME_DELAY_MS, max_intervals=1, disabled=True
            ),
            dcc.Store(id="theme", storage_type="local"),
            # The theme of the mounted viewer; None: the empty placeholder.
            dcc.Store(id="viewer-theme"),
            dcc.Store(id="selected"),
            dcc.Store(id="geometry", data=geometry(structure, CELL_PX)),
            dcc.Store(id="geometry-prev", data=geometry(structure, CELL_PX)),
            dcc.Store(id="coloring", data=coloring_key(initial.viewer_data)),
        ]
    )

    def current_structure(index: Any) -> Structure:
        return structures[int(index) if index is not None else current]

    app.clientside_callback(
        _THEME_JS,
        Output("theme", "data"),
        Output("theme-toggle", "children"),
        Input("theme-toggle", "n_clicks"),
        State("theme", "data"),
    )

    @app.callback(
        Output("viewer", "focus", allow_duplicate=True),
        Input("frame", "n_intervals"),
        State("structure", "value"),
        prevent_initial_call=True,
    )
    def _frame(n, index):
        return viewer.frame_chain(current_structure(index)) if n else dash.no_update

    @app.callback(
        Output("viewer-box", "children"),
        Output("viewer-theme", "data"),
        Output("frame", "n_intervals"),
        Output("frame", "disabled"),
        Input("theme", "data"),
        State("viewer-theme", "data"),
        State("viewer", "data"),
        prevent_initial_call=True,
    )
    def _viewer_theme(theme, mounted, data):
        theme = theme if theme in VIEWER_BACKGROUND else "light"
        if theme == mounted:
            return (dash.no_update,) * 4
        # The first mount gets the initial coloring; later ones whatever is shown.
        data = initial.viewer_data if mounted is None else data
        return _viewer(data, theme), theme, 0, False

    @app.callback(
        Output("warnings-modal", "className", allow_duplicate=True),
        Output("info-modal", "className"),
        Input("warnings-close", "n_clicks"),
        Input("warnings-backdrop", "n_clicks"),
        Input("info-open", "n_clicks"),
        Input("info-close", "n_clicks"),
        Input("info-backdrop", "n_clicks"),
        prevent_initial_call=True,
    )
    def _modals(*_):
        if dash.ctx.triggered_id == "info-open":
            return dash.no_update, "sv-modal open"
        if str(dash.ctx.triggered_id).startswith("warnings"):
            return "sv-modal", dash.no_update
        return dash.no_update, "sv-modal"

    @app.callback(
        Output("brand", "children"),
        Output("fade", "options"),
        Output("cutoff", "disabled"),
        Output("plddt-hint", "style"),
        Output("frame", "n_intervals", allow_duplicate=True),
        Output("frame", "disabled", allow_duplicate=True),
        Input("structure", "value"),
        prevent_initial_call=True,
    )
    def _structure_chrome(index):
        s, has = current_structure(index), plddts[int(index)]
        # Frame the new chain once it has loaded (manualReset keeps the old camera).
        return (
            brand(s),
            _fade_options(has),
            not has,
            {} if not has else hidden,
            0,
            False,
        )

    if run is None:

        @app.callback(
            Output("viewer", "data"),
            Input("fade", "value"),
            Input("cutoff", "value"),
            Input("structure", "value"),
            prevent_initial_call=True,
        )
        def _recolor_structure_only(fade, cutoff, index):
            s = ViewSettings(
                fade=bool(fade),
                plddt_cutoff=DEFAULT_PLDDT_CUTOFF if cutoff is None else float(cutoff),
            )
            return compute_view(None, current_structure(index), s).viewer_data

        return app

    @app.callback(
        Output("structure-vmin", "placeholder"),
        Output("structure-vmax", "placeholder"),
        Input("method", "value"),
    )
    def _structure_placeholders(method):
        low, high = ViewSettings(method=method).default_structure_range
        return f"{low}", f"{high}"

    @app.callback(
        Output("offset", "value"),
        Output("selected", "data", allow_duplicate=True),
        Input("structure", "value"),
        prevent_initial_call=True,
    )
    def _switch_structure(index):
        # A new `selected` (with no residue) redraws everything for the new structure.
        return offsets[int(index)], {"selected": None, "structure": int(index)}

    control_inputs = [
        Input("score", "value"),
        Input("corrected", "value"),
        Input("offset", "value"),
        Input("method", "value"),
        Input("types", "value"),
        Input("threshold", "value"),
        Input("structure-vmin", "value"),
        Input("structure-vmax", "value"),
        Input("fade", "value"),
        Input("cutoff", "value"),
        Input("map-vmin", "value"),
        Input("map-vmax", "value"),
        Input("map-clinvar", "value"),
        Input("structure-clinvar", "value"),
    ]

    @app.callback(
        Output("viewer", "data"),
        Output("plot", "figure"),
        Output("plot", "style"),
        Output("map-labels", "figure"),
        Output("map-colorbar", "figure"),
        Output("map-title", "children"),
        Output("geometry", "data"),
        Output("coloring", "data"),
        Output("warnings-modal", "className", allow_duplicate=True),
        Output("warnings-body", "children"),
        Output("info-warnings", "children"),
        *control_inputs,
        Input("selected", "data"),
        Input("zoom", "value"),
        Input("theme", "data"),
        State("structure", "value"),
        State("coloring", "data"),
        prevent_initial_call=True,
    )
    def _update(*args):
        *values, selected, zoom, theme, index, current_key = args
        s = _settings(settings, *values)
        structure = current_structure(index)
        try:
            view = compute_view(run, structure, s, clinvar)
        except MappingError as err:
            return (
                *(dash.no_update,) * 8,
                "sv-modal open",
                _warnings([f"Can't apply these settings: {err}"]),
                dash.no_update,
            )
        cell_px = float(zoom or CELL_PX)
        residue = (selected or {}).get("selected")
        figures = map_figures(view, s, structure, residue, cell_px, theme, tracks)
        # Re-send the structure only when its colors changed: a new `data` reloads it.
        key = coloring_key(view.viewer_data)
        warnings = [*notes, *view.messages]
        return (
            view.viewer_data if key != current_key else dash.no_update,
            figures[0],
            _plot_style(structure, cell_px, len(tracks)),
            figures[1],
            figures[2],
            map_title(structure, residue),
            geometry(structure, cell_px),
            key,
            dash.no_update,
            _warnings(warnings),
            _warnings(warnings) or "None.",
        )

    @app.callback(
        Output("selected", "data"),
        Output("viewer", "selection"),
        Output("viewer", "focus"),
        Input("viewer", "focus"),
        Input("plot", "clickData"),
        State("structure", "value"),
        prevent_initial_call=True,
    )
    def _link(focus_data, click_data, index):
        structure = current_structure(index)
        trigger = {"viewer": "viewer", "plot": "plot"}.get(dash.ctx.triggered_id)
        residue, zoom = link_residue(trigger, focus_data, click_data, structure)
        if residue is None:
            return (dash.no_update,) * 3
        if zoom:
            # A map click: select and center the residue in the structure.
            return (
                {"selected": residue, "scroll": False},
                viewer.selection(structure, residue),
                viewer.focus(structure, residue),
            )
        # A structure click: scroll the map to the residue.
        return {"selected": residue, "scroll": True}, dash.no_update, dash.no_update

    app.clientside_callback(
        _SCROLL_JS,
        Output("geometry-prev", "data"),
        Input("geometry", "data"),
        Input("selected", "data"),
        State("geometry-prev", "data"),
    )

    @app.callback(
        Output("viewer", "hover"),
        Input("plot", "hoverData"),
        State("structure", "value"),
        prevent_initial_call=True,
    )
    def _hover(hover_data, index):
        structure = current_structure(index)
        point = (hover_data or {}).get("points", [{}])[0]
        if not point or point.get("customdata") == "region":
            return viewer.selection(structure, None)
        return viewer.selection(structure, int(point["x"]))

    return app


def _plot_style(
    structure: Structure, cell_px: float, n_regions: int = 0
) -> dict[str, str]:
    return {
        "width": f"{map_width(structure, cell_px)}px",
        "height": f"{rows(n_regions).height}px",
    }
