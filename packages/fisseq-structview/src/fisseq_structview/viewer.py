"""The dash-molstar ``data``, ``selection`` and ``focus`` values for a colored structure.

dash-molstar has no color-by-value option. Each color is its own *component* instead: a
uniform-colored cartoon over the residues of that color, faded (cartoon ``alpha``) for
low-confidence residues. Colors are quantized (`colors.N_BINS`), so a structure needs at most
``2 * (N_BINS + 1)`` components plus one per other chain.
"""

from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import Any

from dash_molstar.helpers import (
    create_component,
    get_focus,
    get_selection,
    get_targets,
    parse_molecule,
)
from dash_molstar.utils import Representation, Target

from .colors import MISSING_GRAY
from .structure import Structure

#: Cartoon opacity of faded (low-pLDDT) residues.
FADED_ALPHA = 0.2


def _cartoon(color: str, alpha: float | None) -> Representation:
    rep = Representation("cartoon", "uniform")
    rep.set_color_params({"value": int(color.lstrip("#"), 16)})
    if alpha is not None:
        rep.set_type_params({"alpha": alpha})
    return rep


def _sticks(color: str) -> Representation:
    rep = Representation("ball-and-stick", "uniform")
    rep.set_color_params({"value": int(color.lstrip("#"), 16)})
    return rep


def viewer_data(
    structure: Structure,
    colors: Mapping[int, str],
    faded: Collection[int] = (),
    *,
    fade_alpha: float = FADED_ALPHA,
    markers: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """The viewer's ``data``: ``structure`` as cartoon, residues colored by ``colors``.

    Parameters
    ----------
    structure : Structure
    colors : Mapping[int, str]
        Residue number -> hex color, for ``structure.chain``; residues not in it are
        `colors.MISSING_GRAY`.
    faded : Collection[int]
        Residues drawn at ``fade_alpha`` opacity.
    markers : Mapping[int, str] | None
        Residue -> hex color of residues also drawn as ball-and-stick (ClinVar sites).
    """
    faded = set(faded)
    groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for residue in structure.residues:
        groups[(colors.get(residue, MISSING_GRAY), residue in faded)].append(residue)
    components = [
        create_component(
            f"{'faded ' if fade else ''}{color}",
            get_targets(structure.chain, residues, auth=True),
            _cartoon(color, fade_alpha if fade else None),
        )
        for (color, fade), residues in sorted(groups.items())
    ]
    components += [
        create_component(
            f"chain {chain}",
            get_targets(chain, auth=True),
            _cartoon(MISSING_GRAY, None),
        )
        for chain in structure.chains
        if chain != structure.chain
    ]
    by_color: dict[str, list[int]] = defaultdict(list)
    for residue, color in (markers or {}).items():
        if residue in structure.residues:
            by_color[color].append(residue)
    components += [
        create_component(
            f"sites {color}",
            get_targets(structure.chain, sorted(residues), auth=True),
            _sticks(color),
        )
        for color, residues in sorted(by_color.items())
    ]
    return parse_molecule(
        structure.text,
        fmt=structure.fmt,
        component=components,
        preset={"kind": "empty"},
    )


def residue_target(structure: Structure, residue: int) -> Target:
    return get_targets(structure.chain, [residue], auth=True)


def selection(structure: Structure, residue: int | None) -> dict[str, Any]:
    """The viewer's ``selection`` (or ``hover``) value for one residue; ``None`` clears it."""
    if residue is None:
        return get_selection(None)
    return get_selection(residue_target(structure, residue))


def focus(structure: Structure, residue: int) -> dict[str, Any]:
    """The viewer's ``focus`` value: center the camera on ``residue`` (no interaction
    analysis, so the viewer doesn't write the focus back)."""
    return get_focus(residue_target(structure, residue), analyse=False)


def frame_chain(structure: Structure) -> dict[str, Any]:
    """The viewer's ``focus`` value that frames the whole painted chain."""
    return get_focus(get_targets(structure.chain, auth=True), analyse=False)


def focused_residue(focus_data: Mapping[str, Any] | None, chain: str) -> int | None:
    """The residue number of the viewer's ``focus`` (what the user clicked) on ``chain``,
    or ``None`` if it is elsewhere or empty."""
    if not focus_data:
        return None
    target = Target(dict(focus_data))
    for ch in target.chains:
        if chain not in (ch.auth_name, ch.name):
            continue
        for residue in ch.residues:
            if residue.number is not None:
                return int(residue.number)
    return None
