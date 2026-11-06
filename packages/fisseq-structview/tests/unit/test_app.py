import json

import polars as pl
import pytest
from structview_testdata import write_pdb

from fisseq_structview import viewer
from fisseq_structview.app import (
    ViewSettings,
    clinvar_sites,
    coloring_key,
    compute_view,
    create_app,
    link_residue,
    map_figures,
    map_title,
)
from fisseq_structview.colors import MISSING_GRAY
from fisseq_structview.data import (
    CLINVAR_CLASSES,
    ClinvarCalls,
    load_distinguishability,
)
from fisseq_structview.mapping import MappingError
from fisseq_structview.structure import load_structure


@pytest.fixture
def run(run_dir):
    return load_distinguishability(run_dir)


@pytest.fixture
def structure(tmp_path):
    return load_structure(
        write_pdb(tmp_path / "m.pdb", bfactors=[95, 90, 40, 30, 88, 91, 70, 65])
    )


def _colors(view) -> dict[int, int]:
    """Residue -> color value, from the viewer components."""
    out = {}
    for component in view.viewer_data["component"]:
        color = component["representation"][0]["colorParams"]["value"]
        for target in component["targets"]:
            for res in target["chains"][0]["residues"]:
                out[res["number"]] = color
    return out


def _ids(layout) -> set[str]:
    found = set()

    def walk(node):
        if getattr(node, "id", None):
            found.add(node.id)
        children = getattr(node, "children", None)
        for child in children if isinstance(children, list) else [children]:
            if child is not None and not isinstance(child, str):
                walk(child)

    walk(layout)
    return found


def test_view_colors_scored_residues_and_grays_the_rest(run, structure):
    view = compute_view(run, structure, ViewSettings(score="auroc_median_fold"))
    colors = _colors(view)
    gray = int(MISSING_GRAY.lstrip("#"), 16)
    assert set(colors) == set(range(1, 9))
    assert {r for r, c in colors.items() if c != gray} == {2, 4, 5}
    assert view.table["residue"].to_list() == [2, 4, 5]


def test_switching_score_changes_figure_and_viewer(run, structure):
    a = ViewSettings(score="auroc_median_fold", corrected=False)
    b = ViewSettings(score="auroc_pooled", corrected=False)
    view_a, view_b = compute_view(run, structure, a), compute_view(run, structure, b)
    (fig_a, _, _), (fig_b, _, bar_b) = (
        map_figures(view_a, a, structure),
        map_figures(view_b, b, structure),
    )
    assert fig_a.data[0].z.tolist() != fig_b.data[0].z.tolist()
    assert "auroc_pooled" in bar_b.data[0].marker.colorbar.title.text
    assert view_a.viewer_data != view_b.viewer_data


def test_offset_error_surfaces(run, structure):
    with pytest.raises(MappingError, match="--offset"):
        compute_view(run, structure, ViewSettings(score="auroc_median_fold", offset=50))


def test_fading_uses_plddt(run, structure):
    s = ViewSettings(score="auroc_median_fold", fade=True, plddt_cutoff=70)
    view = compute_view(run, structure, s)
    faded = {
        res["number"]
        for c in view.viewer_data["component"]
        if c["representation"][0].get("typeParams", {}).get("alpha")
        for t in c["targets"]
        for res in t["chains"][0]["residues"]
    }
    assert faded == {3, 4, 8}


def test_title_names_the_selected_residue(structure):
    assert map_title(structure, 4).endswith("residue A4")
    assert "click" in map_title(structure, None)


def test_structure_and_map_ranges_are_independent(run, structure):
    s = ViewSettings(score="auroc_median_fold", structure_vmin=0.7, map_vmax=0.9)
    assert s.structure_range == (0.7, 1.0) and s.map_range == (0.5, 0.9)
    fig, _, bar = map_figures(compute_view(run, structure, s), s, structure)
    assert (fig.data[0].zmin, fig.data[0].zmax) == (0.5, 0.9)
    assert (bar.data[1].marker.cmin, bar.data[1].marker.cmax) == (0.7, 1.0)
    frac = ViewSettings(method="frac_deleterious")
    assert frac.structure_range == (0.0, 1.0) and frac.map_range == (0.5, 1.0)
    assert ViewSettings(
        method="frac_deleterious", structure_vmax=0.5
    ).structure_range == (
        0.0,
        0.5,
    )


def test_link_residue(structure):
    assert link_residue("plot", None, {"points": [{"x": 5, "y": 0.4}]}, structure) == (
        5,
        True,
    )
    focus = viewer.focus(structure, 3)["targets"][0]
    assert link_residue("viewer", focus, None, structure) == (3, False)
    assert link_residue(None, None, None, structure) == (None, False)


def _by_id(layout, id_):
    return next(c for c in _walk_all(layout) if getattr(c, "id", None) == id_)


def test_app_with_scores(run, structure):
    app = create_app(run, structure)
    ids = _ids(app.layout)
    assert {
        "viewer",
        "plot",
        "map-labels",
        "map-colorbar",
        "map-scroll",
        "zoom",
        "score",
        "corrected",
        "offset",
        "method",
        "structure-vmin",
        "structure-vmax",
        "map-vmin",
        "map-vmax",
        "info-open",
        "theme-toggle",
        "warnings-modal",
        "info-modal",
    } <= ids
    assert not {"status", "vmin", "vmax", "window"} & ids
    assert _by_id(app.layout, "score").value == "auroc_median_fold"
    # The test structure covers every scored position: no warnings, so no modal on load.
    assert "open" not in _by_id(app.layout, "warnings-modal").className
    json.dumps(app.layout.to_plotly_json(), default=str)  # serializable


def test_warnings_modal_opens_on_load(structure):
    app = create_app(None, structure, notes=["no OvWT"])
    modal = _by_id(app.layout, "warnings-modal")
    assert "open" in modal.className
    assert "no OvWT" in json.dumps(modal.to_plotly_json(), default=str)


def test_app_without_scores(structure):
    app = create_app(None, structure, notes=["no OvWT"])
    ids = _ids(app.layout)
    assert {"viewer", "fade", "cutoff", "info-open", "theme-toggle"} <= ids
    assert not {"plot", "score", "method", "map-vmin", "offset"} & ids


def _walk_all(node):
    yield node
    children = getattr(node, "children", None)
    for child in children if isinstance(children, list) else [children]:
        if child is not None and not isinstance(child, str):
            yield from _walk_all(child)


def test_several_structures(run, structure, tmp_path):
    longer = load_structure(write_pdb(tmp_path / "long.pdb", sequence="MKTAYIAKQL"))
    app = create_app(
        run, [structure, longer], labels=["Short", "Long"], offsets=[0, None]
    )
    dropdown = _by_id(app.layout, "structure")
    assert [o["label"] for o in dropdown.options] == ["Short", "Long"]
    assert dropdown.value == 0
    s = ViewSettings(score="auroc_median_fold")
    short, long_ = compute_view(run, structure, s), compute_view(run, longer, s)
    assert short.viewer_data != long_.viewer_data
    assert list(map_figures(long_, s, longer)[0].data[0].x) == list(range(1, 11))


def test_unmatched_structure_is_kept_with_a_warning(run, structure, tmp_path):
    other = load_structure(write_pdb(tmp_path / "o.pdb", sequence="GGGGGGGG"))
    app = create_app(run, [other, structure], labels=["Other", "Good"])
    dropdown = _by_id(app.layout, "structure")
    assert dropdown.value == 1  # the first structure that lines up
    assert "doesn't match" in dropdown.options[0]["label"]
    assert "open" in _by_id(app.layout, "warnings-modal").className
    with pytest.raises(MappingError):
        create_app(run, [other])


def test_clinvar_sites_on_the_structure_only_when_toggled(run, structure):
    calls = ClinvarCalls(
        pl.DataFrame(
            {
                "variant": ["A4V", "K2E"],
                "clinvar_class": ["pathogenic", "vus"],
                "clinical_significance": ["Pathogenic", "Uncertain significance"],
                "stars": [2, 1],
                "conditions": [None, None],
            }
        ),
        "x.tsv",
        "202601",
    )
    off = compute_view(run, structure, ViewSettings(score="auroc_median_fold"), calls)
    assert not [
        c for c in off.viewer_data["component"] if c["label"].startswith("sites")
    ]
    on = compute_view(
        run,
        structure,
        ViewSettings(
            score="auroc_median_fold", structure_clinvar=("pathogenic", "vus")
        ),
        calls,
    )
    sites = {
        c["label"]: [
            r["number"] for t in c["targets"] for r in t["chains"][0]["residues"]
        ]
        for c in on.viewer_data["component"]
        if c["label"].startswith("sites")
    }
    assert sites == {
        f"sites {CLINVAR_CLASSES['pathogenic'][1]}": [4],
        f"sites {CLINVAR_CLASSES['vus'][1]}": [2],
    }
    assert clinvar_sites(on.cells, ["pathogenic"]) == {
        4: CLINVAR_CLASSES["pathogenic"][1]
    }
    assert coloring_key(on.viewer_data) != coloring_key(off.viewer_data)
