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
    feature_options,
    link_residue,
    map_figures,
    map_title,
)
from fisseq_structview.colors import MISSING_GRAY, bin_colors
from fisseq_structview.data import (
    CLINVAR_CLASSES,
    ClinvarCalls,
    FeatureProfiles,
    load_distinguishability,
    load_features,
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


@pytest.fixture
def both(feature_run_dir):
    return load_distinguishability(feature_run_dir), load_features(feature_run_dir)


def test_feature_view_is_diverging_around_zero(both, structure):
    run, features = both
    s = ViewSettings(source="features", feature="Area_median")
    assert s.colormap == "RdBu_r" and s.threshold_value == 2.0
    assert s.map_range == (-3.0, 3.0) and s.structure_range == (-3.0, 3.0)
    view = compute_view(run, structure, s, features=features)
    rdbu = bin_colors(colorscale="RdBu_r")
    colors = _colors(view)
    assert colors[4] == int(rdbu[-1].lstrip("#"), 16)  # A4V: z 28, clipped at +3
    assert view.label == "Area_median (z)"
    fig, _, bar = map_figures(view, s, structure)
    assert fig.data[0].colorscale == go_scale("RdBu_r")
    assert "z = " in "".join(t for row in fig.data[0].text for t in row)
    assert bar.data[0].marker.colorbar.title.text == "Area_median (z)"


def go_scale(name):
    import plotly.graph_objects as go

    return go.Heatmap(colorscale=name).colorscale


def test_colormap_defaults_per_source_and_can_be_chosen(both, structure):
    run, features = both
    assert ViewSettings().colormap == "Blues"
    assert ViewSettings(source="features").colormap == "RdBu_r"
    s = ViewSettings(score="auroc_median_fold", colorscale="Reds")
    view = compute_view(run, structure, s, features=features)
    reds = bin_colors(colorscale="Reds")
    assert set(_colors(view).values()) - {int(MISSING_GRAY.lstrip("#"), 16)} <= {
        int(c.lstrip("#"), 16) for c in reds
    }
    fig, _, bar = map_figures(view, s, structure)
    assert fig.data[0].colorscale == go_scale("Reds")
    assert bar.data[0].marker.colorscale == go_scale("Reds")


def test_app_with_features_only(feature_run_dir, structure):
    features = load_features(feature_run_dir)
    app = create_app(None, structure, features=features)
    assert {"plot", "feature", "source", "colorscale", "score"} <= _ids(app.layout)
    assert _by_id(app.layout, "source").value == "features"
    assert _by_id(app.layout, "source-item").style == {"display": "none"}
    assert _by_id(app.layout, "score-item").style == {"display": "none"}
    assert _by_id(app.layout, "feature-item").style == {}
    assert _by_id(app.layout, "feature").value == features.features[0]
    assert _by_id(app.layout, "colorscale").value == "RdBu_r"
    assert _by_id(app.layout, "threshold").value == 2.0
    json.dumps(app.layout.to_plotly_json(), default=str)


def test_app_with_both_sources(both, structure):
    run, features = both
    app = create_app(run, structure, features=features)
    assert _by_id(app.layout, "source-item").style == {}
    assert _by_id(app.layout, "source").value == "ovwt"
    assert _by_id(app.layout, "feature-item").style == {"display": "none"}
    assert _by_id(app.layout, "colorscale").value == "Blues"
    options = _by_id(app.layout, "feature").options
    assert [o["value"] for o in options] == list(features.features)


@pytest.fixture
def many_features():
    features = (
        "Lamin_Intensity_median",
        *(f"Nuclei_Texture_{i}_median" for i in range(300)),
    )
    wide = pl.DataFrame(
        {
            "variant": ["M1V"],
            "variant_type": ["Single Missense"],
            "position": [1],
            "wt": ["M"],
            "n_experiments": [1],
            **{f: [0.5] for f in features},
        }
    )
    return FeatureProfiles(
        source="run",
        batches=(),
        wide=wide,
        features=features,
        labels={"Lamin_Intensity_median": "Lamin A intensity (median)"},
    )


def test_feature_options_are_capped_with_landmarks_first(many_features):
    options = feature_options(many_features, limit=50)
    assert len(options) == 50
    assert options[0] == {
        "label": "Lamin A intensity (median)",
        "value": "Lamin_Intensity_median",
        "search": "Lamin A intensity (median) Lamin_Intensity_median",
    }
    assert options[1] == {
        "label": "Nuclei_Texture_0_median",
        "value": "Nuclei_Texture_0_median",
    }


def test_feature_options_search_every_word_of_name_or_column(many_features):
    def values(search, value=None):
        return [o["value"] for o in feature_options(many_features, search, value)]

    assert values("lamin A") == ["Lamin_Intensity_median"]
    assert values("intensity_MEDIAN") == ["Lamin_Intensity_median"]
    assert values("texture 29") == [
        f"Nuclei_Texture_{i}_median" for i in range(300) if "29" in str(i)
    ]
    assert values("nothing") == []
    # The selected feature stays, so the dropdown doesn't clear it.
    assert values("lamin", "Nuclei_Texture_299_median") == [
        "Lamin_Intensity_median",
        "Nuclei_Texture_299_median",
    ]
    assert values(None, "Nuclei_Texture_299_median")[-1] == "Nuclei_Texture_299_median"
    assert len(values("")) == 100
    assert feature_options(None, "x") == []


def test_feature_dropdown_starts_with_at_most_the_limit(many_features, structure):
    app = create_app(None, structure, features=many_features)
    dropdown = _by_id(app.layout, "feature")
    assert len(dropdown.options) == 100
    assert dropdown.labels == {"search": "Search 301 features"}
