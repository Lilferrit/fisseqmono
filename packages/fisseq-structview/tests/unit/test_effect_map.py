import math

import polars as pl
import pytest
from structview_testdata import write_pdb

from fisseq_structview.aggregate import RESIDUE, VALUE
from fisseq_structview.app import ViewSettings, compute_view, link_residue, map_figures
from fisseq_structview.data import (
    CLINVAR_CLASS,
    N_EXPERIMENTS,
    POSITION,
    SCORE,
    VARIANT,
    VARIANT_TYPE,
    WT,
    load_distinguishability,
)
from fisseq_structview.effect_map import (
    AA,
    geometry,
    map_cells,
    map_width,
    residue_of_click,
    rows,
)
from fisseq_structview.params import RegionSpec
from fisseq_structview.regions import track
from fisseq_structview.structure import load_structure
from fisseqborn.fisseq import AMINO_ACID_ORDER


def _variants(rows):
    return pl.DataFrame(
        rows,
        schema=[VARIANT, VARIANT_TYPE, POSITION, WT, SCORE, N_EXPERIMENTS],
        orient="row",
    )


def test_map_cells_keeps_substitutions_and_applies_the_offset():
    variants = _variants(
        [
            ("K2E", "Single Missense", 2, "K", 0.7, 2),
            ("K2K", "Synonymous", 2, "K", 0.5, 2),
            ("A4*", "Nonsense", None, None, 0.9, 2),
            ("A4-|Y5-", "3nt Deletion", None, None, 0.8, 1),
            ("Y5F", "Single Missense", 5, "Y", None, 0),
        ]
    )
    cells = map_cells(variants, offset=10)
    assert cells.select(RESIDUE, AA, SCORE).rows() == [(12, "E", 0.7), (12, "K", 0.5)]


@pytest.fixture
def view_inputs(tmp_path, run_dir):
    run = load_distinguishability(run_dir)
    structure = load_structure(write_pdb(tmp_path / "m.pdb"))
    return run, structure


def test_heatmap_cells_and_missing(view_inputs):
    run, structure = view_inputs
    s = ViewSettings(score="auroc_median_fold", corrected=False)
    fig = map_figures(compute_view(run, structure, s), s, structure)[0]
    heatmap = fig.data[0]
    assert list(heatmap.y) == AMINO_ACID_ORDER
    assert list(heatmap.x) == list(range(1, 9))
    z = heatmap.z
    col = list(heatmap.x).index(4)
    assert z[AMINO_ACID_ORDER.index("V")][col] == pytest.approx(0.88)  # A4V
    assert math.isnan(z[AMINO_ACID_ORDER.index("W")][col])  # A4W: no score, gray
    assert (heatmap.zmin, heatmap.zmax, heatmap.colorscale[0][1]) == (
        0.5,
        1.0,
        "rgb(247,251,255)",
    )


def test_wild_type_dots(view_inputs):
    run, structure = view_inputs
    s = ViewSettings(score="auroc_median_fold")
    dots = map_figures(compute_view(run, structure, s), s, structure)[0].data[1]
    assert list(zip(dots.x, dots.y)) == list(structure.residues.items())


def test_track_is_the_aggregate_that_colors_the_structure(view_inputs):
    run, structure = view_inputs
    s = ViewSettings(score="auroc_median_fold", method="max")
    view = compute_view(run, structure, s)
    track = map_figures(view, s, structure)[0].data[2]
    assert list(track.x) == view.table[RESIDUE].to_list()
    assert list(track.y) == view.table[VALUE].to_list()
    frac = ViewSettings(score="auroc_median_fold", method="frac_deleterious")
    assert frac.structure_range == (0.0, 1.0) and frac.map_range == (0.5, 1.0)


def test_selection_outline(view_inputs):
    run, structure = view_inputs
    s = ViewSettings(score="auroc_median_fold")
    fig = map_figures(compute_view(run, structure, s), s, structure, selected=4)[0]
    outlines = [sh for sh in fig.layout.shapes if sh.x0 == 3.5 and sh.x1 == 4.5]
    assert {sh.yref for sh in outlines} == {"y domain", "y2 domain"}


def test_full_length_map_scrolls_instead_of_zooming(view_inputs):
    run, structure = view_inputs
    s = ViewSettings(score="auroc_median_fold")
    view = compute_view(run, structure, s)
    narrow = map_figures(view, s, structure, cell_px=8)[0]
    wide = map_figures(view, s, structure, cell_px=16)[0]
    assert (narrow.layout.width, wide.layout.width) == (64, 128)
    assert map_width(structure, 16) == 128
    assert list(narrow.layout.xaxis.range) == [0.5, 8.5]  # every residue
    for fig in (narrow, wide):
        assert fig.layout.dragmode is False
        axes = [fig.layout[a] for a in fig.layout if a.startswith(("xaxis", "yaxis"))]
        assert axes and all(ax.fixedrange for ax in axes)
    assert geometry(structure, 16) == {"first": 1, "cell": 16.0}


TRACKS = [
    track(RegionSpec({"T1": (1, 5), "T2": (4, 8)}, label="Tile")),
    track(RegionSpec({"Head": (1, 3), "Rod": (4, 8)}, label="Domain")),
]


@pytest.mark.parametrize("tracks", [[], TRACKS])
def test_labels_and_colorbar_line_up_with_the_map(view_inputs, tracks):
    run, structure = view_inputs
    s = ViewSettings(score="auroc_median_fold")
    fig, labels, bar = map_figures(
        compute_view(run, structure, s), s, structure, tracks=tracks
    )
    layout = rows(len(tracks))
    assert (
        fig.layout.height == labels.layout.height == bar.layout.height == layout.height
    )
    assert (
        (fig.layout.margin.t, fig.layout.margin.b)
        == (
            labels.layout.margin.t,
            labels.layout.margin.b,
        )
        == (bar.layout.margin.t, bar.layout.margin.b)
    )
    for f in (fig, labels):
        assert list(f.layout.yaxis.domain) == pytest.approx(layout.map)
        assert list(f.layout.yaxis2.domain) == pytest.approx(layout.track)
        assert list(f.layout.yaxis2.range) == list(fig.layout.yaxis2.range)
        for i, domain in enumerate(layout.regions):
            assert list(f.layout[f"yaxis{i + 3}"].domain) == pytest.approx(domain)
    assert list(labels.layout.yaxis.tickvals) == AMINO_ACID_ORDER
    assert not fig.layout.yaxis.showticklabels


def test_rows_stack_regions_above_the_map():
    layout = rows(2)
    (t1, t2), m, track_ = layout.regions, layout.map, layout.track
    assert t1[0] > t2[1] and t2[0] > m[1] and m[0] > track_[1]
    assert t1[1] == 1.0 and track_[0] == 0.0
    assert layout.height > rows(0).height


def test_region_bars(view_inputs):
    run, structure = view_inputs
    s = ViewSettings(score="auroc_median_fold")
    fig, labels, _ = map_figures(
        compute_view(run, structure, s), s, structure, tracks=TRACKS, cell_px=24
    )
    tile, domain = (t for t in fig.data if t.yaxis in ("y3", "y4"))
    # T1 and T2 overlap at 4-5, so they take two lanes.
    assert tile.z.shape == (2, 8)
    assert tile.z[0][0] == 0 and tile.z[1][4] == 1 and math.isnan(tile.z[1][0])
    assert "Tile T2: 4–8" in tile.text[1][4]
    assert domain.z.shape == (1, 8) and domain.z[0][5] == 1
    assert {a.text for a in fig.layout.annotations} >= {"Head", "Rod"}
    assert {a.text for a in labels.layout.annotations} == {"Tile", "Domain"}
    # Clicking a bar selects nothing.
    assert residue_of_click({"points": [{"x": 4, "customdata": "region"}]}) is None


def _clinvar(rows_):
    return pl.DataFrame(
        rows_,
        schema=[VARIANT, CLINVAR_CLASS, "clinical_significance", "stars", "conditions"],
        orient="row",
    )


def test_clinvar_markers_and_hover(view_inputs):
    run, structure = view_inputs
    clinvar = _clinvar(
        [
            ("A4V", "pathogenic", "Pathogenic", 2, "Laminopathy"),
            ("K2E", "benign", "Benign", 1, None),
        ]
    )
    from fisseq_structview.data import ClinvarCalls

    calls = ClinvarCalls(clinvar, "test.tsv", "202601")
    s = ViewSettings(score="auroc_median_fold", corrected=False)
    view = compute_view(run, structure, s, calls)
    fig = map_figures(view, s, structure)[0]
    marks = {
        t.name: list(zip(t.x, t.y)) for t in fig.data if t.name.startswith("ClinVar")
    }
    assert marks == {"ClinVar benign": [(2, "E")], "ClinVar pathogenic": [(4, "V")]}
    heatmap = fig.data[0]
    hover = heatmap.text[AMINO_ACID_ORDER.index("V")][list(heatmap.x).index(4)]
    assert "ClinVar: Pathogenic (★★)" in hover and "Laminopathy" in hover
    only = ViewSettings(score="auroc_median_fold", map_clinvar=("benign",))
    fig = map_figures(compute_view(run, structure, only, calls), only, structure)[0]
    assert [t.name for t in fig.data if t.name.startswith("ClinVar")] == [
        "ClinVar benign"
    ]


def test_clicking_a_cell_selects_its_residue(view_inputs):
    _, structure = view_inputs
    click = {"points": [{"x": 4, "y": "V", "z": 0.88}]}
    assert link_residue("plot", None, click, structure) == (4, True)
