import math

import numpy as np
import polars as pl
import pytest

import fisseqborn as fb
from fisseqborn import fisseq


@pytest.fixture
def variants() -> pl.DataFrame:
    # positions 10-21 with wild type "A", "L", "A", ...; every substitution but a few
    rows = []
    for pos in range(10, 22):
        wt = "A" if pos % 2 == 0 else "L"
        for i, mut in enumerate(fisseq.AMINO_ACID_ORDER):
            if (pos + i) % 7 == 0 and mut != wt:
                continue  # missing
            rows.append((f"{wt}{pos}{mut}", 0.0 if mut == wt else pos - 15 + i / 10))
    rows += [("A10fs", 99.0), ("A10-", 99.0), ("A10X", 99.0), ("A10*", 99.0),
             ("A10V|L11A", 99.0), ("A10A:downsampled", 99.0), ("WT", 99.0)]  # fmt: skip
    labels, values = zip(*rows)
    return pl.DataFrame({"meta_aa_changes": labels, "z": values})


def test_matrix_layout_and_ignored_labels(variants):
    mat = fb.VariantEffectMap(variants, "z").matrix()
    assert list(mat.index) == fisseq.AMINO_ACID_ORDER
    assert list(mat.columns) == list(range(10, 22))
    assert (
        np.nanmax(mat.to_numpy()) < 99
    )  # fs, deletions, nonsense, |, :tag, WT dropped
    assert mat.loc["V", 10] == pytest.approx(10 - 15 + 0.1)
    assert mat.loc["A", 10] == 0.0  # synonymous
    # (pos + i) % 7 == 0 is missing: position 12 with "I" (i=2) is NaN
    assert math.isnan(mat.loc["I", 12])


def test_duplicates_aggregate_or_raise():
    df = pl.DataFrame({"meta_aa_changes": ["A5V", "A5V", "A5V", "A6L"],
                       "z": [1.0, 2.0, 10.0, 0.0]})  # fmt: skip
    assert fb.VariantEffectMap(df, "z").matrix().loc["V", 5] == 2.0
    assert fb.VariantEffectMap(df, "z", aggregate="mean").matrix().loc["V", 5] == 13 / 3
    with pytest.raises(ValueError, match="aggregate"):
        fb.VariantEffectMap(df, "z", aggregate=None)
    with pytest.raises(ValueError, match="aggregate must be"):
        fb.VariantEffectMap(df, "z", aggregate="sum")


def test_wild_type_dots_and_position_means(variants):
    plot = fb.VariantEffectMap(variants, "z")
    wild_type = plot.wild_type()
    assert wild_type[10] == "A" and wild_type[11] == "L"
    means = plot.position_means()
    assert list(means.index) == list(range(10, 22))
    expected = plot.cells.filter((pl.col("position") == 12) & (pl.col("mut") != "A"))[
        "value"
    ].mean()
    assert means[12] == pytest.approx(expected)

    _, axes = plot.plot()
    [dots] = axes.heatmaps[0].collections
    offsets = dots.get_offsets()
    a_row, l_row = (
        fisseq.AMINO_ACID_ORDER.index("A"),
        fisseq.AMINO_ACID_ORDER.index("L"),
    )
    assert len(offsets) == 12
    assert {tuple(o) for o in offsets} == {
        (p, a_row if p % 2 == 0 else l_row) for p in range(10, 22)
    }


def test_positions_range_and_wrapping(variants):
    plot = fb.VariantEffectMap(variants, "z", positions=(12, 16))
    assert list(plot.matrix().columns) == [12, 13, 14, 15, 16]

    _, axes = fb.VariantEffectMap(
        variants, "z", positions_per_row=5, marginal=True
    ).plot()
    assert len(axes.heatmaps) == math.ceil(12 / 5)
    assert axes.positions == [(10, 14), (15, 19), (20, 21)]
    widths = {ax.get_xlim()[1] - ax.get_xlim()[0] for ax in axes.heatmaps}
    assert widths == {5.0}
    assert all(len(group) == 3 for group in (axes.regions, axes.means, axes.marginals))
    # the short last row has no ticks past its data
    assert max(axes.heatmaps[-1].get_xticks()) <= 21


def test_panels_can_be_turned_off(variants):
    _, axes = fb.VariantEffectMap(variants, "z").plot()
    assert axes.marginals == []  # opt-in
    _, axes = fb.VariantEffectMap(
        variants, "z", regions=None, position_mean=False
    ).plot()
    assert axes.regions == axes.means == axes.marginals == []
    assert len(axes.heatmaps) == 1


def test_color_scale(variants):
    _, axes = fb.VariantEffectMap(variants, "z", vmin=-2, vmax=2).plot()
    image = axes.heatmaps[0].get_images()[0]
    assert (image.norm.vmin, image.norm.vmax) == (-2, 2)
    assert axes.colorbar.get_ylabel() == "z"

    _, axes = fb.VariantEffectMap(variants, "z", cbar_label="score").plot()
    norm = axes.heatmaps[0].get_images()[0].norm
    assert norm.vmin == -norm.vmax  # symmetric around center=0
    assert axes.colorbar.get_ylabel() == "score"


def test_validation_and_dataset_input(variants):
    with pytest.raises(ValueError, match="not found"):
        fb.VariantEffectMap(variants, "missing")
    with pytest.raises(ValueError, match="No single substitutions"):
        fb.VariantEffectMap(pl.DataFrame({"meta_aa_changes": ["WT"], "z": [1.0]}), "z")
    with pytest.raises(ValueError, match="numeric"):
        fb.VariantEffectMap(variants.with_columns(pl.col("z").cast(str)), "z")
    plot = fb.VariantEffectMap(fb.Dataset(variants), "z")
    assert plot.matrix().shape == (20, 12)


def test_layers_apply_to_first_heatmap(variants, tmp_path):
    plot = (
        fb.VariantEffectMap(variants, "z").set(title="layer").save(tmp_path / "m.png")
    )
    assert (tmp_path / "m.png").exists()
    _, axes = plot.plot()
    assert axes.heatmaps[0].get_title() == "layer"


def _offsets(ax) -> set[tuple[float, float]]:
    """Cells marked on ``ax`` beyond the wild-type dots (the first collection)."""
    return {tuple(o) for c in ax.collections[1:] for o in c.get_offsets()}


def test_highlight_marks_matching_cells_on_every_row(variants):
    base = fb.VariantEffectMap(variants, "z", positions_per_row=5)
    where = pl.col("meta_aa_changes").is_in(["A10V", "L11A", "A16C", "WT", "A10fs"])
    plot = base.highlight(where, label="picked")
    assert plot._highlights and not base._highlights  # chaining copies
    fig, axes = plot.plot()
    rows = {aa: i for i, aa in enumerate(fisseq.AMINO_ACID_ORDER)}
    assert _offsets(axes.heatmaps[0]) == {(10, rows["V"]), (11, rows["A"])}
    assert _offsets(axes.heatmaps[1]) == {(16, rows["C"])}
    assert _offsets(axes.heatmaps[2]) == set()
    [legend] = fig.legends
    assert [t.get_text() for t in legend.get_texts()] == ["picked"]
    _, axes = base.plot()
    assert _offsets(axes.heatmaps[0]) == set()


def test_highlight_hue(variants):
    df = variants.with_columns(
        clin=pl.when(pl.col("meta_aa_changes") == "A10V")
        .then(pl.lit("Pathogenic"))
        .when(pl.col("meta_aa_changes") == "A12V")
        .then(pl.lit("Benign"))
    )
    plot = fb.VariantEffectMap(df, "z").highlight(
        pl.col("clin").is_not_null(),
        hue="clin",
        palette={"Pathogenic": "red", "Benign": "blue"},
    )
    _, axes = plot.plot()
    marks = axes.heatmaps[0].collections[1]
    colors = {
        int(x): tuple(c[:3])
        for (x, _), c in zip(marks.get_offsets(), marks.get_facecolors())
    }
    assert colors == {10: (1.0, 0.0, 0.0), 12: (0.0, 0.0, 1.0)}
    with pytest.raises(TypeError, match="categorical"):
        fb.VariantEffectMap(df, "z").highlight(pl.lit(True), hue="z").plot()
    with pytest.raises(ValueError, match="not found"):
        fb.VariantEffectMap(df, "z").highlight(pl.lit(True), hue="missing")


def test_portrait(variants):
    plot = fb.VariantEffectMap(variants, "z", orientation="portrait", marginal=True)
    fig, axes = plot.highlight(pl.col("meta_aa_changes") == "A10V").plot()
    heat = axes.heatmaps[0]
    assert heat.get_images()[0].get_array().shape == (
        12,
        20,
    )  # positions by amino acids
    assert heat.get_ylim() == (21.5, 9.5)  # positions run downward
    rows = {aa: i for i, aa in enumerate(fisseq.AMINO_ACID_ORDER)}
    wild_type = {tuple(o) for o in heat.collections[0].get_offsets()}
    assert (rows["A"], 10) in wild_type and (rows["L"], 11) in wild_type
    assert _offsets(heat) == {(rows["V"], 10)}
    box = heat.get_position()
    width, height = fig.get_size_inches()
    # 20 amino-acid cells across, 12 position cells down
    assert box.width * width == pytest.approx(20 * 0.12)
    assert box.height * height == pytest.approx(12 * 0.12)

    _, axes = fb.VariantEffectMap(
        variants, "z", orientation="portrait", positions_per_row=5, marginal=True
    ).plot()
    lefts = [ax.get_position().x0 for ax in axes.heatmaps]
    assert lefts == sorted(lefts) and len(set(lefts)) == 3  # columns, left to right
    assert all(len(g) == 3 for g in (axes.regions, axes.means, axes.marginals))

    with pytest.raises(ValueError, match="orientation"):
        fb.VariantEffectMap(variants, "z", orientation="sideways")
