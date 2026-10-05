import polars as pl
import pytest

import fisseqborn as fb
from fisseqborn import _data, fisseq


def test_missing_column_suggests_close_match(profiles):
    with pytest.raises(ValueError, match="meta_variant_type"):
        fb.BoxPlot(profiles, x="meta_varient_type", y="meta_impact_score")


def test_requires_polars(profiles):
    with pytest.raises(TypeError, match="polars"):
        fb.BoxPlot(profiles.to_pandas(), x="meta_variant_type", y="meta_impact_score")


def test_chaining_returns_copies(profiles):
    base = fb.BoxPlot(profiles, x="meta_variant_type", y="meta_impact_score")
    with_line = base.refline(y=0.5)
    assert base._layers == []
    assert len(with_line._layers) == 1
    assert with_line is not base


def test_refline_and_set(profiles):
    _, ax = (
        fb.BoxPlot(profiles, x="meta_variant_type", y="meta_impact_score")
        .refline(y=0.5)
        .set(ylim=(0, 1), ylabel="Impact")
        .plot()
    )
    assert ax.get_ylim() == (0, 1)
    assert ax.get_ylabel() == "Impact"
    assert any(list(line.get_ydata()) == [0.5, 0.5] for line in ax.get_lines())


def test_save_creates_parent_dirs(profiles, tmp_path):
    out = tmp_path / "vis" / "box.png"
    fb.BoxPlot(profiles, x="meta_variant_type", y="meta_impact_score").save(out)
    assert out.stat().st_size > 0


def test_plot_onto_existing_axes(profiles):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2)
    got_fig, got_ax = fb.BoxPlot(
        profiles, x="meta_variant_type", y="meta_impact_score"
    ).plot(ax=axes[1])
    assert got_fig is fig and got_ax is axes[1]


def test_resolve_order_known_then_natural(profiles):
    df = pl.DataFrame({"c": ["Frameshift", "Synonymous", "10", "2", "Single Missense"]})
    assert _data.resolve_order(df, "c") == [
        "Synonymous",
        "Single Missense",
        "Frameshift",
        "2",
        "10",
    ]
    assert _data.resolve_order(df, "c", ["2"]) == ["2"]


def test_resolve_palette():
    assert _data.resolve_palette(["Synonymous", fisseq.PATHOGENIC]) == {
        "Synonymous": "darkgreen",
        fisseq.PATHOGENIC: "red",
    }
    assert len(set(_data.resolve_palette([str(i) for i in range(15)]).values())) == 15
    assert _data.resolve_palette(["a"], {"a": "blue"}) == {"a": "blue"}


def test_batch_palette_groups_tiles():
    palette = fisseq.batch_palette(["T1_R1", "T1_R2", "T2_R1"])
    assert set(palette) == {"T1_R1", "T1_R2", "T2_R1"}
    assert palette["T1_R1"] != palette["T1_R2"]
