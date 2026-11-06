import fisseqborn as fb
from fisseq_structview.params import RegionSpec
from fisseq_structview.regions import lanes, track


def test_overlapping_tiles_alternate_lanes():
    tiles = {t: (a, b) for t, a, b in fb.genes.LMNA.tiles}
    assert lanes(tiles) == {
        "T1": 0, "T2": 1, "T3": 0, "T4": 1, "T5": 0, "T6": 1, "T7": 0, "T8": 1,
    }  # fmt: skip
    assert set(lanes(dict(fb.genes.LMNA.domains)).values()) == {0, 1}  # 222 is shared
    assert lanes({"a": (1, 5), "b": (6, 9)}) == {"a": 0, "b": 0}


def test_colors_follow_fisseqborn_palettes():
    t = track(RegionSpec({"A": (1, 2), "B": (3, 4)}, label="x", palette={"A": "red"}))
    assert t.colors["A"] == "#ff0000" and t.colors["B"] == "#bbbbbb"
    t = track(RegionSpec({"A": (1, 2), "B": (3, 4)}))
    assert all(c.startswith("#") and len(c) == 7 for c in t.colors.values())
    assert t.colors["A"] != t.colors["B"] and t.n_lanes == 1
