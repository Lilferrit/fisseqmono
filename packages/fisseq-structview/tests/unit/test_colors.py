from fisseq_structview.colors import (
    COLORSCALES,
    DEFAULT_RANGE,
    FEATURE_RANGE,
    N_BINS,
    bin_colors,
    normalize,
    quantize,
    residue_colors,
)


def test_range_ends_get_the_ends_of_blues():
    palette = bin_colors()
    colors = residue_colors({1: 0.5, 2: 1.0, 3: 0.75}, *DEFAULT_RANGE)
    assert colors[1] == palette[0] and colors[2] == palette[-1]
    assert colors[3] == palette[N_BINS // 2]
    assert palette[0] == "#F7FBFF" and palette[-1] == "#08306B"  # Blues, lightest first


def test_values_outside_the_range_clip():
    palette = bin_colors()
    colors = residue_colors({1: 0.2, 2: 1.4}, 0.5, 1.0)
    assert colors == {1: palette[0], 2: palette[-1]}


def test_quantize_and_normalize():
    assert quantize(0.5) == N_BINS // 2
    assert quantize(0.0) == 0 and quantize(1.0) == N_BINS - 1
    assert normalize(2.0, 0.0, 1.0) == 1.0 and normalize(-1, 0, 1) == 0.0
    assert normalize(1.0, 1.0, 1.0) == 0.5


def test_colorscales():
    palettes = {c: bin_colors(colorscale=c) for c in COLORSCALES}
    assert len({tuple(p) for p in palettes.values()}) == len(COLORSCALES)
    assert palettes["Reds"][0] == "#FFF5F0" and palettes["Reds"][-1] == "#67000D"
    rdbu = palettes["RdBu_r"]
    colors = residue_colors(
        {1: -3.0, 2: 0.0, 3: 3.0}, *FEATURE_RANGE, colorscale="RdBu_r"
    )
    assert colors == {1: rdbu[0], 2: rdbu[N_BINS // 2], 3: rdbu[-1]}
    assert rdbu[0] == "#053061" and rdbu[-1] == "#67001F"  # blue low, red high
