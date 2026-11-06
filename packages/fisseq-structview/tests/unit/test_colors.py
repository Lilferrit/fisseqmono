from fisseq_structview.colors import (
    DEFAULT_RANGE,
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
