"""The colors of the variant-effect notebooks, shared by the structure and the effect map.

Scores are drawn like ``fb.VariantEffectMap(..., cmap="Blues", vmin=0.5, vmax=1.0,
center=None)``: a sequential Blues scale from AUROC 0.5 (chance) to 1.0. Values outside the
range take the end colors, as matplotlib clips them, and missing values are fisseqborn's
``missing_color="0.6"`` gray. The viewer gets the scale quantized into `N_BINS` colors (one
viewer component per color; see `viewer`).
"""

from collections.abc import Mapping

import plotly.colors as pc

#: The colorscale of the notebooks' OvWT maps.
COLORSCALE = "Blues"

#: The notebooks' ``vmin`` and ``vmax`` for distinguishability scores.
DEFAULT_RANGE: tuple[float, float] = (0.5, 1.0)

#: The range of a fraction (the frac_deleterious aggregation).
FRACTION_RANGE: tuple[float, float] = (0.0, 1.0)

#: Missing values: matplotlib's ``"0.6"`` gray, fisseqborn's default ``missing_color``.
MISSING_GRAY = "#999999"

#: Colors the viewer's scale is quantized into.
N_BINS = 33


def normalize(value: float, low: float, high: float) -> float:
    """``value``'s position in ``[low, high]``, clipped to ``[0, 1]``."""
    if high <= low:
        return 0.5
    return min(1.0, max(0.0, (value - low) / (high - low)))


def quantize(t: float, n_bins: int = N_BINS) -> int:
    """The bin of a normalized value ``t`` in ``[0, 1]``."""
    return min(n_bins - 1, max(0, round(t * (n_bins - 1))))


def _hex(color: str) -> str:
    r, g, b = pc.unlabel_rgb(color)
    return f"#{round(r):02X}{round(g):02X}{round(b):02X}"


def bin_colors(n_bins: int = N_BINS) -> list[str]:
    """``n_bins`` hex colors evenly sampled along `COLORSCALE`, lightest first."""
    scale = pc.get_colorscale(COLORSCALE)
    samples = pc.sample_colorscale(scale, [i / (n_bins - 1) for i in range(n_bins)])
    return [_hex(c) for c in samples]


def residue_colors(
    values: Mapping[int, float],
    low: float = DEFAULT_RANGE[0],
    high: float = DEFAULT_RANGE[1],
    n_bins: int = N_BINS,
) -> dict[int, str]:
    """Residue -> hex color of its value on `COLORSCALE` over ``[low, high]`` (binned;
    values outside the range take the end colors)."""
    palette = bin_colors(n_bins)
    return {
        r: palette[quantize(normalize(v, low, high), n_bins)] for r, v in values.items()
    }
