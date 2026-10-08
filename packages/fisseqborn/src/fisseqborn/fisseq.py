"""fisseq-specific theme: palettes, category orders and LMNA reference data.

Plot classes consult `PALETTE` and `ORDERS` automatically when a hue/x column's levels
are all known here, so fisseq plots get consistent colors without passing a palette.
"""

import colorsys
import re
from collections.abc import Iterable

import seaborn as sns

PATHOGENIC = "Pathogenic/Likely pathogenic"
UNCERTAIN = "Uncertain significance"

VARIANT_TYPE_PALETTE: dict[str, str] = {
    "Single Missense": "grey",
    "Synonymous": "darkgreen",
    "WT": "grey",
    "Frameshift": "purple",
    "3nt Deletion": "grey",
    "Nonsense": "purple",
    "Other": "grey",
}

CLINVAR_PALETTE: dict[str, str] = {
    PATHOGENIC: "red",
    "Pathogenic": "red",
    UNCERTAIN: "gold",
}

PALETTE: dict[str, str] = {**VARIANT_TYPE_PALETTE, **CLINVAR_PALETTE}

VARIANT_TYPE_ORDER: list[str] = [
    "Synonymous",
    "Single Missense",
    "Frameshift",
    "Nonsense",
    "3nt Deletion",
    "WT",
    "Other",
]

CLINVAR_ORDER: list[str] = [
    "Synonymous",
    "Single Missense",
    UNCERTAIN,
    PATHOGENIC,
    "Pathogenic",
]


LMNA_LANDMARK_FEATURES: dict[str, str] = {
    "Mean_NucleiExpanded_Intensity_MeanIntensity_CH1": "Lamin A intensity in the nucleus",
    "Mean_NuclearBoundary_Intensity_MeanIntensity_CH1": "Lamin A intensity at the nuclear boundary",
    "Mean_NucleiExpanded_Granularity_1_CH1": "Lamin A nuclear granularity",
    "Mean_Nuclei_AreaShape_FormFactor": "Nuclear circularity",
    "AreaShape_Area": "Cell Size",
}

LMNA_DOMAIN_REGIONS: dict[str, tuple[int, int]] = {
    "Head": (1, 28),
    "Coil 1A": (29, 67),
    "L1": (68, 77),
    "Coil 1B": (78, 222),
    "Linker 12": (222, 241),
    "Coil 2": (242, 386),
    "Tail": (387, 416),
    "NLS": (417, 420),
    "Ig-fold": (421, 544),
    "Unfolded": (545, 664),
}

#: The 20 amino acids in the order of the paper's variant effect maps (hydrophobic,
#: special, polar, charged), top to bottom on `VariantEffectMap`'s y axis.
AMINO_ACID_ORDER: list[str] = list("AVILGFYWCMPSTNQDEHKR")

# Every known level, in the order it should appear on an axis / legend.
ORDERS: list[list[str]] = [VARIANT_TYPE_ORDER, CLINVAR_ORDER, list(LMNA_DOMAIN_REGIONS)]

#: Library tiles as ``(name, first, last)`` inclusive amino-acid ranges; neighbors overlap.
LMNA_TILES: list[tuple[str, int, int]] = [
    ("T1", 1, 96),
    ("T2", 90, 185),
    ("T3", 178, 273),
    ("T4", 266, 361),
    ("T5", 354, 449),
    ("T6", 442, 535),
    ("T7", 530, 623),
    ("T8", 607, 664),
]

_BATCH_RE = re.compile(r"^(?P<time>T\d+)_(?P<replicate>R\d+)$")


def _parse_batch_name(batch: str) -> tuple[str, str]:
    """Parse ``"T5_R2"`` into ``("T5", "R2")``; unparseable names become their own group."""
    match = _BATCH_RE.match(batch)
    if match:
        return match.group("time"), match.group("replicate")
    return batch, "R0"


def _max_distance_hue_order(n: int) -> list[int]:
    """Indices ``0..n-1`` reordered so consecutive entries land far apart on the color wheel."""
    step = max(1, round(n * 0.4))
    seen: list[int] = []
    used: set[int] = set()
    idx = 0
    while len(seen) < n:
        while idx in used:
            idx = (idx + 1) % n
        seen.append(idx)
        used.add(idx)
        idx = (idx + step) % n
    return seen


def batch_palette(batches: Iterable[str]) -> dict[str, tuple[float, float, float]]:
    """Palette for ``T{tile}_R{replicate}`` experiment names.

    Batches from the same tile share a hue (tiles are spread apart on the color wheel);
    lower replicates are lighter, higher replicates darker.
    """
    batches = list(dict.fromkeys(batches))
    parsed = {b: _parse_batch_name(b) for b in batches}
    time_groups = sorted({t for t, _ in parsed.values()})
    n = len(time_groups)
    if n == 0:
        return {}

    all_hues = sns.color_palette("husl", n)
    time_to_rgb = {
        time_groups[sorted_pos]: all_hues[wheel_pos]
        for sorted_pos, wheel_pos in enumerate(_max_distance_hue_order(n))
    }

    palette: dict[str, tuple[float, float, float]] = {}
    for time_group in time_groups:
        group_batches = sorted(
            (b for b in batches if parsed[b][0] == time_group),
            key=lambda b: parsed[b][1],
        )
        n_rep = len(group_batches)
        h, base_l, s = colorsys.rgb_to_hls(*time_to_rgb[time_group])
        for i, batch in enumerate(group_batches):
            lightness = base_l
            if n_rep > 1:
                lightness = base_l + 0.30 * ((i / (n_rep - 1)) - 0.5)
                lightness = min(max(lightness, 0.15), 0.85)
            palette[batch] = colorsys.hls_to_rgb(h, lightness, s)
    return palette
