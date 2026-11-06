"""Region bars over the effect map (domains, tiles), as `fisseqborn.RegionTrack` draws them.

Colors come from fisseqborn's palette rules, so a track matches the notebooks' maps.
Overlapping regions (neighbouring library tiles share a few residues) can't share one bar:
fisseqborn stripes them, here they go on separate lanes (`lanes`).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from matplotlib.colors import to_hex

from fisseqborn import _data

from .params import RegionSpec


@dataclass(frozen=True)
class Track:
    """One region bar: name -> inclusive ``(first, last)`` residue range, and colors."""

    label: str | None
    regions: dict[str, tuple[int, int]]
    colors: dict[str, str]
    #: Region name -> its lane (row within the bar), from `lanes`.
    lane: dict[str, int]

    @property
    def n_lanes(self) -> int:
        return max(self.lane.values(), default=0) + 1


def lanes(regions: Mapping[str, tuple[int, int]]) -> dict[str, int]:
    """Region -> lane: each region goes on the first lane whose last region ends before it
    starts, in order of start (greedy interval packing)."""
    ends: list[int] = []
    out = {}
    for name, (first, last) in sorted(regions.items(), key=lambda kv: kv[1]):
        lane = next((i for i, end in enumerate(ends) if end < first), len(ends))
        if lane == len(ends):
            ends.append(last)
        else:
            ends[lane] = last
        out[name] = lane
    return out


def track(spec: RegionSpec) -> Track:
    """A `Track` from a parameters file's region entry."""
    palette = _data.resolve_palette(list(spec.regions), spec.palette)
    colors = {}
    for name in spec.regions:
        try:
            colors[name] = to_hex(palette.get(name, "#bbbbbb"))
        except ValueError:
            raise ValueError(
                f"region {name!r}: {palette.get(name)!r} is not a color"
            ) from None
    return Track(spec.label, dict(spec.regions), colors, lanes(spec.regions))
