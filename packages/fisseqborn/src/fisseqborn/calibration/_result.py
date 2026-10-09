"""The fitted calibration: score thresholds per evidence point, plus what produced them.

stdlib and numpy only.
"""

import dataclasses
import json
import pathlib
from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any

import numpy as np

#: Integer ACMG/AMP evidence strengths: pathogenic 1..8, benign -1..-8.
POINT_VALUES: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
#: Order a score is checked against the point ranges (upstream ExCALIBR's order).
POINT_ORDER: tuple[int, ...] = POINT_VALUES + tuple(-p for p in POINT_VALUES)

FORMAT_VERSION = 1


def assign_points(
    scores: np.ndarray, point_ranges: Mapping[int, Sequence[Sequence[float]]]
) -> np.ndarray:
    """Points for each score: the first point (in `POINT_ORDER`) with a range containing
    it (bounds inclusive), else 0. NaN scores get 0."""
    scores = np.asarray(scores, dtype=float)
    out = np.zeros(scores.shape, dtype=np.int64)
    unset = ~np.isnan(scores)
    for point in POINT_ORDER:
        for lo, hi in point_ranges.get(point, ()):
            hit = unset & (scores >= lo) & (scores <= hi)
            out[hit] = point
            unset &= ~hit
    return out


def posterior_from_lr(lr: np.ndarray, prior: float) -> np.ndarray:
    """Posterior probability of pathogenicity from the local likelihood ratio."""
    lr = np.asarray(lr, dtype=float)
    return lr * prior / ((lr - 1.0) * prior + 1.0)


@dataclasses.dataclass(frozen=True)
class Calibration:
    """An ExCALIBR calibration of one score.

    A score gets ``p`` pathogenic points (``1..8``) when it falls in
    ``point_ranges[p]``, ``p`` benign points (``-1..-8``) in ``point_ranges[-p]``, and 0
    otherwise. Each point is a fixed step in the local positive likelihood ratio,
    ``LR+ >= C ** (p / 8)``, where the Tavtigian constant ``C`` is chosen for ``prior``.
    A range comes from the conservative bootstrap curves: the 5th percentile of
    ``log LR+`` for pathogenic points and the 95th for benign points.

    Attributes
    ----------
    point_ranges : dict[int, list[tuple[float, float]]]
        Score intervals for each point value (an empty list when that strength is never
        reached). Bounds can be ``-inf``/``inf``.
    prior : float
        Prior probability of pathogenicity: the median over bootstrap fits of the
        gnomAD-based estimate, or the value passed in.
    prior_unstable : bool
        One-class (PU/NU) estimate that sits on its 0.01 floor.
    tavtigian_c : int
        The constant ``C`` for ``prior`` (350 at a prior of 0.1).
    mode : str
        ``"standard"`` (pathogenic and benign controls), ``"positive_unlabeled"`` or
        ``"negative_unlabeled"`` (one class, the other recovered from gnomAD).
    n_components : int
        Skew-normal components in the chosen model (2 or 3).
    benign_method : str
        Benign reference used: ``"benign"``, ``"synonymous"`` or ``"avg"``.
    direction : str
        ``"lower_pathogenic"``, ``"higher_pathogenic"`` or ``"both"`` (bidirectional).
    grid : list[float]
        Scores at which the likelihood-ratio curves are evaluated.
    log_lr_low, log_lr_median, log_lr_high : list[float]
        5th percentile, median and 95th percentile of ``log LR+`` across bootstrap fits,
        on ``grid`` (NaN where no fit is defined).
    group_counts : dict[str, int]
        Control variants with a score in each group.
    n_bootstrap, n_valid_fits : int
        Bootstrap iterations run, and those with a usable fit and prior.
    three_component_support : float | None
        Fraction of bootstraps where 3 components beat 2 on held-out likelihood (3 is
        chosen at 0.95 or above); ``None`` when ``n_components`` was fixed.
    bidirectional_votes : float | None
        Fraction of bootstrap fits that looked bidirectional (with ``direction="auto"``).
    reliable : bool
        False when a control group is small or another warning applies; see ``warnings``.
    warnings : list[str]
    settings : dict
        The arguments the calibration was fit with, including ``seed``.
    fits : list[dict]
        Each valid bootstrap fit: its mixture (``params``, ``weights``) and prior.
    """

    point_ranges: dict[int, list[tuple[float, float]]]
    prior: float
    prior_unstable: bool
    tavtigian_c: int
    mode: str
    n_components: int
    benign_method: str
    direction: str
    grid: list[float] = dataclasses.field(repr=False)
    log_lr_low: list[float] = dataclasses.field(repr=False)
    log_lr_median: list[float] = dataclasses.field(repr=False)
    log_lr_high: list[float] = dataclasses.field(repr=False)
    group_counts: dict[str, int]
    n_bootstrap: int
    n_valid_fits: int
    three_component_support: float | None
    bidirectional_votes: float | None
    reliable: bool
    warnings: list[str]
    settings: dict[str, Any]
    fits: list[dict[str, Any]] = dataclasses.field(repr=False)

    # ----- scoring --------------------------------------------------------------------

    def points(self, scores: Any) -> np.ndarray:
        """Evidence points (``-8..8``, 0 for indeterminate) for each score."""
        return assign_points(scores, self.point_ranges)

    def log_lr(self, scores: Any) -> np.ndarray:
        """Median bootstrap ``log LR+`` at each score (linear interpolation on ``grid``,
        held constant beyond its ends)."""
        grid = np.asarray(self.grid)
        curve = np.asarray(self.log_lr_median, dtype=float)
        ok = np.isfinite(curve)
        scores = np.asarray(scores, dtype=float)
        out = np.interp(scores, grid[ok], curve[ok])
        out[np.isnan(scores)] = np.nan
        return out

    def lr(self, scores: Any) -> np.ndarray:
        """Median bootstrap local positive likelihood ratio at each score."""
        return np.exp(self.log_lr(scores))

    def posterior(self, scores: Any) -> np.ndarray:
        """Posterior probability of pathogenicity at each score, from `lr` and ``prior``."""
        return posterior_from_lr(self.lr(scores), self.prior)

    def thresholds(self) -> dict[int, float | None]:
        """For each point value, the score where that strength begins (the bound of its
        range nearest the indeterminate region), or ``None`` if never reached. Only
        meaningful for a one-directional calibration."""
        out: dict[int, float | None] = {}
        higher = self.direction == "higher_pathogenic"
        for point in POINT_ORDER:
            ranges = self.point_ranges.get(point, [])
            if not ranges:
                out[point] = None
                continue
            lows = [lo for lo, _ in ranges]
            highs = [hi for _, hi in ranges]
            # Pathogenic points begin at their bound facing benign, and vice versa.
            facing_high = (point > 0) != higher
            out[point] = float(max(highs) if facing_high else min(lows))
        return out

    # ----- serialization --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["point_ranges"] = {
            str(k): [list(r) for r in v] for k, v in self.point_ranges.items()
        }
        d["format_version"] = FORMAT_VERSION
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Calibration":
        d = dict(d)
        version = d.pop("format_version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported calibration format version {version!r}")
        d["point_ranges"] = {
            int(k): [(float(lo), float(hi)) for lo, hi in v]
            for k, v in d["point_ranges"].items()
        }
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(d) - names)
        if unknown:
            raise ValueError(f"Unknown calibration fields: {unknown}")
        return cls(**d)

    def to_json(self, path: str | PathLike | None = None, **dump_kw: Any) -> str:
        """The calibration as JSON (infinite bounds as ``Infinity``, as upstream ExCALIBR
        writes them); also written to ``path`` if given."""
        text = json.dumps(_plain(self.to_dict()), **dump_kw)
        if path is not None:
            path = pathlib.Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return text

    @classmethod
    def from_json(cls, source: str | PathLike) -> "Calibration":
        """Load a calibration from a JSON string or a path to a JSON file."""
        if isinstance(source, PathLike) or (
            isinstance(source, str) and not source.lstrip().startswith("{")
        ):
            source = pathlib.Path(source).read_text()
        return cls.from_dict(json.loads(source))


def _plain(value: Any) -> Any:
    """numpy scalars and arrays -> Python types, recursively."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
