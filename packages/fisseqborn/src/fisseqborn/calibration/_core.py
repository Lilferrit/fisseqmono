"""ExCALIBR on plain arrays: bootstrap mixture fits -> prior -> local LR -> points.

A native port of the method of Zeiberg et al. (bioRxiv 2025.04.29.651326, v2 and its
Supplementary Information) as implemented in upstream ExCALIBR
(github.com/rosstewart/exCALIBR, MIT). Upstream's GPU (JAX) path is a speed-up only and
is not ported; bootstrap iterations run in parallel with joblib instead.

Each bootstrap iteration:

1. assigns each control variant that is in several groups (e.g. ClinVar-labelled and in
   gnomAD) to one of them at random;
2. resamples every group with replacement at its own size, holding out the variants not
   drawn;
3. fits a skew-normal mixture with components shared across groups (`_mixture`), best
   of ``n_restarts`` on held-out likelihood, for each candidate number of components.

Across iterations, 3 components are kept only if they beat 2 on held-out likelihood in at
least 95% of iterations (Supp. 1.6). For each fit the prior probability of pathogenicity
is estimated on the gnomAD sample (label-shift EM, Supp. 1.5; with one control class, the
unmixing boundary estimator upstream uses), and ``log LR+ = log f_P - log f_B`` is
evaluated on a score grid. Points come from the 5th (pathogenic) and 95th (benign)
percentile ``log LR+`` curves at the median prior, then a monotonicity post-processing
step; out-of-bag points for each control variant use only the fits that held it out.

numpy, scipy, scikit-learn and joblib only.
"""

import dataclasses
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from . import _mixture, _skewnorm
from ._result import POINT_VALUES, Calibration, assign_points

logger = logging.getLogger(__name__)

PATHOGENIC, BENIGN, GNOMAD, SYNONYMOUS = range(4)
GROUPS: tuple[str, ...] = ("pathogenic", "benign", "gnomad", "synonymous")

DIRECTIONS = ("auto", "lower_pathogenic", "higher_pathogenic", "both")
BENIGN_METHODS = ("avg", "benign", "synonymous")

#: Below this many variants a group is dropped from the fit (with a warning).
MIN_GROUP_SIZE = 5
#: Below this many variants in a control group the calibration is flagged unreliable.
RELIABLE_GROUP_SIZE = 10
#: Share of bootstraps where 3 components must beat 2 to choose 3 (Supp. 1.6).
THREE_COMPONENT_SUPPORT = 0.95
#: Share of bootstrap fits that must look bidirectional to treat the assay as such.
BIDIRECTIONAL_VOTE_THRESHOLD = 0.5
#: Out-of-bag points need at least this many fits that held the variant out.
MIN_OOB_FITS = 10
GRID_POINTS = 2000
PATHOGENIC_PERCENTILE, BENIGN_PERCENTILE = 5.0, 95.0
_UNMIX_FLOOR, _UNMIX_DISCARD, _UNMIX_NO_SIGNAL = 0.01, 0.99, 0.1


# ----- Tavtigian constant ------------------------------------------------------------

_FRACS = np.array(
    [1 / 8, 1 / 4, 1 / 2, 1.0]
)  # supporting, moderate, strong, very strong
# Evidence counts per ACMG/AMP combining rule (modified rules, as upstream's default).
_PATHOGENIC_RULES = np.array(
    [[0, 0, 1, 1], [0, 2, 0, 1], [1, 1, 0, 1], [2, 0, 0, 1],
     [0, 3, 1, 0], [2, 2, 1, 0], [4, 1, 1, 0], [0, 1, 0, 1]]
)  # fmt: skip
_LIKELY_PATHOGENIC_RULES = np.array(
    [[0, 1, 1, 0], [2, 0, 1, 0], [0, 3, 0, 0], [2, 2, 0, 0], [4, 1, 0, 0], [0, 0, 2, 0]]
)
_BENIGN_RULES = np.array([[0, 0, 2, 0]])
_LIKELY_BENIGN_RULES = np.array([[1, 0, 1, 0], [2, 0, 0, 0]])


def _posterior(lr: np.ndarray, prior: float) -> np.ndarray:
    return lr * prior / ((lr - 1) * prior + 1)


def _rule_posteriors(
    c: np.ndarray, rules: np.ndarray, sign: float, prior: float
) -> np.ndarray:
    exponents = sign * (rules @ _FRACS)
    return np.round(_posterior(c[:, None] ** exponents[None, :], prior), 3)


def _tavtigian_fails(c: np.ndarray, prior: float) -> np.ndarray:
    c = c.astype(float)
    return (
        (_rule_posteriors(c, _PATHOGENIC_RULES, 1, prior) < 0.99).sum(1)
        + (_rule_posteriors(c, _LIKELY_PATHOGENIC_RULES, 1, prior) < 0.90).sum(1)
        + (_rule_posteriors(c, _BENIGN_RULES, -1, prior) > 0.01).sum(1)
        + (_rule_posteriors(c, _LIKELY_BENIGN_RULES, -1, prior) > 0.10).sum(1)
    )


def tavtigian_constant(prior: float) -> int:
    """The integer ``C`` (odds of pathogenicity for very strong evidence) whose implied
    posteriors break the fewest ACMG/AMP combining rules at ``prior`` (Tavtigian et al.
    2018): a coarse log-spaced scan of ``1..1e15``, then every integer near the best."""
    coarse = np.unique(np.round(np.logspace(0, 15, 600)).astype(np.int64))
    best = int(np.argmin(_tavtigian_fails(coarse, prior)))
    lo = int(coarse[max(best - 1, 0)])
    hi = int(coarse[min(best + 1, len(coarse) - 1)])
    if hi - max(1, lo) + 1 > 250_000:
        fine = np.unique(
            np.round(np.geomspace(max(1, lo), hi, 250_000)).astype(np.int64)
        )
    else:
        fine = np.arange(max(1, lo), hi + 1)
    return int(fine[np.argmin(_tavtigian_fails(fine, prior))])


def point_thresholds(prior: float) -> tuple[np.ndarray, int]:
    """``log LR+`` thresholds for 1..8 pathogenic points (benign: their negatives)."""
    c = tavtigian_constant(prior)
    return np.log(float(c)) * np.asarray(POINT_VALUES) / len(POINT_VALUES), c


# ----- prior --------------------------------------------------------------------------


def label_shift_prior(
    f_pathogenic: np.ndarray,
    f_benign: np.ndarray,
    *,
    max_iter: int = 10_000,
    tol: float = 1e-6,
) -> float:
    """Prior from densities at the gnomAD scores by label-shift EM (Saerens et al. 2002;
    Supp. 1.5): ``prior <- mean(posterior(s))`` from 0.5 until it changes by < ``tol``.
    Returns NaN outside ``(0.001, 0.999)``."""
    prior = 0.5
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = f_benign / f_pathogenic
        for _ in range(max_iter):
            new = float(np.nanmean(1.0 / (1.0 + (1.0 - prior) / prior * ratio)))
            done = abs(new - prior) < tol
            prior = new
            if done or not 0 < prior < 1:
                break
    if not np.isfinite(prior) or prior <= 0.001 or prior >= 0.999:
        return float("nan")
    return prior


def unmixing_prior(f_class: np.ndarray, f_population: np.ndarray, mode: str) -> float:
    """Prior with one control class: the mixture-proportion boundary
    ``min f_population / f_class`` over the gnomAD scores (Blanchard et al. 2010), as
    upstream does. It is the prior itself for positive-unlabeled data and one minus it
    for negative-unlabeled. Floored at 0.01; NaN at 0.99 or above."""
    valid = f_class > 0
    if not valid.any():
        return _UNMIX_NO_SIGNAL
    boundary = float(np.min(f_population[valid] / f_class[valid]))
    prior = boundary if mode == "positive_unlabeled" else 1.0 - boundary
    if prior >= _UNMIX_DISCARD or prior < 0:
        return float("nan")
    return max(prior, _UNMIX_FLOOR)


def _benign_weights(weights: np.ndarray, method: str) -> np.ndarray:
    if method == "avg":
        return (weights[BENIGN] + weights[SYNONYMOUS]) / 2
    return weights[SYNONYMOUS if method == "synonymous" else BENIGN]


def fit_prior(
    mixture: _mixture.Mixture, population: np.ndarray, mode: str, benign_method: str
) -> float:
    """One fit's prior, estimated on the gnomAD scores ``population``."""
    w = mixture.weights
    f_pop = np.exp(mixture.logpdf(population, w[GNOMAD]))
    if mode == "standard":
        f_p = np.exp(mixture.logpdf(population, w[PATHOGENIC]))
        f_b = np.exp(mixture.logpdf(population, _benign_weights(w, benign_method)))
        return label_shift_prior(f_p, f_b)
    if mode == "positive_unlabeled":
        return unmixing_prior(
            np.exp(mixture.logpdf(population, w[PATHOGENIC])), f_pop, mode
        )
    f_b = np.exp(mixture.logpdf(population, _benign_weights(w, benign_method)))
    return unmixing_prior(f_b, f_pop, mode)


def fit_log_densities(
    mixture: _mixture.Mixture,
    grid: np.ndarray,
    prior: float,
    mode: str,
    benign_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """``log f_P`` and ``log f_B`` on ``grid``. With one control class the missing density
    is unmixed from gnomAD: ``f_B = (f_G - prior f_P) / (1 - prior)`` (Supp. Eq. S8, S9),
    floored at ``1e-10 f_G``."""
    w = mixture.weights
    if mode == "standard":
        return (
            mixture.logpdf(grid, w[PATHOGENIC]),
            mixture.logpdf(grid, _benign_weights(w, benign_method)),
        )
    f_pop = np.exp(mixture.logpdf(grid, w[GNOMAD]))
    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "positive_unlabeled":
            log_fp = mixture.logpdf(grid, w[PATHOGENIC])
            f_b = (f_pop - prior * np.exp(log_fp)) / (1 - prior)
            return log_fp, np.log(np.maximum(f_b, f_pop * 1e-10))
        log_fb = mixture.logpdf(grid, _benign_weights(w, benign_method))
        f_p = (f_pop - (1 - prior) * np.exp(log_fb)) / prior
        return np.log(np.maximum(f_p, f_pop * 1e-10)), log_fb


# ----- point ranges -------------------------------------------------------------------

Ranges = dict[int, list[list[float]]]


def _segments(grid: np.ndarray, points: np.ndarray, ranges: Ranges) -> None:
    """Add a ``[start, end]`` range per run of equal nonzero points along ``grid``. A run
    ends at the first grid point of the next run, as upstream's ``get_point_ranges``."""
    change = np.flatnonzero(np.diff(points) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(grid) - 1]])
    for start, end in zip(starts, ends, strict=True):
        point = int(points[start])
        if point != 0:
            ranges[point].append(sorted([float(grid[start]), float(grid[end])]))


def raw_point_ranges(
    grid: np.ndarray, log_lr_low: np.ndarray, log_lr_high: np.ndarray, prior: float
) -> tuple[Ranges, int]:
    """Score ranges reaching each point before post-processing: pathogenic points from
    ``log_lr_low``, benign points from ``log_lr_high``. NaN scores give no evidence."""
    tau, c = point_thresholds(prior)
    ranges: Ranges = {p: [] for p in (*POINT_VALUES, *(-p for p in POINT_VALUES))}
    with np.errstate(invalid="ignore"):
        low = np.where(np.isnan(log_lr_low), -np.inf, log_lr_low)
        high = np.where(np.isnan(log_lr_high), np.inf, log_lr_high)
        pathogenic = np.searchsorted(tau, low, side="right")
        benign = -np.searchsorted(tau, -high, side="right")
    _segments(grid, pathogenic, ranges)
    _segments(grid, benign, ranges)
    return ranges, c


def _enforce_monotonicity(ranges: Ranges, flipped: bool) -> None:
    """Liberal monotonicity (upstream's default): keep one fragment per point (the one
    nearest the benign side for pathogenic points, and vice versa), then drop any point
    whose range lies inside a stronger point's range."""
    for p in POINT_VALUES:
        if ranges[p]:
            ranges[p] = [ranges[p][0] if flipped else ranges[p][-1]]
        if ranges[-p]:
            ranges[-p] = [ranges[-p][-1] if flipped else ranges[-p][0]]
    for i in POINT_VALUES:
        for j in POINT_VALUES:
            if j <= i:
                continue
            for a, b in ((i, j), (-i, -j)):
                if ranges[a] and ranges[b]:
                    if (
                        ranges[a][0][0] >= ranges[b][0][0]
                        and ranges[a][0][1] <= ranges[b][0][1]
                    ):
                        ranges[a] = []


def _extend_to_limits(ranges: Ranges, flipped: bool) -> None:
    """Extend the strongest pathogenic and benign points to -inf/inf on their side, then
    drop any point that spans the whole axis (a misjudged direction)."""
    for p in POINT_VALUES:
        stronger = [q for q in POINT_VALUES if q > p]
        if ranges[p] and not any(ranges[q] for q in stronger):
            ranges[p] = (
                [[ranges[p][0][0], np.inf]]
                if flipped
                else [[-np.inf, ranges[p][-1][-1]]]
            )
        if ranges[-p] and not any(ranges[-q] for q in stronger):
            ranges[-p] = (
                [[-np.inf, ranges[-p][-1][-1]]]
                if flipped
                else [[ranges[-p][0][0], np.inf]]
            )
    for p in (*POINT_VALUES, *(-q for q in POINT_VALUES)):
        if ranges[p] and ranges[p][0][0] == -np.inf and ranges[p][-1][-1] == np.inf:
            ranges[p] = []


def _postprocess_one_sided(ranges: Ranges, flipped: bool) -> None:
    _enforce_monotonicity(ranges, flipped)
    _extend_to_limits(ranges, flipped)
    _enforce_monotonicity(ranges, flipped)


def _clean_benign_no_extend(ranges: Ranges) -> None:
    """Bidirectional benign cleanup (liberal): strongest point first, each takes the
    envelope of its fragments minus what stronger points already claimed; no extension."""
    claimed: list[list[float]] = []
    for p in sorted(POINT_VALUES, reverse=True):
        frags = ranges[-p]
        if not frags:
            continue
        env = [min(r[0] for r in frags), max(r[1] for r in frags)]
        pieces = [list(env)]
        for c_lo, c_hi in claimed:
            nxt = []
            for lo, hi in pieces:
                if c_hi <= lo or c_lo >= hi:
                    nxt.append([lo, hi])
                    continue
                if c_lo > lo:
                    nxt.append([lo, c_lo])
                if c_hi < hi:
                    nxt.append([c_hi, hi])
            pieces = nxt
        ranges[-p] = pieces
        claimed.append(env)


def _clean_bidirectional_pathogenic(ranges: Ranges, benign_center: float) -> None:
    """Split pathogenic fragments at ``benign_center`` and post-process each side as a
    one-sided calibration (left: lower is pathogenic; right: higher is)."""
    sides = []
    for flipped in (False, True):
        side: Ranges = {q: [] for q in (*POINT_VALUES, *(-p for p in POINT_VALUES))}
        for p in POINT_VALUES:
            for lo, hi in ranges[p]:
                if ((lo + hi) / 2 < benign_center) != flipped:
                    side[p].append([lo, hi])
        _postprocess_one_sided(side, flipped)
        sides.append(side)
    for p in POINT_VALUES:
        ranges[p] = sides[0][p] + sides[1][p]


def postprocess(
    ranges: Ranges, flipped: bool, bidirectional: bool, benign_center: float | None
) -> Ranges:
    ranges = {k: [list(r) for r in v] for k, v in ranges.items()}
    if bidirectional:
        _clean_benign_no_extend(ranges)
        if benign_center is not None:
            _clean_bidirectional_pathogenic(ranges, benign_center)
    else:
        _postprocess_one_sided(ranges, flipped)
    return ranges


def _bidirectional_by_weights(
    mixture: _mixture.Mixture, source: np.ndarray, ref: np.ndarray
) -> bool:
    """A benign-like component (by weight) with pathogenic-like components on both sides."""
    means = _skewnorm.mean(*mixture.params.T)
    path_like = [source[k] > ref[k] for k in np.argsort(means, kind="stable")]
    return _sandwiched(path_like)


def _bidirectional_by_raw_points(ranges: Ranges) -> bool:
    """Pathogenic evidence on both sides of benign evidence along the score axis."""
    segments = sorted(
        [(lo, p > 0) for p, rs in ranges.items() for lo, _ in rs], key=lambda s: s[0]
    )
    return _sandwiched([is_p for _, is_p in segments])


def _sandwiched(path_like: Sequence[bool]) -> bool:
    seen_p = False
    for i, is_p in enumerate(path_like):
        if is_p:
            seen_p = True
        elif seen_p and any(path_like[i + 1 :]):
            return True
    return False


# ----- bootstrap ----------------------------------------------------------------------


@dataclasses.dataclass
class _Boot:
    fits: dict[int, tuple[_mixture.Mixture | None, float]]
    held_out: np.ndarray  # row indices (into the control rows) not drawn


def _one_hot(member: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One group per row, chosen uniformly among the groups the row belongs to."""
    return np.argmax(rng.random(member.shape) * member, axis=1)


def _split(
    sample: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Resample each group with replacement at its own size; held-out rows are those not
    drawn (a group of one is used whole, with nothing held out)."""
    train, held = [], []
    for s in range(len(GROUPS)):
        idx = np.flatnonzero(sample == s)
        if len(idx) == 0:
            continue
        if len(idx) == 1:
            train.append(idx)
            continue
        for _ in range(100):
            drawn = rng.choice(idx, size=len(idx), replace=True)
            rest = np.setdiff1d(idx, drawn)
            if len(rest):
                break
        else:
            raise RuntimeError("Could not draw a bootstrap sample with held-out rows")
        train.append(drawn)
        held.append(rest)
    return (
        np.concatenate(train),
        np.concatenate(held) if held else np.array([], dtype=int),
    )


def _bootstrap(
    seed: np.random.SeedSequence,
    x: np.ndarray,
    member: np.ndarray,
    ks: Sequence[int],
    n_restarts: int,
    center: float,
    spread: float,
    em_kw: dict[str, Any],
) -> _Boot:
    rng = np.random.default_rng(seed)
    sample = _one_hot(member, rng)
    train, held = _split(sample, rng)
    fits = {}
    for k in ks:
        mixture, val_ll = _mixture.fit_restarts(
            x[train],
            sample[train],
            len(GROUPS),
            k,
            rng,
            n_restarts=n_restarts,
            x_val=x[held],
            sample_val=sample[held],
            **em_kw,
        )
        fits[k] = (mixture.rescaled(center, spread) if mixture else None, val_ll)
    return _Boot(fits, held)


# ----- calibration --------------------------------------------------------------------


def _check_groups(member: np.ndarray, notes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Drop groups too small to fit; collect reasons the result is unreliable."""
    member = member.copy()
    unreliable = []
    for g, name in enumerate(GROUPS):
        n = int(member[:, g].sum())
        if 0 < n < MIN_GROUP_SIZE and g != GNOMAD:
            msg = f"Only {n} {name} variant(s) with a score (< {MIN_GROUP_SIZE}); group dropped"
            notes.append(msg)
            member[:, g] = False
        elif MIN_GROUP_SIZE <= n < RELIABLE_GROUP_SIZE and g != GNOMAD:
            unreliable.append(f"only {n} {name} variants (< {RELIABLE_GROUP_SIZE})")
    n_gnomad = int(member[:, GNOMAD].sum())
    if n_gnomad < MIN_GROUP_SIZE:
        raise ValueError(
            f"The gnomAD sample has {n_gnomad} variant(s) with a score; at least "
            f"{MIN_GROUP_SIZE} are needed to estimate the prior"
        )
    if not member[:, [PATHOGENIC, BENIGN, SYNONYMOUS]].any():
        raise ValueError(
            "No control group is large enough to calibrate: need at least "
            f"{MIN_GROUP_SIZE} pathogenic, benign or synonymous variants with a score"
        )
    return member, unreliable


def _resolve_mode(has: dict[int, bool]) -> str:
    if has[PATHOGENIC] and (has[BENIGN] or has[SYNONYMOUS]):
        return "standard"
    return "positive_unlabeled" if has[PATHOGENIC] else "negative_unlabeled"


def _resolve_benign_method(method: str, has: dict[int, bool], notes: list[str]) -> str:
    if method not in BENIGN_METHODS:
        raise ValueError(
            f"benign_method must be one of {BENIGN_METHODS}, got {method!r}"
        )
    if not has[BENIGN] and not has[SYNONYMOUS]:
        return method
    resolved = method
    if method in ("avg", "synonymous") and not has[SYNONYMOUS]:
        resolved = "benign"
    elif method in ("avg", "benign") and not has[BENIGN]:
        resolved = "synonymous"
    if resolved != method:
        notes.append(
            f"benign_method {method!r} needs missing controls; using {resolved!r}"
        )
    return resolved


def _reference_means(
    scores: np.ndarray, member: np.ndarray, mode: str, benign_method: str
) -> tuple[float, float]:
    """Mean pathogenic-side and benign-side scores (gnomAD stands in for a missing side)."""

    def mean(g: int) -> float:
        return float(scores[member[:, g]].mean())

    path = mean(PATHOGENIC) if member[:, PATHOGENIC].any() else mean(GNOMAD)
    has_b, has_s = member[:, BENIGN].any(), member[:, SYNONYMOUS].any()
    if not has_b and not has_s:
        return path, mean(GNOMAD)
    if benign_method == "avg" and has_b and has_s:
        return path, (mean(BENIGN) + mean(SYNONYMOUS)) / 2
    if benign_method == "synonymous" and has_s:
        return path, mean(SYNONYMOUS)
    return path, mean(BENIGN) if has_b else mean(SYNONYMOUS)


def _percentile_curves(log_lr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        return (
            np.nanpercentile(log_lr, PATHOGENIC_PERCENTILE, axis=0),
            np.nanmedian(log_lr, axis=0),
            np.nanpercentile(log_lr, BENIGN_PERCENTILE, axis=0),
        )


def _ranges_for(
    grid: np.ndarray,
    log_lr: np.ndarray,
    prior: float,
    flipped: bool,
    bidirectional: bool,
    benign_center: float | None,
) -> tuple[Ranges, int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    curves = _percentile_curves(log_lr)
    defined = ~np.isnan(log_lr).all(axis=0)
    raw, c = raw_point_ranges(
        grid[defined], curves[0][defined], curves[2][defined], prior
    )
    return postprocess(raw, flipped, bidirectional, benign_center), c, curves


def _oob_points(
    rows: np.ndarray,
    scores: np.ndarray,
    oob_fits: list[np.ndarray],
    priors: np.ndarray,
    log_lr: np.ndarray,
    grid: np.ndarray,
    flipped: bool,
    bidirectional: bool,
    benign_center: float | None,
) -> list[float]:
    out = []
    for row in rows:
        idx = oob_fits[row]
        if len(idx) < MIN_OOB_FITS:
            out.append(float("nan"))
            continue
        prior = float(np.median(priors[idx]))
        ranges, _, _ = _ranges_for(
            grid, log_lr[idx], prior, flipped, bidirectional, benign_center
        )
        out.append(float(assign_points(np.array([scores[row]]), ranges)[0]))
    return out


def fit_calibration(
    scores: np.ndarray,
    member: np.ndarray,
    *,
    direction: str = "auto",
    benign_method: str = "avg",
    n_components: int | str = "auto",
    n_bootstrap: int = 1000,
    n_restarts: int = 8,
    prior: float | None = None,
    strict_constraint: bool = False,
    out_of_bag: bool = True,
    seed: int = 0,
    n_jobs: int | None = -1,
    extra_warnings: Sequence[str] = (),
    em_kw: dict[str, Any] | None = None,
) -> tuple[Calibration, np.ndarray]:
    """Calibrate ``scores`` against control groups.

    Parameters
    ----------
    scores : array, shape (n,)
        Every scored variant. All of them set the score grid; only those in a group are
        fit.
    member : bool array, shape (n, 4)
        Membership in the pathogenic, benign, gnomAD and synonymous groups (`GROUPS`
        order). A variant may belong to several.

    Returns
    -------
    calibration : Calibration
    oob_points : array, shape (n,)
        Out-of-bag points for each control variant (NaN for variants in no group, with
        too few out-of-bag fits, or with ``out_of_bag=False``).
    """
    scores = np.asarray(scores, dtype=float)
    member = np.asarray(member, dtype=bool)
    if member.shape != (len(scores), len(GROUPS)):
        raise ValueError(f"member must have shape ({len(scores)}, {len(GROUPS)})")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if n_components not in ("auto", 2, 3):
        raise ValueError(f"n_components must be 'auto', 2 or 3, got {n_components!r}")
    if n_bootstrap < 1 or n_restarts < 1:
        raise ValueError("n_bootstrap and n_restarts must be at least 1")
    if prior is not None and not 0 < prior < 1:
        raise ValueError(f"prior must be in (0, 1), got {prior!r}")
    finite = np.isfinite(scores)
    member = member & finite[:, None]
    if not finite.any():
        raise ValueError("No finite scores to calibrate")

    notes = list(extra_warnings)
    member, unreliable = _check_groups(member, notes)
    has = {g: bool(member[:, g].any()) for g in range(len(GROUPS))}
    mode = _resolve_mode(has)
    if mode != "standard":
        notes.append(
            f"Only {'pathogenic' if has[PATHOGENIC] else 'benign'} controls: the other "
            f"class is recovered from gnomAD ({mode.replace('_', '-')})"
        )
    benign_method = _resolve_benign_method(benign_method, has, notes)

    controls = np.flatnonzero(member.any(axis=1))
    x = scores[controls]
    center, spread = float(x.mean()), float(x.std())
    if not spread > 0:
        raise ValueError("All control scores are identical; nothing to calibrate")
    lo, hi = float(scores[finite].min()), float(scores[finite].max())
    em_kw = {
        "score_range": ((lo - center) / spread, (hi - center) / spread),
        "strict": strict_constraint,
        **(em_kw or {}),
    }
    ks = (2, 3) if n_components == "auto" else (int(n_components),)
    seeds = np.random.SeedSequence(seed).spawn(n_bootstrap)
    boots: list[_Boot] = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap)(
            s,
            (x - center) / spread,
            member[controls],
            ks,
            n_restarts,
            center,
            spread,
            em_kw,
        )
        for s in seeds
    )

    # Model selection (Supp. 1.6).
    support = None
    k = ks[0]
    if len(ks) == 2:
        pairs = [
            (b.fits[3][1], b.fits[2][1])
            for b in boots
            if np.isfinite(b.fits[2][1]) and np.isfinite(b.fits[3][1])
        ]
        support = float(np.mean([l3 > l2 for l3, l2 in pairs])) if pairs else 0.0
        k = 3 if support >= THREE_COMPONENT_SUPPORT else 2

    fits = [(i, b.fits[k][0]) for i, b in enumerate(boots) if b.fits[k][0] is not None]
    population = scores[member[:, GNOMAD]]
    if prior is None:
        priors = np.array(
            [fit_prior(m, population, mode, benign_method) for _, m in fits]
        )
    else:
        priors = np.full(len(fits), float(prior))
    valid = np.isfinite(priors) & (priors > 0) & (priors < 1)
    fits = [f for f, ok in zip(fits, valid, strict=True) if ok]
    priors = priors[valid]
    if not fits:
        raise ValueError(
            "No bootstrap fit gave a prior probability of pathogenicity in (0.001, 0.999): "
            "relative to the controls, the gnomAD sample looks entirely benign-like or "
            "entirely pathogenic-like. Check the score direction and controls, or pass "
            "prior= to set the prior"
        )
    median_prior = float(np.median(priors))
    prior_unstable = (
        mode != "standard"
        and prior is None
        and bool(np.isclose(median_prior, _UNMIX_FLOOR))
    )
    if prior_unstable:
        notes.append("The one-class prior estimate sits on its 0.01 floor")

    grid = np.linspace(lo, hi, GRID_POINTS)
    log_lr = np.empty((len(fits), GRID_POINTS))
    for j, ((_, m), p) in enumerate(zip(fits, priors, strict=True)):
        log_fp, log_fb = fit_log_densities(m, grid, p, mode, benign_method)
        with np.errstate(invalid="ignore"):
            log_lr[j] = log_fp - log_fb

    path_mean, benign_mean = _reference_means(scores, member, mode, benign_method)
    if direction == "auto":
        flipped = path_mean > benign_mean
        votes = _bidirectional_votes(fits, priors, log_lr, grid, k, mode, benign_method)
        bidirectional = votes >= BIDIRECTIONAL_VOTE_THRESHOLD
    else:
        votes = None
        flipped = direction == "higher_pathogenic"
        bidirectional = direction == "both"
    benign_center = benign_mean if bidirectional else None
    ranges, c, curves = _ranges_for(
        grid, log_lr, median_prior, flipped, bidirectional, benign_center
    )

    oob = np.full(len(scores), np.nan)
    if out_of_bag:
        fit_index = {boot: j for j, (boot, _) in enumerate(fits)}
        oob_fits: list[list[int]] = [[] for _ in controls]
        for boot, b in enumerate(boots):
            if boot in fit_index:
                for row in b.held_out:
                    oob_fits[row].append(fit_index[boot])
        oob_arrays = [np.asarray(v, dtype=int) for v in oob_fits]
        chunks = np.array_split(
            np.arange(len(controls)), max(1, min(len(controls), 64))
        )
        results = Parallel(n_jobs=n_jobs)(
            delayed(_oob_points)(
                chunk,
                x,
                oob_arrays,
                priors,
                log_lr,
                grid,
                flipped,
                bidirectional,
                benign_center,
            )
            for chunk in chunks
            if len(chunk)
        )
        oob[controls] = np.concatenate([np.asarray(r, dtype=float) for r in results])

    for reason in unreliable:
        notes.append(f"Unreliable: {reason}")
    for msg in notes:
        logger.warning("ExCALIBR: %s", msg)
    calibration = Calibration(
        point_ranges={
            p: [(float(a), float(b)) for a, b in r] for p, r in ranges.items()
        },
        prior=median_prior,
        prior_unstable=prior_unstable,
        tavtigian_c=c,
        mode=mode,
        n_components=k,
        benign_method=benign_method,
        direction="both"
        if bidirectional
        else ("higher_pathogenic" if flipped else "lower_pathogenic"),
        grid=grid.tolist(),
        log_lr_low=curves[0].tolist(),
        log_lr_median=curves[1].tolist(),
        log_lr_high=curves[2].tolist(),
        group_counts={name: int(member[:, g].sum()) for g, name in enumerate(GROUPS)},
        n_bootstrap=n_bootstrap,
        n_valid_fits=len(fits),
        three_component_support=support,
        bidirectional_votes=votes,
        reliable=not unreliable and not prior_unstable,
        warnings=notes,
        settings={
            "direction": direction,
            "benign_method": benign_method,
            "n_components": n_components,
            "n_bootstrap": n_bootstrap,
            "n_restarts": n_restarts,
            "prior": prior,
            "strict_constraint": strict_constraint,
            "out_of_bag": out_of_bag,
            "seed": seed,
        },
        fits=[
            {"prior": float(p), **m.to_dict()}
            for (_, m), p in zip(fits, priors, strict=True)
        ],
    )
    return calibration, oob


def _bidirectional_votes(
    fits: list[tuple[int, _mixture.Mixture]],
    priors: np.ndarray,
    log_lr: np.ndarray,
    grid: np.ndarray,
    k: int,
    mode: str,
    benign_method: str,
) -> float:
    """Share of fits that look bidirectional: by component weights with 3 or more
    components, else by each fit's own raw point ranges."""
    flagged = 0
    for (_, m), p, lr in zip(fits, priors, log_lr, strict=True):
        if k >= 3:
            w = m.weights
            if mode == "negative_unlabeled":
                source, ref = w[GNOMAD], _benign_weights(w, benign_method)
            elif mode == "positive_unlabeled":
                source, ref = w[PATHOGENIC], w[GNOMAD]
            else:
                source, ref = w[PATHOGENIC], _benign_weights(w, benign_method)
            flagged += _bidirectional_by_weights(m, source, ref)
        else:
            defined = ~np.isnan(lr)
            raw, _ = raw_point_ranges(grid[defined], lr[defined], lr[defined], p)
            flagged += _bidirectional_by_raw_points(raw)
    return flagged / len(fits)
