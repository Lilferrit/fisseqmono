"""Skew-normal mixtures with components shared across samples, fit by constrained EM.

Every sample (pathogenic, benign, gnomAD, synonymous) is a mixture of the same ``K``
skew-normal components with its own weights (Zeiberg et al., Supp. 1.2). EM follows
Lin et al. (2007) / Peng et al. (2020) in the ``(loc, Delta, Gamma)`` parameterization.
Each parameter update is held to the density constraint (Supp. 1.3.2): with components
ordered by location, ``log SN_k(s) - log SN_{k+1}(s)`` is non-increasing in ``s`` on a
grid over the score range, checked only where both log densities exceed -7 unless
``strict``. A proposed update that would break it is pulled back toward the previous
value by binary search, as upstream ExCALIBR does.

Two deliberate differences from the upstream code: the log-likelihood uses log weights
(upstream adds raw weights), and the ``Delta`` update is Lin et al.'s exact M-step
(``sum p v (x - loc) / sum p w``; upstream divides by ``sum p``). Both agree at
convergence; the first changes which restart wins.

numpy, scipy and scikit-learn only. Scores are expected standardized (mean 0, sd 1), so
the absolute tolerances mean the same thing for every assay.
"""

import dataclasses

import numpy as np
from scipy.special import logsumexp
from scipy.stats import skew as sample_skew
from sklearn.cluster import KMeans

from . import _skewnorm

#: Log densities at or below this are "negligible" for the relaxed density constraint.
NEGLIGIBLE_LOG_DENSITY = -7.0
INIT_METHODS = ("kmeans", "method_of_moments")
_MIN_GAMMA = 1e-8


class FitError(RuntimeError):
    """A single EM run could not be initialized or diverged."""


@dataclasses.dataclass(frozen=True)
class Mixture:
    """Fitted components and per-sample weights.

    ``params`` has one row ``(a, loc, scale)`` per component, ordered by location.
    ``weights`` has one row per sample (NaN for a sample with no observations).
    """

    params: np.ndarray
    weights: np.ndarray

    @property
    def k(self) -> int:
        return len(self.params)

    def component_logpdf(self, x: np.ndarray) -> np.ndarray:
        return _skewnorm.component_logpdf(x, self.params)

    def logpdf(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Log density at ``x`` of the mixture with the given component ``weights``."""
        with np.errstate(divide="ignore"):
            log_w = np.log(np.asarray(weights, dtype=float))
        return logsumexp(self.component_logpdf(x) + log_w[:, None], axis=0)

    def rescaled(self, center: float, spread: float) -> "Mixture":
        """The same mixture for scores ``center + spread * x`` (undo standardization)."""
        params = self.params.copy()
        params[:, 1] = center + spread * params[:, 1]
        params[:, 2] = spread * params[:, 2]
        return Mixture(params, self.weights.copy())

    def to_dict(self) -> dict:
        return {"params": self.params.tolist(), "weights": self.weights.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Mixture":
        return cls(
            np.asarray(d["params"], dtype=float), np.asarray(d["weights"], dtype=float)
        )


def _joint(lp: np.ndarray, sample: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``log w_{sample(i), k} + log SN_k(x_i)``, shape ``(n, K)``, from the component log
    densities ``lp`` (shape ``(K, n)``)."""
    with np.errstate(divide="ignore"):
        return lp.T + np.log(weights)[sample]


def _lse_rows(joint: np.ndarray) -> np.ndarray:
    top = joint.max(axis=1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    return (top + np.log(np.exp(joint - top).sum(axis=1, keepdims=True)))[:, 0]


def _posterior(joint: np.ndarray, lse: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.exp(joint - lse[:, None]))


def _responsibilities(
    lp: np.ndarray, sample: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    joint = _joint(lp, sample, weights)
    return _posterior(joint, _lse_rows(joint))


def log_likelihood(
    x: np.ndarray, sample: np.ndarray, params: np.ndarray, weights: np.ndarray
) -> float:
    """Total log-likelihood of observations ``x`` drawn from samples ``sample``."""
    lp = _skewnorm.component_logpdf(x, params)
    return float(_lse_rows(_joint(lp, sample, weights)).sum())


class _Constraint:
    """The pairwise density-ratio constraint on a fixed score grid."""

    def __init__(self, lo: float, hi: float, strict: bool, points: int = 1000) -> None:
        self.grid = np.linspace(lo, hi, points)
        self.strict = strict

    def logpdf(self, a: float, loc: float, scale: float) -> np.ndarray:
        return _skewnorm.logpdf(self.grid, a, loc, scale)

    def violated(self, lp_left: np.ndarray, lp_right: np.ndarray) -> bool:
        """Whether ``log left - log right`` increases anywhere it is checked."""
        if self.strict:
            ratio = lp_left - lp_right
        else:
            keep = (lp_left > NEGLIGIBLE_LOG_DENSITY) & (
                lp_right > NEGLIGIBLE_LOG_DENSITY
            )
            ratio = (lp_left - lp_right)[keep]
        if ratio.size < 2:
            return False
        return bool(np.any(np.diff(ratio) > 1e-10))

    def all_ok(self, grid_lp: np.ndarray) -> bool:
        return not any(
            self.violated(grid_lp[i], grid_lp[i + 1]) for i in range(len(grid_lp) - 1)
        )


class _EM:
    """One EM run from one initialization."""

    def __init__(
        self,
        x: np.ndarray,
        sample: np.ndarray,
        n_samples: int,
        params: np.ndarray,
        weights: np.ndarray,
        constraint: _Constraint | None,
        tol_bsearch: float,
    ) -> None:
        self.x, self.sample, self.n_samples = x, sample, n_samples
        self.params, self.weights = params, weights
        self.constraint = constraint
        self.tol_bsearch = tol_bsearch
        if constraint is not None:
            self.grid_lp = np.array([constraint.logpdf(*p) for p in params])
        # Component log densities at x for the current params, and the joint log
        # densities for the current params and weights, reused across a step.
        self.lp = _skewnorm.component_logpdf(x, params)
        self._set_joint()

    def _set_joint(self) -> None:
        self.joint = _joint(self.lp, self.sample, self.weights)
        self.lse = _lse_rows(self.joint)

    def _neighbors_ok(self, k: int, grid_lp_k: np.ndarray) -> bool:
        c = self.constraint
        assert c is not None
        if k > 0 and c.violated(self.grid_lp[k - 1], grid_lp_k):
            return False
        if k < len(self.params) - 1 and c.violated(grid_lp_k, self.grid_lp[k + 1]):
            return False
        return True

    def _trial(self, k: int, alt: list[float]) -> tuple[bool, np.ndarray | None]:
        if not np.isfinite(alt).all() or alt[2] <= 0:
            return False, None
        canon = _skewnorm.to_canonical(*alt)
        if not np.isfinite(canon).all():
            return False, None
        if self.constraint is None:
            return True, None
        lp = self.constraint.logpdf(*canon)
        return self._neighbors_ok(k, lp), lp

    def _update(
        self, k: int, alt: list[float], idx: int, candidate: float
    ) -> list[float]:
        """Set ``alt[idx]`` to ``candidate``, or as close to it as the constraint allows."""
        trial = list(alt)
        trial[idx] = candidate
        ok, lp = self._trial(k, trial)
        if ok:
            if lp is not None:
                self.grid_lp[k] = lp
            return trial
        lo, hi = alt[idx], candidate
        best = None
        while abs(hi - lo) > self.tol_bsearch:
            mid = 0.5 * (lo + hi)
            trial[idx] = mid
            ok, lp = self._trial(k, trial)
            if ok:
                lo, best = mid, lp
            else:
                hi = mid
        trial[idx] = lo
        if best is not None:
            self.grid_lp[k] = best
        return trial

    def step(self) -> float:
        """One EM iteration; returns the new log-likelihood."""
        x = self.x
        post = _posterior(self.joint, self.lse)
        for k in range(len(self.params)):
            p = post[:, k]
            total = p.sum()
            if total <= 1e-12:
                continue
            a, loc, scale = self.params[k]
            alt = list(_skewnorm.to_alternate(a, loc, scale))
            v, w = _skewnorm.truncnorm_moments(x, a, loc, scale)
            alt = self._update(k, alt, 0, float((p * (x - v * alt[1])).sum() / total))
            new_loc = alt[0]
            # Delta's update uses the previous location, as upstream does.
            alt = self._update(
                k, alt, 1, float((p * v * (x - loc)).sum() / (p * w).sum())
            )
            new_delta = alt[1]
            resid = x - new_loc
            gamma = (
                p * (resid**2 - 2 * new_delta * v * resid + new_delta**2 * w)
            ).sum() / total
            alt = self._update(k, alt, 2, float(max(gamma, _MIN_GAMMA)))
            self.params[k] = _skewnorm.to_canonical(*alt)
        self.lp = _skewnorm.component_logpdf(x, self.params)
        post = _responsibilities(self.lp, self.sample, self.weights)
        for s in range(self.n_samples):
            rows = self.sample == s
            if rows.any():
                self.weights[s] = post[rows].mean(axis=0)
        self._set_joint()
        return self.log_likelihood()

    def log_likelihood(self) -> float:
        return float(self.lse.sum())


def _mom_params(values: np.ndarray) -> tuple[float, float, float]:
    """Method-of-moments skew-normal fit, with the skew clipped to a stable range."""
    m, sd = float(values.mean()), float(values.std())
    g = float(sample_skew(values)) if len(values) > 2 and sd > 0 else 0.0
    if not np.isfinite(g):
        g = 0.0
    b = min(abs(g), 0.99) ** (2.0 / 3.0)
    d2 = (np.pi / 2.0) * b / (b + ((4.0 - np.pi) / 2.0) ** (2.0 / 3.0))
    d = float(np.sign(g) * min(np.sqrt(d2), 0.95))
    omega = sd / np.sqrt(1.0 - 2.0 * d * d / np.pi)
    a = d / np.sqrt(1.0 - d * d)
    return a, m - omega * d * np.sqrt(2.0 / np.pi), omega


def _initialize(
    x: np.ndarray,
    sample: np.ndarray,
    n_samples: int,
    k: int,
    rng: np.random.Generator,
    method: str,
    constraint: _Constraint | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial components and weights (Supp. 1.3.3), shrunk until the constraint holds."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    for _ in range(100):
        if method == "kmeans":
            resample = x[rng.integers(0, n, n)][:, None]
            km = KMeans(
                n_clusters=k,
                init="random",
                n_init=1,
                random_state=int(rng.integers(2**31 - 1)),
            ).fit(resample)
            centers = np.sort(km.cluster_centers_.ravel())
            labels = np.abs(x[:, None] - centers[None, :]).argmin(axis=1)
        else:
            # Random partition into K score-contiguous groups.
            cuts = np.sort(rng.choice(np.arange(2, n - 1), size=k - 1, replace=False))
            labels = np.empty(n, dtype=int)
            labels[order] = np.searchsorted(cuts, np.arange(n), side="right")
        counts = np.bincount(labels, minlength=k)
        if (counts < 2).any():
            continue
        params = np.empty((k, 3))
        for j in range(k):
            values = x[labels == j]
            if method == "kmeans":
                a = rng.choice([-1, 0, 1]) * rng.uniform(0.0, 1.0)
                params[j] = (a, values.mean(), values.std())
            else:
                params[j] = _mom_params(values)
        if not np.isfinite(params).all() or (params[:, 2] <= 0).any():
            continue
        params = params[np.argsort(params[:, 1], kind="stable")]
        if constraint is not None:
            for _ in range(100):
                if constraint.all_ok(np.array([constraint.logpdf(*p) for p in params])):
                    break
                params[:, 0] *= 0.9
                params[:, 2] *= 0.9
            else:
                continue
        uniform = np.full((n_samples, k), 1.0 / k)
        post = _responsibilities(_skewnorm.component_logpdf(x, params), sample, uniform)
        weights = np.full((n_samples, k), np.nan)
        for s in range(n_samples):
            rows = sample == s
            if rows.any():
                weights[s] = post[rows].mean(axis=0)
        return params, weights
    raise FitError(f"Could not initialize a {k}-component mixture ({method})")


def fit_em(
    x: np.ndarray,
    sample: np.ndarray,
    n_samples: int,
    k: int,
    rng: np.random.Generator,
    *,
    method: str = "kmeans",
    score_range: tuple[float, float] | None = None,
    constrained: bool = True,
    strict: bool = False,
    max_iter: int = 10_000,
    tol: float = 1e-3,
    patience: int = 25,
    tol_bsearch: float = 1e-3,
) -> Mixture:
    """Fit one mixture from one initialization.

    ``sample[i]`` is the sample index (``0 .. n_samples - 1``) observation ``x[i]`` belongs
    to. EM stops when the log-likelihood has not improved by more than ``tol`` for
    ``patience`` iterations, or after ``max_iter``.
    """
    lo, hi = score_range if score_range is not None else (x.min(), x.max())
    constraint = _Constraint(lo, hi, strict) if constrained else None
    params, weights = _initialize(x, sample, n_samples, k, rng, method, constraint)
    em = _EM(x, sample, n_samples, params, weights, constraint, tol_bsearch)
    best = em.log_likelihood()
    stale = 0
    for _ in range(max_iter):
        ll = em.step()
        if not np.isfinite(ll):
            raise FitError("Log-likelihood is not finite")
        if ll - best > tol:
            best, stale = ll, 0
        else:
            stale += 1
            if stale > patience:
                break
    order = np.argsort(em.params[:, 1], kind="stable")
    return Mixture(em.params[order].copy(), em.weights[:, order].copy())


def fit_restarts(
    x: np.ndarray,
    sample: np.ndarray,
    n_samples: int,
    k: int,
    rng: np.random.Generator,
    *,
    n_restarts: int,
    x_val: np.ndarray | None = None,
    sample_val: np.ndarray | None = None,
    **em_kw,
) -> tuple[Mixture | None, float]:
    """Best of ``n_restarts`` EM runs, each initialized at random by k-means or method of
    moments, ranked by mean held-out log-likelihood (training log-likelihood when there
    is no held-out set). Returns ``(None, -inf)`` if every run fails."""
    best, best_ll = None, -np.inf
    for _ in range(n_restarts):
        method = INIT_METHODS[int(rng.integers(len(INIT_METHODS)))]
        try:
            fit = fit_em(x, sample, n_samples, k, rng, method=method, **em_kw)
        except (FitError, FloatingPointError, ValueError):
            continue
        if x_val is not None and sample_val is not None and len(x_val):
            ll = log_likelihood(x_val, sample_val, fit.params, fit.weights) / len(x_val)
        else:
            ll = log_likelihood(x, sample, fit.params, fit.weights) / len(x)
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = fit, ll
    return best, best_ll
