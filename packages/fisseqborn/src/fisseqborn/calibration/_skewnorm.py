"""Skew-normal densities, parameterizations and the truncated-normal moments the EM uses.

Canonical parameters are ``(a, loc, scale)`` (skew, location, scale), as in
`scipy.stats.skewnorm`. The EM works in the alternate parameterization
``(loc, Delta, Gamma)`` of Lin et al. (2007): ``X = loc + Delta * T + sqrt(Gamma) * U``
with ``T`` a standard normal truncated below at 0 and ``U`` standard normal.

numpy and scipy only.
"""

import numpy as np
from scipy.special import log_ndtr

_LOG_2 = float(np.log(2.0))
_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


def logpdf(x: np.ndarray, a: float, loc: float, scale: float) -> np.ndarray:
    """Log density of one skew-normal at ``x`` (numpy broadcasting applies)."""
    z = (np.asarray(x, dtype=float) - loc) / scale
    return _LOG_2 - np.log(scale) - 0.5 * z * z - _HALF_LOG_2PI + log_ndtr(a * z)


def component_logpdf(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Log density of each component (``params`` rows ``(a, loc, scale)``) at ``x``,
    shape ``(K, len(x))``."""
    params = np.asarray(params, dtype=float)
    return logpdf(
        np.asarray(x, dtype=float)[None, :], *(params[:, i, None] for i in range(3))
    )


def delta(a: np.ndarray) -> np.ndarray:
    return np.asarray(a) / np.sqrt(1.0 + np.asarray(a) ** 2)


def mean(a: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.asarray(loc) + np.asarray(scale) * delta(a) * np.sqrt(2.0 / np.pi)


def to_alternate(a: float, loc: float, scale: float) -> tuple[float, float, float]:
    """``(a, loc, scale)`` -> ``(loc, Delta, Gamma)``."""
    big_delta = scale * float(delta(a))
    return float(loc), big_delta, float(scale**2 - big_delta**2)


def to_canonical(
    loc: float, big_delta: float, gamma: float
) -> tuple[float, float, float]:
    """``(loc, Delta, Gamma)`` -> ``(a, loc, scale)``."""
    return (
        float(np.sign(big_delta) * np.sqrt(big_delta**2 / gamma)),
        float(loc),
        float(np.sqrt(gamma + big_delta**2)),
    )


def truncnorm_moments(
    x: np.ndarray, a: float, loc: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """``E[T | x]`` and ``E[T^2 | x]`` for each observation under one component.

    ``T | x`` is normal with mean ``delta * (x - loc) / scale`` and variance
    ``1 - delta^2``, truncated below at 0.
    """
    d = float(delta(a))
    m = d * (x - loc) / scale
    sigma = np.sqrt(1.0 - d * d)
    alpha = m / sigma
    # phi(alpha) / Phi(alpha), in log space so it stays finite far in the left tail.
    ratio = np.exp(-0.5 * alpha * alpha - _HALF_LOG_2PI - log_ndtr(alpha))
    m1 = m + sigma * ratio
    m2 = m * m + sigma * sigma + sigma * m * ratio
    return m1, m2
