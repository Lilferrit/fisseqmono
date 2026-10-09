import json
import logging

import numpy as np
import pytest
from fisseqborn_calibration_data import groups, true_log_lr
from scipy.stats import skewnorm, truncnorm

from fisseqborn.calibration import Calibration, _core, _mixture, _skewnorm
from fisseqborn.calibration._result import assign_points

# Small and loose for speed; test_synthetic_recovery uses the default EM tolerance.
FAST = {
    "n_bootstrap": 8,
    "n_restarts": 2,
    "n_components": 2,
    "n_jobs": 1,
    "em_kw": {"tol": 0.05},
}


@pytest.fixture(scope="module")
def fitted() -> Calibration:
    scores, member = groups(0)
    return _core.fit_calibration(scores, member, **FAST)[0]


# ----- skew normal -------------------------------------------------------------------


def test_logpdf_matches_scipy():
    x = np.linspace(-4, 4, 50)
    for a, loc, scale in [(0.0, 0.0, 1.0), (3.0, -1.0, 0.5), (-2.0, 2.0, 2.0)]:
        np.testing.assert_allclose(
            _skewnorm.logpdf(x, a, loc, scale), skewnorm.logpdf(x, a, loc, scale)
        )


def test_parameterizations_round_trip():
    for params in [(2.0, 0.5, 1.5), (-0.7, -3.0, 0.2), (0.0, 1.0, 1.0)]:
        back = _skewnorm.to_canonical(*_skewnorm.to_alternate(*params))
        np.testing.assert_allclose(back, params, atol=1e-12)


def test_truncnorm_moments_match_scipy():
    x = np.array([-3.0, -0.5, 0.0, 1.2, 4.0])
    a, loc, scale = 1.5, 0.3, 1.1
    v, w = _skewnorm.truncnorm_moments(x, a, loc, scale)
    d = a / np.sqrt(1 + a * a)
    mu, sigma = d * (x - loc) / scale, np.sqrt(1 - d * d)
    ref = truncnorm(-mu / sigma, np.inf, loc=mu, scale=sigma)
    np.testing.assert_allclose(v, ref.mean(), rtol=1e-8)
    np.testing.assert_allclose(w, ref.moment(2), rtol=1e-8)


# ----- mixture -----------------------------------------------------------------------


def test_mixture_recovers_weights_and_keeps_constraint():
    scores, member = groups(1, n_unlabeled=0)
    sample = member.argmax(axis=1)
    x = (scores - scores.mean()) / scores.std()
    rng = np.random.default_rng(0)
    fit, _ = _mixture.fit_restarts(x, sample, 4, 2, rng, n_restarts=3)
    assert fit is not None
    abnormal = 0  # components are ordered by location; abnormal is the low one
    assert fit.weights[0, abnormal] == pytest.approx(0.9, abs=0.08)
    assert fit.weights[1, abnormal] == pytest.approx(0.05, abs=0.08)
    assert np.isnan(fit.weights[3]).all()
    constraint = _mixture._Constraint(x.min(), x.max(), strict=False)
    grid_lp = np.array([constraint.logpdf(*p) for p in fit.params])
    assert constraint.all_ok(grid_lp)


# ----- prior and points --------------------------------------------------------------


def test_tavtigian_constant():
    assert _core.tavtigian_constant(0.0441) == 1124  # Pejaver et al. 2022
    assert _core.tavtigian_constant(0.1) == 348  # ~350, the Tavtigian et al. value


def test_label_shift_prior_recovers_mixing_proportion():
    rng = np.random.default_rng(0)
    pathogenic, benign = rng.random(5000) < 0.25, None
    x = np.where(pathogenic, rng.normal(-2, 1, 5000), rng.normal(1, 1, 5000))
    f_p = np.exp(-0.5 * (x + 2) ** 2)
    f_b = np.exp(-0.5 * (x - 1) ** 2)
    assert benign is None
    assert _core.label_shift_prior(f_p, f_b) == pytest.approx(0.25, abs=0.03)


def test_unmixing_prior_bounds():
    f_class = np.array([1.0, 2.0, 4.0])
    assert (
        _core.unmixing_prior(f_class, np.array([0.3, 2.0, 4.0]), "positive_unlabeled")
        == 0.3
    )
    assert (
        _core.unmixing_prior(f_class, np.array([0.3, 2.0, 4.0]), "negative_unlabeled")
        == 0.7
    )
    assert _core.unmixing_prior(f_class, f_class * 0.001, "positive_unlabeled") == 0.01
    assert np.isnan(_core.unmixing_prior(f_class, f_class * 2, "positive_unlabeled"))


def test_raw_point_ranges_and_postprocess_lower_pathogenic():
    grid = np.linspace(-3, 3, 601)
    log_lr = -2.0 * grid  # strongly pathogenic at low scores, benign at high
    raw, c = _core.raw_point_ranges(grid, log_lr, log_lr, 0.1)
    assert c == 348
    tau = np.log(c) * np.arange(1, 9) / 8
    # +1 starts where log LR first reaches tau[0], scanning from the low end.
    assert raw[1][0][1] == pytest.approx(-tau[0] / 2, abs=0.02)
    ranges = _core.postprocess(
        raw, flipped=False, bidirectional=False, benign_center=None
    )
    assert ranges[8][0][0] == -np.inf and ranges[-8][0][1] == np.inf
    points = assign_points(np.array([-3.0, 0.0, 3.0]), ranges)
    assert points.tolist() == [8, 0, -8]


def test_postprocess_keeps_one_range_per_point():
    # Liberal monotonicity (upstream's default): fragmented points keep one range each and
    # the strongest point on each side extends to the axis end. It does not fill gaps
    # between points, which only noisy curves (not mixture fits) produce.
    grid = np.linspace(-3, 3, 601)
    rng = np.random.default_rng(0)
    log_lr = -2.0 * grid + rng.normal(0, 0.3, grid.size)
    raw, _ = _core.raw_point_ranges(grid, log_lr, log_lr, 0.2)
    assert len(raw[1]) > 1
    ranges = _core.postprocess(
        raw, flipped=False, bidirectional=False, benign_center=None
    )
    assert all(len(r) <= 1 for r in ranges.values())
    strongest = max(p for p in ranges if p > 0 and ranges[p])
    weakest_benign = min(p for p in ranges if p < 0 and ranges[p])
    assert ranges[strongest][0][0] == -np.inf
    assert ranges[weakest_benign][0][1] == np.inf


def test_postprocess_flipped_mirrors():
    grid = np.linspace(-3, 3, 601)
    raw, _ = _core.raw_point_ranges(grid, 2.0 * grid, 2.0 * grid, 0.1)
    ranges = _core.postprocess(
        raw, flipped=True, bidirectional=False, benign_center=None
    )
    points = assign_points(grid, ranges)
    assert (np.diff(points) >= 0).all()
    assert points[0] == -8 and points[-1] == 8


# ----- calibration -------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_synthetic_recovery(seed):
    scores, member = groups(seed, prior=0.3, n_gnomad=600)
    kw = {**FAST, "n_bootstrap": 12, "em_kw": None}
    cal, oob = _core.fit_calibration(scores, member, seed=seed, **kw)
    assert cal.mode == "standard"
    assert cal.direction == "lower_pathogenic"
    assert cal.prior == pytest.approx(0.3, abs=0.08)
    # The conservative supporting thresholds bracket the true ones.
    grid = np.asarray(cal.grid)
    true = true_log_lr(grid)
    tau = np.log(cal.tavtigian_c) / 8
    true_path = grid[true >= tau].max()
    true_benign = grid[true <= -tau].min()
    thresholds = cal.thresholds()
    assert thresholds[1] == pytest.approx(true_path, abs=0.5)
    assert thresholds[-1] == pytest.approx(true_benign, abs=0.5)
    assert thresholds[1] < thresholds[-1]
    # Control variants get out-of-bag points (when held out often enough).
    assert np.isnan(oob[~member.any(axis=1)]).all()


def test_higher_pathogenic_is_detected():
    scores, member = groups(0)
    cal, _ = _core.fit_calibration(-scores, member, **FAST)
    assert cal.direction == "higher_pathogenic"
    assert cal.points(np.array([3.0]))[0] > 0
    assert cal.points(np.array([-3.0]))[0] < 0


def test_direction_override():
    scores, member = groups(0)
    cal, _ = _core.fit_calibration(scores, member, direction="lower_pathogenic", **FAST)
    assert cal.direction == "lower_pathogenic"
    assert cal.bidirectional_votes is None


def test_determinism_across_n_jobs():
    scores, member = groups(3)
    kw = {**FAST, "n_bootstrap": 4}
    a, oob_a = _core.fit_calibration(scores, member, seed=5, **{**kw, "n_jobs": 1})
    b, oob_b = _core.fit_calibration(scores, member, seed=5, **{**kw, "n_jobs": 2})
    assert a.to_json() == b.to_json()
    np.testing.assert_array_equal(oob_a, oob_b)
    c, _ = _core.fit_calibration(scores, member, seed=6, **{**kw, "n_jobs": 1})
    assert c.prior != a.prior


def test_auto_components_records_support():
    scores, member = groups(0, n_gnomad=300)
    kw = {**FAST, "n_components": "auto", "n_bootstrap": 3}
    cal, _ = _core.fit_calibration(scores, member, **kw)
    assert cal.n_components in (2, 3)
    assert 0.0 <= cal.three_component_support <= 1.0


def test_manual_prior():
    scores, member = groups(0)
    cal, _ = _core.fit_calibration(scores, member, prior=0.1, **FAST)
    assert cal.prior == 0.1 and cal.tavtigian_c == 348


# ----- one control class, small and degenerate inputs --------------------------------


def test_positive_unlabeled(caplog):
    scores, member = groups(0, n_benign=0)
    with caplog.at_level(logging.WARNING):
        cal, _ = _core.fit_calibration(scores, member, **FAST)
    assert cal.mode == "positive_unlabeled"
    assert any("recovered from gnomAD" in w for w in cal.warnings)
    assert "positive-unlabeled" in caplog.text
    assert 0.01 <= cal.prior < 0.99


def test_negative_unlabeled():
    scores, member = groups(0, n_pathogenic=0)
    cal, _ = _core.fit_calibration(scores, member, **FAST)
    assert cal.mode == "negative_unlabeled"
    assert 0 < cal.prior < 1


def test_tiny_group_is_dropped_and_small_group_flagged():
    scores, member = groups(0, n_pathogenic=8, n_benign=3)
    cal, _ = _core.fit_calibration(scores, member, **FAST)
    assert cal.group_counts["benign"] == 0
    assert cal.mode == "positive_unlabeled"
    assert not cal.reliable
    assert any("Only 3 benign" in w for w in cal.warnings)
    assert any("only 8 pathogenic" in w for w in cal.warnings)


def test_too_few_gnomad_raises():
    scores, member = groups(0, n_gnomad=3)
    with pytest.raises(ValueError, match="gnomAD sample has 3"):
        _core.fit_calibration(scores, member, **FAST)


def test_no_controls_raises():
    scores, member = groups(0, n_pathogenic=0, n_benign=2)
    with pytest.raises(ValueError, match="No control group"):
        _core.fit_calibration(scores, member, **FAST)


def test_constant_scores_raise():
    _, member = groups(0)
    with pytest.raises(ValueError, match="identical"):
        _core.fit_calibration(np.ones(len(member)), member, **FAST)


def test_bounded_and_tied_scores():
    scores, member = groups(0)
    bounded = np.round(1 / (1 + np.exp(scores)), 2)  # in [0, 1], many ties
    cal, _ = _core.fit_calibration(bounded, member, **FAST)
    assert cal.direction == "higher_pathogenic"
    assert 0 < cal.prior < 1


def test_no_overlap_between_groups():
    scores, member = groups(0)
    scores = np.where(member[:, 0], scores - 10, scores)  # pathogenic far from the rest
    # No gnomAD variant looks pathogenic, so the prior estimate collapses to 0.
    with pytest.raises(ValueError, match="pass prior="):
        _core.fit_calibration(scores, member, **FAST)
    cal, _ = _core.fit_calibration(scores, member, prior=0.1, **FAST)
    assert cal.points(np.array([scores.min()]))[0] > 0
    assert cal.points(np.array([1.0]))[0] < 0


def test_all_gnomad_benign_like():
    scores, member = groups(0, prior=0.0)
    cal, _ = _core.fit_calibration(scores, member, **FAST)
    assert cal.prior < 0.1


def test_bimodal_gnomad_both_directions():
    rng = np.random.default_rng(0)
    pathogenic = np.concatenate([rng.normal(-3, 0.5, 60), rng.normal(3, 0.5, 60)])
    benign = rng.normal(0, 0.6, 120)
    gnomad = np.concatenate(
        [rng.normal(-3, 0.5, 60), rng.normal(0, 0.6, 400), rng.normal(3, 0.5, 60)]
    )
    scores = np.concatenate([pathogenic, benign, gnomad])
    member = np.zeros((len(scores), 4), dtype=bool)
    member[:120, 0], member[120:240, 1], member[240:, 2] = True, True, True
    cal, _ = _core.fit_calibration(
        scores,
        member,
        direction="both",
        **{**FAST, "n_components": 3, "n_bootstrap": 4},
    )
    assert cal.direction == "both"
    assert cal.points(np.array([-3.5]))[0] > 0
    assert cal.points(np.array([3.5]))[0] > 0
    assert cal.points(np.array([0.0]))[0] <= 0


def test_bad_arguments():
    scores, member = groups(0)
    with pytest.raises(ValueError, match="direction"):
        _core.fit_calibration(scores, member, direction="up")
    with pytest.raises(ValueError, match="benign_method"):
        _core.fit_calibration(scores, member, benign_method="mean", **FAST)
    with pytest.raises(ValueError, match="prior"):
        _core.fit_calibration(scores, member, prior=1.5)


# ----- result ------------------------------------------------------------------------


def test_calibration_json_round_trip(tmp_path, fitted):
    cal = fitted
    path = tmp_path / "cal.json"
    text = cal.to_json(path)
    assert "Infinity" in text
    back = Calibration.from_json(path)
    assert back == Calibration.from_json(text)
    assert back.point_ranges == cal.point_ranges
    x = np.linspace(-4, 4, 41)
    np.testing.assert_array_equal(back.points(x), cal.points(x))
    np.testing.assert_allclose(back.posterior(x), cal.posterior(x))
    assert json.loads(text)["format_version"] == 1


def test_calibration_scoring_helpers(fitted):
    cal = fitted
    x = np.array([-3.0, np.nan, 3.0])
    points = cal.points(x)
    assert points[1] == 0 and points[0] > 0 > points[2]
    post = cal.posterior(x)
    assert post[0] > cal.prior > post[2] and np.isnan(post[1])
