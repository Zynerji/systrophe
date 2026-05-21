"""Tests for the growth catcher (sustained exponential-trend detection)."""

import numpy as np

from systrophe.growth_catcher import catch_growth, summarize_growth_for_report


def _u(n=60, lo=1.0, hi=10.0):
    return np.linspace(lo, hi, n)


def test_constant_is_stationary():
    u = _u()
    res = catch_growth(u, np.full_like(u, 3.0))
    assert res.verdict == "stationary"
    assert abs(res.growth_exponent) < 1e-6


def test_pure_growth_recovers_exponent():
    u = _u()
    sigma = 0.20
    res = catch_growth(u, np.exp(sigma * u))
    assert res.verdict == "growing"
    assert abs(res.growth_exponent - sigma) < 1e-6
    assert res.r_squared > 0.999
    assert res.endpoint_drift > 0.9
    assert res.monotone_fraction > 0.99


def test_pure_decay_is_decaying():
    u = _u()
    sigma = -0.15
    res = catch_growth(u, 5.0 * np.exp(sigma * u))
    assert res.verdict == "decaying"
    assert abs(res.growth_exponent - sigma) < 1e-6
    assert res.monotone_fraction < 0.01


def test_noisy_flat_no_false_alarm():
    u = _u(n=80)
    rng = np.random.default_rng(42)
    # Stationary AR-like noise around a constant mean.
    vals = 2.0 + 0.05 * rng.standard_normal(len(u))
    res = catch_growth(u, np.abs(vals), seed=1)
    assert res.verdict == "stationary"
    assert res.p_value > 0.01


def test_growth_with_noise_still_detected():
    u = _u(n=80)
    rng = np.random.default_rng(7)
    sigma = 0.15
    vals = np.exp(sigma * u) * (1.0 + 0.05 * rng.standard_normal(len(u)))
    res = catch_growth(u, vals, seed=2)
    assert res.verdict == "growing"
    assert abs(res.growth_exponent - sigma) < 0.03


def test_white_noise_monotone_fraction_near_half():
    u = _u(n=200)
    rng = np.random.default_rng(11)
    res = catch_growth(u, np.abs(rng.standard_normal(len(u))) + 1.0, seed=3)
    assert 0.35 < res.monotone_fraction < 0.65
    assert res.verdict == "stationary"


def test_too_few_points_is_stationary():
    res = catch_growth(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert res.verdict == "stationary"
    assert res.n_points == 3


def test_summary_string_mentions_verdict():
    u = _u()
    res = catch_growth(u, np.exp(0.1 * u))
    s = summarize_growth_for_report(res)
    assert "growing" in s
    assert "sigma_hat" in s
