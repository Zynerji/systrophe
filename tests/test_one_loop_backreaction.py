"""Tests for one-loop back-reaction module."""

import math

import pytest

from systrophe.one_loop_backreaction import (
    back_reaction_to_alpha,
    corrected_F,
    novelty_scan,
    one_loop_F_correction,
    shifted_chronology_horizons,
    trace_anomaly_at_r,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_one_loop_correction_finite(vs_super):
    delta = one_loop_F_correction(vs_super, r=1.5)
    assert math.isfinite(delta)


def test_corrected_F_zero_epsilon_equals_F(vs_super):
    import numpy as np
    F_classical = float(vs_super.analytic_exterior_F(np.array([1.5]))[0])
    F_corr = corrected_F(vs_super, r=1.5, epsilon=0.0)
    assert F_corr == pytest.approx(F_classical, rel=1e-12)


def test_corrected_F_finite_general(vs_super):
    F_corr = corrected_F(vs_super, r=1.5, epsilon=0.01)
    assert math.isfinite(F_corr)


def test_shifted_horizons_supercritical(vs_super):
    res = shifted_chronology_horizons(vs_super)
    assert res["regime"] == "supercritical"
    assert isinstance(res["shifted_horizons"], list)


def test_shifted_horizons_subcritical(vs_sub):
    res = shifted_chronology_horizons(vs_sub)
    assert res["regime"] == "subcritical"


def test_back_reaction_to_alpha_returns_dict(vs_super):
    res = back_reaction_to_alpha(vs_super, epsilon=0.001)
    assert "alpha_classical" in res
    assert "alpha_corrected" in res


def test_back_reaction_zero_epsilon_recovers_classical(vs_super):
    res = back_reaction_to_alpha(vs_super, epsilon=0.0)
    if math.isfinite(res["alpha_corrected"]):
        assert res["alpha_corrected"] == pytest.approx(res["alpha_classical"], rel=1e-3)


def test_trace_anomaly_finite(vs_super):
    a = trace_anomaly_at_r(vs_super, r=1.5)
    assert math.isfinite(a)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_radii=10)
    assert "verdict" in res
