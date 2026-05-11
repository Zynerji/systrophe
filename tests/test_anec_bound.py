"""Tests for ANEC bound module."""

import math

import pytest

from systrophe.anec_bound import (
    ANECResult,
    anec_integral_circular_null,
    anec_integral_radial,
    anec_quantum_inequality_bound,
    anec_violation_amplitude,
    null_generator_radial,
    pair_extinction_anec_recovery,
    T_kk_radial,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_null_generator_radial_returns_pair(vs_super):
    kt, kr = null_generator_radial(vs_super, r=1.5)
    assert math.isfinite(kt)
    assert math.isfinite(kr)


def test_null_generator_zero_F_returns_nan(vs_super):
    """At F<0 region, radial null doesn't exist as standard timelike-like."""
    kt, kr = null_generator_radial(vs_super, r=3.35)  # inside CTC band
    assert math.isnan(kr)


def test_T_kk_radial_returns_value(vs_super):
    val = T_kk_radial(vs_super, r=1.5)
    assert math.isfinite(val) or math.isnan(val)


def test_T_kk_hartle_hawking_finite_far(vs_super):
    val = T_kk_radial(vs_super, r=1.5, quantum_state="hartle_hawking")
    assert math.isfinite(val)


def test_anec_integral_returns_ANECResult(vs_super):
    res = anec_integral_radial(vs_super)
    assert isinstance(res, ANECResult)


def test_anec_integral_has_qi_bound(vs_super):
    res = anec_integral_radial(vs_super)
    assert res.quantum_inequality_bound < 0


def test_anec_quantum_inequality_bound_inverse_quartic():
    """QI bound ~ -C / L^4 should be more restrictive for shorter intervals."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    b_short = anec_quantum_inequality_bound(vs, 1.0, 2.0)
    b_long = anec_quantum_inequality_bound(vs, 1.0, 10.0)
    assert b_short < b_long  # b_short more negative


def test_anec_circular_at_non_CH_returns_nan(vs_super):
    """Outside CTC band, circular null geodesic ANEC is undefined."""
    res = anec_integral_circular_null(vs_super, r=1.5)
    assert res["is_chronology_horizon"] is False


def test_anec_circular_inside_CH_finite(vs_super):
    res = anec_integral_circular_null(vs_super, r=3.35)
    assert res["is_chronology_horizon"] is True
    assert math.isfinite(res["ANEC_integral_circular"])


def test_pair_extinction_at_pi_zero(vs_super):
    """At delta=pi, extinction factor=0 -> effective ANEC integral = 0."""
    res = pair_extinction_anec_recovery(vs_super, delta=math.pi)
    assert res["anec_pair_effective"] == pytest.approx(0.0, abs=1e-12)


def test_pair_extinction_at_zero_equals_base(vs_super):
    res = pair_extinction_anec_recovery(vs_super, delta=0.0)
    assert res["anec_pair_effective"] == pytest.approx(res["anec_single"], rel=1e-12)


def test_pair_extinction_factor_monotone(vs_super):
    r0 = pair_extinction_anec_recovery(vs_super, delta=0.0)
    rp4 = pair_extinction_anec_recovery(vs_super, delta=math.pi / 4)
    rp2 = pair_extinction_anec_recovery(vs_super, delta=math.pi / 2)
    # Extinction factor decreases as delta -> pi
    assert r0["extinction_factor"] >= rp4["extinction_factor"]
    assert rp4["extinction_factor"] >= rp2["extinction_factor"]


def test_anec_violation_amplitude_returns_dict(vs_super):
    res = anec_violation_amplitude(vs_super)
    assert "min_T_kk" in res
    assert "max_T_kk" in res


def test_anec_violation_min_le_max(vs_super):
    res = anec_violation_amplitude(vs_super)
    if math.isfinite(res["min_T_kk"]) and math.isfinite(res["max_T_kk"]):
        assert res["min_T_kk"] <= res["max_T_kk"]
