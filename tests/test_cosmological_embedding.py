"""Tests for cosmological embedding module."""

import math

import pytest

from systrophe.cosmological_embedding import (
    cosmological_corrections_to_alpha,
    cylinder_evaporates_under_expansion,
    frw_scale_factor,
    horizon_chirp_rate,
    hubble_parameter,
    inflation_creates_systrophe_cylinders,
    matching_residual,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_frw_scale_factor_matter_grows():
    a1 = frw_scale_factor(1.0, equation_of_state="matter")
    a10 = frw_scale_factor(10.0, equation_of_state="matter")
    assert a10 > a1


def test_frw_scale_factor_deSitter_exponential():
    H0 = 0.1
    a1 = frw_scale_factor(1.0, H0=H0, equation_of_state="deSitter")
    a2 = frw_scale_factor(2.0, H0=H0, equation_of_state="deSitter")
    assert a2 == pytest.approx(a1 * math.exp(H0), rel=1e-10)


def test_frw_scale_factor_unknown_eos_raises():
    with pytest.raises(ValueError):
        frw_scale_factor(1.0, equation_of_state="quintessence")


def test_hubble_parameter_deSitter_constant():
    H0 = 0.07
    H1 = hubble_parameter(1.0, H0=H0, equation_of_state="Lambda")
    H100 = hubble_parameter(100.0, H0=H0, equation_of_state="Lambda")
    assert H1 == H100 == H0


def test_hubble_parameter_matter_decreases():
    H1 = hubble_parameter(1.0, equation_of_state="matter")
    H100 = hubble_parameter(100.0, equation_of_state="matter")
    assert H100 < H1


def test_matching_residual_returns_dict(vs_super):
    res = matching_residual(vs_super, r_match=10.0)
    assert "F_residual" in res
    assert "L_residual" in res
    assert res["F_residual"] >= 0
    assert res["L_residual"] >= 0


def test_matching_residual_scale_factor_present(vs_super):
    res = matching_residual(vs_super, r_match=10.0, cosmic_time=1.0)
    assert res["scale_factor"] > 0


def test_cosmological_corrections_subcritical_returns_zero(vs_sub):
    res = cosmological_corrections_to_alpha(vs_sub, cosmic_time=10.0)
    assert res["alpha_bare"] == 0.0


def test_cosmological_corrections_supercritical_finite(vs_super):
    res = cosmological_corrections_to_alpha(vs_super, cosmic_time=100.0)
    assert math.isfinite(res["alpha_bare"])
    assert math.isfinite(res["a_eff"])


def test_cosmological_corrections_alpha_less_than_bare_for_finite_H(vs_super):
    """Hubble drag should reduce alpha relative to bare."""
    res = cosmological_corrections_to_alpha(vs_super, cosmic_time=1.0,
                                            equation_of_state="Lambda", H0=0.1)
    if res["still_supercritical"]:
        # omega_eff < omega so alpha_corrected < alpha_bare
        assert res["alpha_corrected"] <= res["alpha_bare"] + 1e-9


def test_horizon_chirp_rate_returns_dict(vs_super):
    chirp = horizon_chirp_rate(vs_super, cosmic_time=10.0)
    assert "first_CH_drift_rate" in chirp


def test_horizon_chirp_subcritical_zero(vs_sub):
    chirp = horizon_chirp_rate(vs_sub, cosmic_time=10.0)
    assert chirp["first_CH_drift_rate"] == 0.0


def test_cylinder_evaporates_returns_dict(vs_super):
    res = cylinder_evaporates_under_expansion(vs_super, equation_of_state="Lambda", H0=10.0)
    assert "evaporates" in res


def test_cylinder_evaporates_subcritical_skip(vs_sub):
    res = cylinder_evaporates_under_expansion(vs_sub)
    assert res["evaporates"] is False


def test_cylinder_evaporates_high_H_true():
    """With very high Hubble, an originally supercritical cylinder evaporates."""
    vs = VanStockumInterior(omega=0.6, R=1.0)
    res = cylinder_evaporates_under_expansion(vs, equation_of_state="Lambda", H0=0.5,
                                              time_horizon=10.0)
    assert res["evaporates"] is True


def test_inflation_creates_returns_dict():
    res = inflation_creates_systrophe_cylinders()
    assert "creation_rate_per_H4" in res
    assert "exponentially_suppressed" in res


def test_inflation_high_H_less_suppressed():
    low_H = inflation_creates_systrophe_cylinders(H_inflation=1e-10)
    high_H = inflation_creates_systrophe_cylinders(H_inflation=1e-2)
    # High H => smaller S_E => less suppression
    assert high_H["euclidean_action"] < low_H["euclidean_action"]
