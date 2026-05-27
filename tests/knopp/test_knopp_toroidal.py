"""Tests for the Toroidal Knopp Drive (counter-rotating Kerr binary backend)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_toroidal import (
    EffectiveToroidalKerrBinary,
    KnoppToroidalBudget,
    KnoppToroidalConfig,
    knopp_toroidal_budget,
    summarise_toroidal_budget,
)


# ----- EffectiveToroidalKerrBinary ---------------------------------------


def test_critical_k_value():
    k_crit = EffectiveToroidalKerrBinary.critical_k()
    assert k_crit == pytest.approx(
        math.sqrt(3.0 * math.sqrt(3.0) / 8.0), rel=1e-12,
    )
    assert 0.8 < k_crit < 0.81


def test_binary_construction_validates_inputs():
    with pytest.raises(ValueError):
        EffectiveToroidalKerrBinary(M=-1.0, d=2.0)
    with pytest.raises(ValueError):
        EffectiveToroidalKerrBinary(M=1.0, d=-2.0)
    with pytest.raises(ValueError):
        EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.5)


def test_omega_eff_positive_and_decreasing():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    omegas = [b.omega_eff(rho) for rho in [0.5, 1.0, 2.0, 5.0, 10.0]]
    assert all(o > 0.0 for o in omegas)
    # Omega_eff = 4 chi M^2 / r^3 with r growing -> monotone decreasing.
    diffs = np.diff(omegas)
    assert np.all(diffs <= 0.0)


def test_omega_eff_formula_at_midplane():
    M, d, chi = 1.0, 2.0, 1.0
    b = EffectiveToroidalKerrBinary(M=M, d=d, chi=chi)
    rho = 1.5
    r = math.sqrt((d / 2.0) ** 2 + rho ** 2)
    expected = 4.0 * chi * M ** 2 / r ** 3
    assert b.omega_eff(rho) == pytest.approx(expected, rel=1e-12)


def test_omega_eff_rejects_negative_rho():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    with pytest.raises(ValueError):
        b.omega_eff(-0.1)


# ----- effective tilt ----------------------------------------------------


def test_t_eff_linear_LT_formula():
    """T_eff = Omega_eff * rho^2 with include_phi=False."""
    M, d, chi = 1.0, 2.0, 1.0
    b = EffectiveToroidalKerrBinary(M=M, d=d, chi=chi)
    rho = 1.5
    expected = 4.0 * chi * M ** 2 * rho ** 2 / (
        ((d / 2.0) ** 2 + rho ** 2) ** (3.0 / 2.0)
    )
    assert b.t_eff(rho, include_phi=False) == pytest.approx(expected, rel=1e-12)


def test_t_eff_phi_smaller_than_linear():
    """The (1-2 Phi_eff) denominator damps T_eff (Phi < 0 -> 1-2Phi > 1)."""
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    T_lin = b.t_eff(1.5, include_phi=False)
    T_phi = b.t_eff(1.5, include_phi=True)
    assert T_phi < T_lin


def test_tipler_gate_zero_in_band():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    # rho=1.5 is inside the band (linear LT)
    g = b.tipler_gate_eff(1.5, include_phi=False)
    assert g == 0.0


def test_tipler_gate_positive_outside_band():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    # rho=10 is well outside any band
    g = b.tipler_gate_eff(10.0, include_phi=False)
    assert g > 0.0


def test_tipler_gate_validates_c_gate():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    with pytest.raises(ValueError):
        b.tipler_gate_eff(1.5, c_gate=-0.1)
    with pytest.raises(ValueError):
        b.tipler_gate_eff(1.5, c_gate=1.5)


# ----- toroidal CTC band -------------------------------------------------


def test_band_exists_for_tight_binary():
    """k = 2 M / d = 1.0 > k_crit -> band exists (linear LT)."""
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    e_in, e_out = b.ctc_band_edges(include_phi=False)
    assert e_in is not None and e_out is not None
    assert e_out > e_in > 0.0


def test_band_absent_for_loose_binary():
    """k = 0.2 < k_crit -> no band (update.txt's d=10M example is subcritical)."""
    b = EffectiveToroidalKerrBinary(M=1.0, d=10.0)
    e_in, e_out = b.ctc_band_edges(include_phi=False)
    assert e_in is None and e_out is None


def test_has_toroidal_ctc_band_matches_edges():
    b_in = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    b_out = EffectiveToroidalKerrBinary(M=1.0, d=10.0)
    assert b_in.has_toroidal_ctc_band(include_phi=False) is True
    assert b_out.has_toroidal_ctc_band(include_phi=False) is False


# ----- composite budget --------------------------------------------------


def test_default_config_in_band_zero_exotic():
    b = knopp_toroidal_budget()
    assert b.inside_ctc_band
    assert b.tipler_gate_factor == 0.0
    assert b.composite_E_neg == 0.0
    assert b.final_zero_exotic


def test_subcritical_config_no_band():
    b = knopp_toroidal_budget(d=10.0, rho_orbit=6.0)
    assert b.inside_ctc_band is False
    assert b.tipler_gate_factor > 0.0
    assert b.composite_E_neg > 0.0
    assert b.final_zero_exotic is False


def test_budget_overrides_replace_dataclass_fields():
    b = knopp_toroidal_budget(M=2.0, d=3.0, rho_orbit=1.0, Q=50.0)
    assert b.config.M == 2.0
    assert b.config.d == 3.0
    assert b.config.Q == 50.0


def test_budget_returns_dataclass_with_required_fields():
    b = knopp_toroidal_budget()
    assert isinstance(b, KnoppToroidalBudget)
    for attr in ("omega_eff", "t_eff", "inside_ctc_band", "band_edges",
                 "tipler_gate_factor", "feedback_factor",
                 "horn_amplification", "krasnikov_bare_E_neg",
                 "composite_E_neg", "sustained_drive_power",
                 "natural_frequency", "cavity_lifetime_tau",
                 "pfenning_ford_compatible", "back_reaction_correction",
                 "final_E_neg", "final_zero_exotic"):
        assert hasattr(b, attr)


def test_feedback_factor_one_over_Q_squared():
    Q = 25.0
    b = knopp_toroidal_budget(Q=Q)
    assert b.feedback_factor == pytest.approx(1.0 / Q ** 2, rel=1e-12)


def test_horn_amplification_one_plus_epsilon():
    eps = 0.07
    b = knopp_toroidal_budget(epsilon_horn=eps)
    assert b.horn_amplification == pytest.approx(1.0 + eps, rel=1e-12)


def test_back_reaction_positive():
    b = knopp_toroidal_budget()
    assert b.back_reaction_correction >= 0.0


def test_back_reaction_decreases_with_M():
    # E_BR ~ 1/M^4: heavier binary -> smaller back-reaction.
    b_small_M = knopp_toroidal_budget(M=1.0, d=2.0, rho_orbit=1.5)
    b_large_M = knopp_toroidal_budget(M=10.0, d=20.0, rho_orbit=15.0)
    assert b_large_M.back_reaction_correction < b_small_M.back_reaction_correction


# ----- summary -----------------------------------------------------------


def test_summary_returns_string():
    b = knopp_toroidal_budget()
    s = summarise_toroidal_budget(b)
    assert isinstance(s, str)
    assert "Knopp-Toroidal" in s
    assert "T_eff" in s
    assert "zero exotic" in s
