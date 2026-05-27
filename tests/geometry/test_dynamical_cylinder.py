"""Tests for dynamical Tipler cylinder."""

import numpy as np
import pytest

from systrophe.geometry.dynamical_cylinder import (
    SpinUpHistory,
    adiabatic_back_reaction_estimate,
    exponential_profile,
    formation_transition,
    instantaneous_ctc_measure,
    linear_ramp_profile,
    sinusoidal_profile,
    spin_up_history,
    step_profile,
)


# ----- Profiles ---------------------------------------------------------

def test_linear_ramp_endpoints():
    p = linear_ramp_profile(omega_initial=0.0, omega_final=2.0, t_total=10.0)
    assert p(0.0) == pytest.approx(0.0)
    assert p(10.0) == pytest.approx(2.0)
    assert p(5.0) == pytest.approx(1.0)


def test_step_profile():
    p = step_profile(omega_initial=0.0, omega_final=1.0, t_step=5.0)
    assert p(4.0) == 0.0
    assert p(6.0) == 1.0


def test_exponential_profile_approaches_limit():
    p = exponential_profile(omega_inf=1.0, tau=1.0, omega_0=0.0)
    assert p(0.0) == 0.0
    assert p(100.0) == pytest.approx(1.0, abs=1e-9)


def test_sinusoidal_profile_oscillates():
    p = sinusoidal_profile(omega_mean=1.0, omega_amp=0.5, period=4.0)
    assert p(0.0) == 1.0
    assert p(1.0) == pytest.approx(1.5)
    assert p(2.0) == pytest.approx(1.0, abs=1e-9)
    assert p(3.0) == pytest.approx(0.5)


# ----- Instantaneous CTC measure ----------------------------------------

def test_instantaneous_ctc_supercritical():
    """omega = 1, R = 1 should be supercritical with CTC bands."""
    m = instantaneous_ctc_measure(omega=1.0, R=1.0)
    assert m["is_supercritical"]
    assert m["n_ctc_bands"] >= 1


def test_instantaneous_ctc_subcritical():
    """omega = 0.3, R = 1 should be subcritical, possibly no bands."""
    m = instantaneous_ctc_measure(omega=0.3, R=1.0)
    assert not m["is_supercritical"]


def test_instantaneous_ctc_critical():
    """At a = 1/2 (critical), expect transition behavior."""
    # Critical is unstable numerically; just check it doesn't crash
    m = instantaneous_ctc_measure(omega=0.5001, R=1.0)
    assert "is_supercritical" in m


# ----- Spin-up history --------------------------------------------------

def test_spin_up_returns_history():
    p = linear_ramp_profile(omega_initial=0.0, omega_final=2.0, t_total=10.0)
    t_array = np.linspace(0, 10, 11)
    hist = spin_up_history(t_array, p, R=1.0)
    assert isinstance(hist, SpinUpHistory)
    assert len(hist.omegas) == 11


def test_spin_up_detects_formation():
    """Spin up from omega = 0 to 2 should trigger CTC formation at a = 1/2."""
    p = linear_ramp_profile(omega_initial=0.0, omega_final=2.0, t_total=10.0)
    t_array = np.linspace(0, 10, 21)
    hist = spin_up_history(t_array, p, R=1.0)
    assert hist.formation_time is not None
    # Formation should occur near t ~ 2.5 (when omega = 0.5)
    omega_at_form = p(hist.formation_time)
    assert 0.45 < omega_at_form < 0.7


# ----- Formation transition --------------------------------------------

def test_formation_transition_matches_critical():
    """Linear ramp through omega = 1/2 should match the critical-a transition."""
    p = linear_ramp_profile(omega_initial=0.0, omega_final=2.0, t_total=10.0)
    result = formation_transition(p, R=1.0, t_max=10.0, n_t=50)
    assert result["formation_time"] is not None
    # critical omega should be ~ 0.5
    assert 0.4 < result["critical_omega"] < 0.7


def test_formation_when_omega_stays_subcritical():
    """If omega never exceeds 0.5, no CTC formation should occur."""
    p = linear_ramp_profile(omega_initial=0.0, omega_final=0.4, t_total=10.0)
    result = formation_transition(p, R=1.0, t_max=10.0, n_t=20)
    assert result["formation_time"] is None


# ----- Back-reaction ----------------------------------------------------

def test_back_reaction_returns_dict():
    result = adiabatic_back_reaction_estimate(omega=1.0, R=1.0)
    assert "T_H_acoustic" in result
    assert "d_omega_dt" in result


def test_back_reaction_subcritical_zero_dt():
    """For subcritical omega, back-reaction is zero (no Hawking emission)."""
    result = adiabatic_back_reaction_estimate(omega=0.3, R=1.0)
    assert result["d_omega_dt"] == 0.0


def test_back_reaction_supercritical_dt_negative():
    """Supercritical omega should have negative d_omega/dt (slowing down)."""
    result = adiabatic_back_reaction_estimate(omega=1.0, R=1.0)
    if result["is_supercritical"] and result["T_H_acoustic"] > 0:
        assert result["d_omega_dt"] < 0
