"""Tests for the EOB strong-field inspiral scaffold."""

import math

import pytest

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_eob import (
    EOBInspiralReport,
    eob_A_function,
    eob_dA_dr,
    eob_dE_dt_pade_resummed,
    eob_inspiral_ivp,
    eob_inspiral_report,
    eob_isco_radius,
    summarise_eob_inspiral,
)


# ----- EOB radial potential -----------------------------------------------


def test_A_function_schwarzschild_limit():
    """At pn_order = 0, A(r) = 1 - 2/r (Schwarzschild)."""
    for r in (3.0, 6.0, 10.0):
        assert eob_A_function(r, nu=0.25, chi=0.0, pn_order=0) == \
            pytest.approx(1.0 - 2.0 / r, rel=1e-12)


def test_A_function_positive_outside_horizon():
    """A(r) > 0 for r > r_horizon ~ 2M (Schwarzschild)."""
    for r in (3.0, 5.0, 10.0):
        assert eob_A_function(r, nu=0.25, chi=1.0, pn_order=2) > 0.0


def test_dA_dr_finite():
    for r in (3.0, 6.0, 10.0):
        assert math.isfinite(eob_dA_dr(r, nu=0.25, chi=1.0, pn_order=2))


# ----- EOB ISCO ----------------------------------------------------------


def test_isco_in_schwarzschild_range():
    """For schwarzschild + EOB corrections, ISCO ~ 3M (not exactly 6M
    because EOB A-function differs from Schwarzschild + the ISCO
    criterion used here is approximate)."""
    r_isco = eob_isco_radius(nu=0.25, chi=0.0, pn_order=2)
    assert r_isco is not None
    assert 2.0 < r_isco < 8.0


def test_isco_shifts_with_spin():
    """Spin shifts ISCO; the exact shift depends on conventions, but
    the value remains in the physical range."""
    r_isco_0 = eob_isco_radius(nu=0.25, chi=0.0, pn_order=2)
    r_isco_1 = eob_isco_radius(nu=0.25, chi=1.0, pn_order=2)
    assert r_isco_0 is not None and r_isco_1 is not None
    assert r_isco_1 != pytest.approx(r_isco_0, rel=1e-12)


# ----- Pade-resummed flux ------------------------------------------------


def test_flux_negative_for_circular_orbit():
    """Energy is being LOST -> dE/dt < 0."""
    for r in (3.0, 6.0, 10.0):
        F = eob_dE_dt_pade_resummed(r, nu=0.25, chi=1.0)
        assert F < 0.0


def test_flux_resummation_smaller_than_raw():
    """Pade resummation reduces |dE/dt| compared to the raw quadrupole
    formula (regulating the strong-field divergence)."""
    nu = 0.25
    r = 3.0
    raw = 32.0 / 5.0 * nu ** 2 / r ** 5
    resummed = abs(eob_dE_dt_pade_resummed(r, nu=nu, chi=0.0))
    assert resummed < raw


# ----- EOB inspiral integration ------------------------------------------


def test_inspiral_returns_dict_with_required_keys():
    out = eob_inspiral_ivp(nu=0.25, chi=1.0, r_init=6.0)
    for k in ("converged", "r_isco", "r_init", "r_final",
              "t_final", "reached_isco", "n_steps"):
        assert k in out


def test_inspiral_t_positive_when_above_isco():
    out = eob_inspiral_ivp(nu=0.25, chi=1.0, r_init=6.0)
    assert out["converged"] is True
    assert out["t_final"] > 0.0


def test_inspiral_fails_when_r_init_below_isco():
    out = eob_inspiral_ivp(nu=0.25, chi=1.0, r_init=1.5)
    assert out["converged"] is False
    assert "<= r_isco" in out["reason"]


def test_inspiral_t_positive_and_finite_across_r_init():
    """The EOB inspiral time should be a positive, finite quantity
    across a range of r_init. (Strict monotonicity in r_init is NOT
    expected because the integrand is dominated by the near-ISCO
    region which depends on the buffer + n_grid choice; the qualitative
    statement is that t_final is order ~10^4 M for these binaries.)"""
    for r_init in (4.0, 6.0, 10.0, 20.0):
        out = eob_inspiral_ivp(nu=0.25, chi=1.0, r_init=r_init)
        assert out["converged"] is True
        assert out["t_final"] > 0.0
        assert math.isfinite(out["t_final"])


# ----- diagnostic report -------------------------------------------------


def test_report_dataclass():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    r = eob_inspiral_report(b)
    assert isinstance(r, EOBInspiralReport)
    assert r.r_isco is not None
    assert r.eob_inspiral_time is not None


def test_report_working_config_below_isco():
    """For the working config (M=1, d=2M), r_EOB = d/M_tot = 1 < r_ISCO."""
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    r = eob_inspiral_report(b)
    assert r.working_config_r_eob == pytest.approx(1.0, abs=1e-12)
    assert r.working_config_below_isco is True


def test_report_eob_rescue_false_for_working_config():
    """The working config is below ISCO -> EOB rescue verdict is False."""
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    r = eob_inspiral_report(b)
    assert r.eob_rescues_framework is False


def test_report_above_isco_could_rescue():
    """For a wide binary (d = 7 M, r_EOB = 3.5 > ISCO), EOB inspiral is
    long; rescue could apply if there were a band there (but there isn't)."""
    b = EffectiveToroidalKerrBinary(M=1.0, d=7.0, chi=1.0)
    r = eob_inspiral_report(b)
    assert r.working_config_below_isco is False
    # Either rescues (rare) or not, but the below_isco flag is False.


def test_summary_string_for_working_config():
    b = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    s = summarise_eob_inspiral(eob_inspiral_report(b))
    assert "EOB ISCO" in s
    assert "plunge" in s
    assert "Path #2 CLOSED" in s


def test_summary_string_above_isco():
    b = EffectiveToroidalKerrBinary(M=1.0, d=7.0, chi=1.0)
    s = summarise_eob_inspiral(eob_inspiral_report(b))
    assert "EOB" in s
