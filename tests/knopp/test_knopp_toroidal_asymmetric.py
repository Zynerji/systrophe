"""Tests for the asymmetric (q, alpha) Toroidal Knopp binary scaffold."""

import math

import pytest

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_asymmetric import (
    AsymmetricKerrBinary,
    AsymmetricRescueVerdict,
    asymmetric_rescue_verdict,
    scan_parameter_space,
    summarise_asymmetric_scan,
)


# ----- AsymmetricKerrBinary construction & validation --------------------


def test_construction_defaults():
    b = AsymmetricKerrBinary()
    assert b.M_1 == 1.0 and b.M_2 == 1.0
    assert b.theta_1 == 0.0 and b.theta_2 == math.pi


def test_q_and_chirp_mass_formulas():
    b = AsymmetricKerrBinary(M_1=2.0, M_2=1.0)
    assert b.q == 0.5
    assert b.total_mass == 3.0
    assert b.reduced_mass == pytest.approx(2.0 * 1.0 / 3.0, rel=1e-12)
    # M_chirp = (M_1 M_2)^(3/5) / (M_1 + M_2)^(1/5)
    expected = 2.0 ** 0.6 / 3.0 ** 0.2
    assert b.chirp_mass == pytest.approx(expected, rel=1e-12)


def test_misalignment_angle():
    b = AsymmetricKerrBinary(theta_1=0.0, theta_2=math.pi)
    assert b.misalignment_angle == pytest.approx(-math.pi, rel=1e-12)


def test_rejects_M2_greater_than_M1():
    with pytest.raises(ValueError):
        AsymmetricKerrBinary(M_1=1.0, M_2=2.0)


def test_rejects_negative_mass():
    with pytest.raises(ValueError):
        AsymmetricKerrBinary(M_1=-1.0)


def test_rejects_chi_out_of_range():
    with pytest.raises(ValueError):
        AsymmetricKerrBinary(chi_1=1.5)


def test_rejects_theta_out_of_range():
    with pytest.raises(ValueError):
        AsymmetricKerrBinary(theta_1=-0.1)
    with pytest.raises(ValueError):
        AsymmetricKerrBinary(theta_2=math.pi + 0.1)


def test_rejects_nonpositive_d():
    with pytest.raises(ValueError):
        AsymmetricKerrBinary(d=0.0)


# ----- omega_eff and t_eff ------------------------------------------------


def test_omega_eff_recovers_symmetric_case():
    """At q=1, alpha=pi, the band must match EffectiveToroidalKerrBinary."""
    b_asym = AsymmetricKerrBinary(
        M_1=1.0, M_2=1.0, chi_1=1.0, chi_2=1.0,
        theta_1=0.0, theta_2=math.pi, d=2.0,
    )
    b_sym = EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)
    edges_a = b_asym.ctc_band_edges()
    edges_s = b_sym.ctc_band_edges(include_phi=False)
    assert edges_a[0] == pytest.approx(edges_s[0], rel=1e-6)
    assert edges_a[1] == pytest.approx(edges_s[1], rel=1e-6)


def test_omega_eff_zero_at_corotating_alignment():
    """alpha = 0 (perfect co-rotation) -> kappa = 0 -> Omega_eff -> 0."""
    b = AsymmetricKerrBinary(
        theta_1=0.0, theta_2=0.0, M_1=1.0, M_2=1.0, d=2.0,
    )
    assert b.omega_eff(1.0) == pytest.approx(0.0, abs=1e-30)
    assert b.t_eff(1.0) == pytest.approx(0.0, abs=1e-30)


def test_omega_eff_rejects_negative_rho():
    b = AsymmetricKerrBinary()
    with pytest.raises(ValueError):
        b.omega_eff(-0.5)


# ----- band detection ----------------------------------------------------


def test_has_band_for_tight_symmetric():
    b = AsymmetricKerrBinary(d=2.0)
    assert b.has_toroidal_ctc_band() is True


def test_no_band_for_wide_or_corotating():
    # Wide
    b_wide = AsymmetricKerrBinary(d=10.0)
    assert b_wide.has_toroidal_ctc_band() is False
    # Co-rotating (kappa=0 -> Omega_eff=0 -> no band)
    b_co = AsymmetricKerrBinary(theta_1=0.0, theta_2=0.0, d=2.0)
    assert b_co.has_toroidal_ctc_band() is False


# ----- stability ---------------------------------------------------------


def test_n_orbits_equal_mass_matches_symmetric_corrected():
    """At equal mass, our canonical-Peters t_merge formula
    (5/256) d^4 / (M_1 M_2 M_tot) matches the symmetric module's
    *corrected* merger time (which includes the SS halving). The
    symmetric module's uncorrected leading order is exactly 2x this
    due to a convention difference in the prefactor; the SS correction
    coincidentally halves it back to the canonical Peters value."""
    b_asym = AsymmetricKerrBinary(M_1=1.0, M_2=1.0, d=2.0)
    from systrophe.knopp.knopp_toroidal_stability import (
        corrected_merger_time, orbital_frequency,
    )
    b_sym = EffectiveToroidalKerrBinary(M=1.0, d=2.0)
    T_orb_sym = 2.0 * math.pi / orbital_frequency(b_sym)
    n_sym = corrected_merger_time(b_sym) / T_orb_sym
    n_asym = b_asym.n_orbits_to_merger()
    assert n_asym == pytest.approx(n_sym, rel=1e-6)


def test_n_orbits_grows_for_unequal_mass():
    """Smaller M_2 -> longer merger time at fixed d."""
    b1 = AsymmetricKerrBinary(M_1=1.0, M_2=1.0, d=2.0)
    b2 = AsymmetricKerrBinary(M_1=1.0, M_2=0.1, d=2.0)
    assert b2.n_orbits_to_merger() > b1.n_orbits_to_merger()


# ----- rescue verdict ----------------------------------------------------


def test_rescue_verdict_returns_dataclass():
    v = asymmetric_rescue_verdict(q=1.0, alpha=math.pi, d=2.0)
    assert isinstance(v, AsymmetricRescueVerdict)


def test_rescue_scan_returns_list():
    verdicts = scan_parameter_space(
        q_values=(1.0, 0.5), alpha_values=(math.pi,),
        d_values=(2.0, 10.0),
    )
    assert len(verdicts) == 4
    for v in verdicts:
        assert isinstance(v, AsymmetricRescueVerdict)


def test_rescue_scan_finds_no_viable():
    """Documented standing finding: NO (q, alpha, d) point yields
    BAND + > 1 orbit under the linear LT + Peters analysis."""
    verdicts = scan_parameter_space()
    n_viable = sum(1 for v in verdicts if v.viable)
    assert n_viable == 0


def test_summary_includes_no_viable_message():
    verdicts = scan_parameter_space()
    s = summarise_asymmetric_scan(verdicts)
    assert "NO VIABLE configurations" in s
    assert "Rescue path #3" in s
