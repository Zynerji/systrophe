"""Tests for the nested-dodecahedron Casimir amplification module."""

import math

import pytest

from systrophe.knopp.knopp_toroidal_casimir_dodecahedron import (
    GapBridgeReport,
    PHI,
    PLANCK_LENGTH,
    amplified_casimir_density,
    dodecahedron_mode_degeneracy,
    gap_bridge_report,
    nested_cavity_amplification,
    parallel_plate_casimir_density,
    required_density_for_binary,
    single_cavity_amplification,
    summarise_gap_bridge,
)


# ----- constants ----------------------------------------------------------


def test_phi_is_golden_ratio():
    assert PHI == pytest.approx((1.0 + math.sqrt(5.0)) / 2.0, rel=1e-12)


def test_planck_length_value():
    assert PLANCK_LENGTH == pytest.approx(1.616e-35, rel=1e-3)


# ----- dodecahedron mode degeneracy --------------------------------------


def test_mode_degeneracy_formula():
    """g_dodec = 12 * 20 / phi^2."""
    g = dodecahedron_mode_degeneracy()
    expected = 12 * 20 / PHI ** 2
    assert g == pytest.approx(expected, rel=1e-12)
    # Sanity: ~91.7
    assert 80 < g < 100


# ----- single-cavity amplification ---------------------------------------


def test_single_cavity_amplification_returns_dataclass():
    s = single_cavity_amplification(Q=1e4)
    assert s.Q == 1e4
    assert s.g_dodec == pytest.approx(dodecahedron_mode_degeneracy(), rel=1e-12)
    assert s.amplification == pytest.approx(s.Q * s.g_dodec, rel=1e-12)


def test_single_cavity_rejects_bad_Q():
    with pytest.raises(ValueError):
        single_cavity_amplification(Q=-1.0)


# ----- nested cavity amplification ---------------------------------------


def test_nested_cavity_amplification_one_shell_equals_single():
    """N=1 reduces to single-cavity amplification."""
    n = nested_cavity_amplification(n_shells=1, Q_per_shell=1e4,
                                     eta_coupling=1.0)
    s = single_cavity_amplification(Q=1e4)
    assert n.total_amplification == pytest.approx(
        s.amplification, rel=1e-12,
    )


def test_nested_cavity_amplification_multiplies():
    """Total amp = (g * Q * eta)^N."""
    n = nested_cavity_amplification(n_shells=5, Q_per_shell=1e4,
                                     eta_coupling=0.5)
    A_per = dodecahedron_mode_degeneracy() * 1e4 * 0.5
    assert n.total_amplification == pytest.approx(
        A_per ** 5, rel=1e-12,
    )


def test_nested_cavity_innermost_golden_ratio_scaling():
    """R_inner / R_outer = phi^{-(N-1)}."""
    n = nested_cavity_amplification(n_shells=3, Q_per_shell=1e4,
                                     eta_coupling=0.5,
                                     outer_radius_m=1.0)
    assert n.innermost_shell_radius == pytest.approx(PHI ** -2, rel=1e-12)


def test_nested_cavity_breaks_below_planck_at_extreme_depth():
    """For very deep nesting (~170 shells at golden ratio), innermost
    shell must go sub-Planck starting from R_outer = 1 m."""
    n = nested_cavity_amplification(n_shells=170, Q_per_shell=1e4,
                                     eta_coupling=0.5,
                                     outer_radius_m=1.0)
    assert n.breaks_below_planck is True


def test_nested_cavity_rejects_bad_inputs():
    with pytest.raises(ValueError):
        nested_cavity_amplification(n_shells=0)
    with pytest.raises(ValueError):
        nested_cavity_amplification(eta_coupling=1.5)
    with pytest.raises(ValueError):
        nested_cavity_amplification(outer_radius_m=-1.0)


# ----- Casimir density --------------------------------------------------


def test_casimir_density_formula():
    """u_Cas = -pi^2 hbar c / (240 d^4)."""
    d = 1.0
    u = parallel_plate_casimir_density(d)
    expected = -(math.pi ** 2) / 240.0 * 1.054571817e-34 * 2.998e8
    assert u == pytest.approx(expected, rel=1e-6)


def test_casimir_density_negative():
    """Standard Casimir is attractive (negative energy density)."""
    assert parallel_plate_casimir_density(d_m=1.0) < 0.0


def test_casimir_density_scales_inverse_fourth():
    u1 = abs(parallel_plate_casimir_density(d_m=1.0))
    u2 = abs(parallel_plate_casimir_density(d_m=2.0))
    assert u1 / u2 == pytest.approx(16.0, rel=1e-12)


def test_amplified_casimir_density_scales_with_amplification():
    u_amp = abs(amplified_casimir_density(d_m=1.0, A_total=100.0))
    u_unamp = abs(parallel_plate_casimir_density(d_m=1.0))
    assert u_amp == pytest.approx(100.0 * u_unamp, rel=1e-12)


# ----- required density for binary ---------------------------------------


def test_required_density_positive():
    """u_required is reported as magnitude (always > 0)."""
    u_req = required_density_for_binary(M_solar=1.0, d_geom=2.0)
    assert u_req > 0.0
    assert math.isfinite(u_req)


def test_required_density_order_of_magnitude():
    """For M_sun, d=2M: u_required ~ 1e36-1e38 J/m^3."""
    u = required_density_for_binary(M_solar=1.0, d_geom=2.0)
    assert 1e35 < u < 1e39


def test_required_density_scales_strongly_with_d():
    """P_GW ~ 1/d^5 and V ~ d^3 so u_req ~ 1/(d^5 * d^3 / d^5) ~ 1/d^3,
    but tau scales as d^{3/2}, so u_req ~ d^{-7/2}. Approximately."""
    u_tight = required_density_for_binary(M_solar=1.0, d_geom=2.0)
    u_wide = required_density_for_binary(M_solar=1.0, d_geom=10.0)
    assert u_tight > u_wide
    # 5-fold separation -> ~ 10^4 ratio
    assert u_tight / u_wide > 100.0


# ----- gap-bridge report -------------------------------------------------


def test_gap_bridge_report_returns_dataclass():
    r = gap_bridge_report()
    assert isinstance(r, GapBridgeReport)


def test_gap_bridge_factor_huge():
    """For stellar BH binary, gap is ~10^77-10^78."""
    r = gap_bridge_report(M_solar=1.0, d_geom=2.0)
    assert 1e75 < r.gap_factor < 1e80


def test_gap_bridge_n_shells_monotone_in_Q():
    """Higher Q per shell -> fewer shells needed."""
    r = gap_bridge_report()
    assert r.n_shells_needed_at_Q_1e6 < r.n_shells_needed_at_Q_1e4
    assert r.n_shells_needed_at_Q_1e4 < r.n_shells_needed_at_Q_1e3


def test_gap_bridge_verdict_mentions_caveat():
    """The verdict must contain the critical physics caveat
    distinguishing coherent modes from vacuum fluctuations."""
    r = gap_bridge_report()
    assert "CRITICAL PHYSICS CAVEAT" in r.verdict
    assert "vacuum" in r.verdict.lower()
    assert "coherent" in r.verdict.lower()


def test_summary_string_well_formed():
    r = gap_bridge_report()
    s = summarise_gap_bridge(r)
    assert "amplification required" in s
    assert "innermost shell" in s
    assert "Ford-Roman" in s
    assert "VERDICT" in s
