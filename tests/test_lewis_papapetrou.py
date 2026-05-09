"""Lewis-Papapetrou exterior integrator tests."""

import numpy as np
import pytest

from systrophe.lewis_papapetrou import integrate_lp_exterior
from systrophe.vanstockum import VanStockumInterior


def test_initial_conditions_match_interior():
    """At r = R+ : F = 1, F' = 0, K = omega R^2, h = exp(-omega^2 R^2)."""
    omega, R = 1.0, 1.0
    sol = integrate_lp_exterior(omega_dust=omega, R=R, r_max=2.0, n_samples=201)
    assert sol.r[0] == pytest.approx(R)
    assert sol.F[0] == pytest.approx(1.0, abs=1e-10)
    assert sol.K[0] == pytest.approx(omega * R * R, abs=1e-10)
    assert sol.h[0] == pytest.approx(np.exp(-omega * omega * R * R), abs=1e-10)
    assert sol.c == pytest.approx(2.0 * omega)


def test_supercritical_matches_analytic_case_III():
    """Numerical F(r) agrees with closed-form Tipler sinusoid for a > 1/2."""
    omega, R = 1.0, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    sol = integrate_lp_exterior(omega_dust=omega, R=R, r_max=8.0, n_samples=4001)
    r_test = np.array([1.5, 2.0, 3.0, 5.0])
    F_num = np.interp(r_test, sol.r, sol.F)
    F_ana = vs.analytic_exterior_F(r_test)
    rel = np.abs(F_num - F_ana) / (np.abs(F_ana) + 1e-12)
    assert rel.max() < 5e-3, f"max rel err = {rel.max()}"


def test_critical_F_zero_at_e_R():
    """At a = 1/2 (Bonnor Case II), F first crosses zero at r = e R."""
    omega, R = 0.5, 1.0
    sol = integrate_lp_exterior(omega_dust=omega, R=R, r_max=10.0, n_samples=4001)
    # at least one zero, and the first one is near e
    assert sol.F_zeros.size >= 1
    assert sol.F_zeros[0] == pytest.approx(np.e * R, rel=1e-3)


def test_log_periodic_zero_spacing_supercritical():
    """In Case III, adjacent F=0 crossings are separated by a constant in ln r.

    Spacing in ln r should equal pi / alpha. We test at moderate a = 1 in
    a range that exercises the basic numerical integrator without
    triggering platform-specific Radau step-size issues at higher a.
    For a > 1.2 use the robust regime-dispatching solver instead
    (tested in test_lp_robust.py).
    """
    omega, R = 1.0, 1.0
    sol = integrate_lp_exterior(omega_dust=omega, R=R, r_max=15.0, n_samples=8001)
    assert sol.F_zeros.size >= 2
    log_spacings = np.diff(np.log(sol.F_zeros / R))
    expected = np.pi / np.sqrt(4.0 * (omega * R) ** 2 - 1.0)
    assert np.allclose(log_spacings, expected, rtol=5e-2), (
        f"spacings={log_spacings}, expected={expected}"
    )


def test_subcritical_has_one_F_zero():
    """For a < 1/2 (Case I), F has exactly one zero in any reasonable interval."""
    omega, R = 0.3, 1.0
    sol = integrate_lp_exterior(omega_dust=omega, R=R, r_max=20.0, n_samples=4001)
    assert sol.F_zeros.size == 1


def test_constraint_FL_plus_K2_equals_r2():
    """Canonical Weyl constraint F L + K^2 = r^2 in the pre-F=0 region.

    L is *defined* as (r^2 - K^2)/F so the constraint is exact by
    construction in exact arithmetic. Floating-point recovery loses
    precision near F = 0; this test stays in the regime where the
    integrator is well-conditioned (a = 0.7, integration domain stops
    before the first F-zero at r ~ 2.2).
    """
    omega, R = 0.7, 1.0
    sol = integrate_lp_exterior(omega_dust=omega, R=R, r_max=2.0, n_samples=2001)
    lhs = sol.F * sol.L + sol.K ** 2
    rhs = sol.r ** 2
    rel = np.abs(lhs - rhs) / (np.abs(rhs) + 1e-12)
    assert rel.max() < 1e-6, f"max constraint violation = {rel.max()}"


def test_zero_omega_is_minkowski_exterior():
    """omega_dust = 0 -> Minkowski cylindrical exterior: F = 1, K = 0, L = r^2."""
    sol = integrate_lp_exterior(omega_dust=0.0, R=1.0, r_max=5.0, n_samples=1001)
    assert np.allclose(sol.F, 1.0, atol=1e-9)
    assert np.allclose(sol.K, 0.0, atol=1e-9)
    assert np.allclose(sol.L, sol.r ** 2, rtol=1e-9)


def test_tipler_sinusoid_F_matches_analytic_to_machine_precision():
    """vs.tipler_sinusoid_F().L(r) reproduces vs.analytic_exterior_F(r) exactly."""
    vs = VanStockumInterior(omega=1.5, R=1.0)
    ts = vs.tipler_sinusoid_F()
    r = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    f_ana = vs.analytic_exterior_F(r)
    f_ts = ts.L(r)
    np.testing.assert_allclose(f_ts, f_ana, rtol=1e-12, atol=1e-12)


def test_tipler_sinusoid_L_matches_analytic_to_machine_precision():
    """vs.tipler_sinusoid_L().L(r) reproduces vs.analytic_exterior_L(r) exactly."""
    vs = VanStockumInterior(omega=1.5, R=1.0)
    ts = vs.tipler_sinusoid_L()
    r = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    L_ana = vs.analytic_exterior_L(r)
    L_ts = ts.L(r)
    # match to ~ machine eps; small absolute tolerance for L(R) which is identically 0
    np.testing.assert_allclose(L_ts, L_ana, rtol=1e-12, atol=1e-12)


def test_constraint_FL_plus_K2_holds_analytically():
    """Closed-form (F, K, L) satisfy F L + K^2 = r^2 to machine precision."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = np.linspace(1.05, 1.7, 30)  # before first F-zero at r ~ 1.83
    F = vs.analytic_exterior_F(r)
    K = vs.analytic_exterior_K(r)
    L = vs.analytic_exterior_L(r)
    np.testing.assert_allclose(F * L + K * K, r * r, rtol=1e-12, atol=1e-12)


def test_K_continuity_at_R():
    """Analytic exterior K(R+) matches interior K(R-) = omega R^2."""
    omega, R = 1.2, 0.7
    vs = VanStockumInterior(omega=omega, R=R)
    assert vs.analytic_exterior_K(R) == pytest.approx(omega * R * R, abs=1e-10)


def test_L_continuity_at_R():
    """Analytic exterior L(R+) matches interior L(R-) = R^2(1 - (omega R)^2).

    For supercritical (a > 1/2 but a < 1) the interior L(R) is positive;
    for a > 1, interior L(R) is negative (CTC right at the boundary).
    """
    omega, R = 1.2, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    expected = R * R * (1.0 - (omega * R) ** 2)
    assert vs.analytic_exterior_L(R) == pytest.approx(expected, abs=1e-10)


def test_subcritical_analytic_continuity_at_R():
    """Subcritical analytic F(R)=1, F'(R)=0, K(R)=omega R^2, L(R)=R^2(1-a^2)."""
    omega, R = 0.3, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    assert vs.regime == "subcritical"
    assert vs.analytic_exterior_F(R) == pytest.approx(1.0, abs=1e-10)
    assert vs.analytic_exterior_K(R) == pytest.approx(omega * R * R, abs=1e-10)
    expected_L = R * R * (1.0 - (omega * R) ** 2)
    assert vs.analytic_exterior_L(R) == pytest.approx(expected_L, abs=1e-10)


def test_critical_analytic_continuity_at_R():
    """Critical (a = 1/2) F(R)=1, F'(R)=0, K(R)=omega R^2, L(R)=3R^2/4."""
    omega, R = 0.5, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    assert vs.regime == "critical"
    assert vs.analytic_exterior_F(R) == pytest.approx(1.0, abs=1e-10)
    assert vs.analytic_exterior_K(R) == pytest.approx(omega * R * R, abs=1e-10)
    assert vs.analytic_exterior_L(R) == pytest.approx(3.0 * R * R / 4.0, abs=1e-10)


def test_critical_F_zero_at_e_R_analytically():
    """Critical: F = (r/R)(1 - ln(r/R)) = 0 -> r = e R exactly."""
    vs = VanStockumInterior(omega=0.5, R=1.0)
    assert vs.analytic_exterior_F(np.e) == pytest.approx(0.0, abs=1e-12)


def test_critical_K_simple_form():
    """Critical K(r) = (r/2)(1 + ln(r/R)) — closed form (a = 1/2)."""
    # a = omega R = 1/2 requires omega R = 1/2; with R=1 use omega=0.5.
    omega, R = 0.5, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    for r in [1.0, 1.5, 2.0, np.e]:
        expected = (r / 2.0) * (1.0 + np.log(r / R))
        assert vs.analytic_exterior_K(r) == pytest.approx(expected, abs=1e-12)


def test_critical_L_simple_form():
    """Critical L(r) = (r R / 4)(3 + ln(r/R)) — closed form."""
    omega, R = 0.5, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    for r in [1.0, 1.5, 2.0, np.e, 5.0]:
        expected = (r * R / 4.0) * (3.0 + np.log(r / R))
        assert vs.analytic_exterior_L(r) == pytest.approx(expected, abs=1e-12)


def test_subcritical_constraint_FL_plus_K2():
    """F L + K^2 = r^2 holds identically for the subcritical analytic forms."""
    vs = VanStockumInterior(omega=0.3, R=1.0)
    r = np.linspace(1.05, 3.5, 50)  # before F-zero at ~3.95
    F = vs.analytic_exterior_F(r)
    K = vs.analytic_exterior_K(r)
    L = vs.analytic_exterior_L(r)
    np.testing.assert_allclose(F * L + K * K, r * r, rtol=1e-12, atol=1e-12)


def test_critical_constraint_FL_plus_K2():
    """F L + K^2 = r^2 holds identically for the critical analytic forms."""
    vs = VanStockumInterior(omega=0.5, R=1.0)
    r = np.linspace(1.05, 2.5, 50)  # before F-zero at e
    F = vs.analytic_exterior_F(r)
    K = vs.analytic_exterior_K(r)
    L = vs.analytic_exterior_L(r)
    np.testing.assert_allclose(F * L + K * K, r * r, rtol=1e-12, atol=1e-12)


def test_subcritical_analytic_matches_numerical():
    """Subcritical analytic agrees with the numerical integrator well before F=0.

    The numerical K is computed via the twist quadrature with a 1/F^2
    integrand; near the F=0 surface this loses precision. We test in the
    well-conditioned region r <= 0.6 * F-zero radius.
    """
    omega, R = 0.3, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    sol = integrate_lp_exterior(omega_dust=omega, R=R, r_max=2.0, n_samples=2001)
    F_ana = vs.analytic_exterior_F(sol.r)
    K_ana = vs.analytic_exterior_K(sol.r)
    L_ana = vs.analytic_exterior_L(sol.r)
    np.testing.assert_allclose(sol.F, F_ana, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sol.K, K_ana, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(sol.L, L_ana, rtol=1e-5, atol=1e-5)
