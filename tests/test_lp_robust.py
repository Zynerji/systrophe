"""Tests for the regime-dispatching robust LP solver."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.lp_robust import LPRobustSolution, integrate_lp_robust


def test_subcritical_regime_dispatched():
    sol = integrate_lp_robust(omega_dust=0.3, R=1.0, r_max=10.0, n_samples=501)
    assert sol.regime == "subcritical"
    assert isinstance(sol, LPRobustSolution)


def test_critical_regime_dispatched():
    sol = integrate_lp_robust(omega_dust=0.5, R=1.0, r_max=10.0, n_samples=501)
    assert sol.regime == "critical"


def test_supercritical_regime_dispatched():
    sol = integrate_lp_robust(omega_dust=1.0, R=1.0, r_max=10.0, n_samples=501)
    assert sol.regime == "supercritical"


def test_supercritical_machine_precision_on_native_grid():
    """For a > 1/2, sol.F == analytic_exterior_F(sol.r) exactly."""
    omega, R = 2.0, 1.0
    sol = integrate_lp_robust(omega_dust=omega, R=R, r_max=50.0, n_samples=1001)
    vs = VanStockumInterior(omega=omega, R=R)
    F_ana = vs.analytic_exterior_F(sol.r)
    np.testing.assert_allclose(sol.F, F_ana, rtol=1e-14, atol=1e-14)


def test_supercritical_handles_arbitrarily_high_a():
    """The dispatcher works at a = 5, where any naive integrator fails."""
    omega, R = 5.0, 1.0
    sol = integrate_lp_robust(omega_dust=omega, R=R, r_max=10.0, n_samples=2001)
    assert sol.regime == "supercritical"
    # At a = 5, alpha = sqrt(99), so log-period 2pi/alpha is short -> many zeros in [1, 10]
    assert len(sol.F_zeros) >= 4
    # F-zeros are log-uniformly spaced
    log_spacings = np.diff(np.log(sol.F_zeros / R))
    expected = np.pi / np.sqrt(4.0 * (omega * R) ** 2 - 1.0)
    np.testing.assert_allclose(log_spacings, expected, rtol=1e-14)


def test_supercritical_F_zeros_match_analytic_formula():
    """F-zeros from the dispatcher equal r_n = R * exp((n pi - gamma) / alpha)."""
    omega, R = 1.5, 1.0
    sol = integrate_lp_robust(omega_dust=omega, R=R, r_max=100.0, n_samples=2001)
    vs = VanStockumInterior(omega=omega, R=R)
    alpha = vs.alpha
    gamma = np.pi - np.arctan(alpha)
    expected = []
    n = 1
    while True:
        u_n = (n * np.pi - gamma) / alpha
        r_n = R * np.exp(u_n)
        if r_n > 100.0 or u_n <= 0:
            if u_n <= 0:
                n += 1
                continue
            break
        expected.append(r_n)
        n += 1
    np.testing.assert_allclose(sol.F_zeros, expected, rtol=1e-14)


def test_supercritical_F_K_L_consistent():
    """F L + K^2 = r^2 holds on the supercritical analytic grid (machine precision)."""
    omega, R = 1.2, 1.0
    sol = integrate_lp_robust(omega_dust=omega, R=R, r_max=20.0, n_samples=501)
    np.testing.assert_allclose(sol.F * sol.L + sol.K ** 2, sol.r ** 2, rtol=1e-12, atol=1e-12)


def test_uniform_interface_methods():
    """F_at and L_at interpolation work for any regime."""
    sol = integrate_lp_robust(omega_dust=1.0, R=1.0, r_max=10.0, n_samples=501)
    r_test = 3.0
    F_val = sol.F_at(r_test)
    L_val = sol.L_at(r_test)
    assert np.isfinite(F_val)
    assert np.isfinite(L_val)
