"""Casimir / Z3 cover tests."""

import numpy as np
import pytest

from systrophe.qftcs.casimir import (
    hurwitz_zeta_neg3,
    standard_casimir_energy_density,
    standard_casimir_force,
    topological_casimir_coefficient,
    topological_casimir_derivative,
    z3_cover_fundamental_eigenvalue,
    z3_cover_mode_density,
    z3_cover_regularised_zeta_sum,
)


# ----- Hurwitz zeta -------------------------------------------------------

def test_hurwitz_zeta_at_simple_values():
    """zeta_H(-3, 1) = -1/120 (standard identity)."""
    # zeta(-3) = 1/120 = -B_4 / 4 evaluated at the natural normalization
    # Actually zeta_H(-3, 1) = zeta_Riemann(-3) = 1/120.
    # Our closed form: -1/4 + 1/2 - 1/4 + 1/120 = 1/120. Check.
    val = float(hurwitz_zeta_neg3(1.0))
    assert val == pytest.approx(1 / 120, rel=1e-12)


def test_hurwitz_zeta_at_zero():
    """zeta_H(-3, 0) = 0 - 0 - 0 + 1/120 = 1/120."""
    val = float(hurwitz_zeta_neg3(0.0))
    assert val == pytest.approx(1 / 120, rel=1e-12)


def test_hurwitz_zeta_matches_bernoulli_polynomial_form():
    """zeta_H(-3, a) = -B_4(a) / 4 where B_4 = a^4 - 2 a^3 + a^2 - 1/30."""
    a_values = np.array([0.0, 1 / 3, 1 / 2, 2 / 3, 1.0, 1.5, 2.0])
    for a in a_values:
        B4 = a ** 4 - 2 * a ** 3 + a ** 2 - 1 / 30
        expected = -B4 / 4
        got = float(hurwitz_zeta_neg3(a))
        assert got == pytest.approx(expected, abs=1e-15)


def test_hurwitz_zeta_array_input():
    a = np.linspace(0, 1, 5)
    result = hurwitz_zeta_neg3(a)
    assert result.shape == (5,)


# ----- Standard Casimir ---------------------------------------------------

def test_casimir_density_negative():
    rho = float(standard_casimir_energy_density(1.0))
    assert rho < 0


def test_casimir_density_d4_scaling():
    """rho(2d) / rho(d) = 1/16 (1/d^4 scaling)."""
    r1 = float(standard_casimir_energy_density(1.0))
    r2 = float(standard_casimir_energy_density(2.0))
    assert r2 / r1 == pytest.approx(1 / 16, rel=1e-12)


def test_casimir_density_textbook_constant():
    """At d = 1: rho = -pi^2 / 720 (natural units)."""
    rho = float(standard_casimir_energy_density(1.0))
    expected = -(np.pi ** 2) / 720
    assert rho == pytest.approx(expected, rel=1e-12)


def test_casimir_force_textbook_constant():
    """At d = 1: F/A = -pi^2 / 240."""
    P = float(standard_casimir_force(1.0))
    expected = -(np.pi ** 2) / 240
    assert P == pytest.approx(expected, rel=1e-12)


# ----- Topological Casimir coefficient ------------------------------------

def test_topological_C_finite():
    """C(gamma_eff) is finite for all real gamma_eff."""
    for g in np.linspace(-10, 10, 11):
        c = float(topological_casimir_coefficient(g))
        assert np.isfinite(c)


def test_topological_C_at_zero():
    """C(0) = (1/720) * sum_{b=0,1,2} zeta_H(-3, b/3)
            = (1/720) * (1/120 + zeta_H(-3, 1/3) + zeta_H(-3, 2/3))."""
    z_0 = float(hurwitz_zeta_neg3(0.0))
    z_1_3 = float(hurwitz_zeta_neg3(1 / 3))
    z_2_3 = float(hurwitz_zeta_neg3(2 / 3))
    expected = (z_0 + z_1_3 + z_2_3) / 720
    got = float(topological_casimir_coefficient(0.0))
    assert got == pytest.approx(expected, rel=1e-12)


def test_topological_C_derivative():
    """C'(gamma) computed via central differences."""
    g0 = 0.5
    eps = 1e-6
    Cp = float(topological_casimir_derivative(g0, eps=eps))
    # Analytic check: dC/dgamma = (1/720) sum_b zeta_H'(-3, b/3 + g/(2pi)) / (2pi)
    # where zeta_H' = -d/da (a^4/4 - a^3/2 + a^2/4) = -a^3 + 3a^2/2 - a/2
    expected = 0.0
    for b in (0, 1, 2):
        a = b / 3 + g0 / (2 * np.pi)
        expected += -a ** 3 + 1.5 * a ** 2 - 0.5 * a
    expected /= (720 * 2 * np.pi)
    assert Cp == pytest.approx(expected, rel=1e-4)


# ----- Z3 cover mode density ---------------------------------------------

def test_z3_mode_density_returns_3N_eigenvalues():
    eigs = z3_cover_mode_density(N=12, gamma_eff=0.0)
    assert eigs.shape == (3 * 12,)


def test_z3_mode_density_sorted_and_real():
    eigs = z3_cover_mode_density(N=8, gamma_eff=0.0)
    assert np.all(np.diff(eigs) >= -1e-12)  # sorted (non-decreasing)
    assert np.all(np.isfinite(eigs))


def test_z3_mode_density_validates_N():
    with pytest.raises(ValueError):
        z3_cover_mode_density(N=1, gamma_eff=0.0)


def test_z3_fundamental_eigenvalues_three_branches():
    """For N = 24, gamma_eff = 0, branch 0 gives 2(1 - cos(2 pi / N))."""
    e0, e1, e2 = z3_cover_fundamental_eigenvalue(N=24, gamma_eff=0.0)
    expected_b0 = 2 * (1 - np.cos(2 * np.pi / 24))
    assert e0 == pytest.approx(expected_b0, rel=1e-12)
    # Branches 1 and 2 are complex-conjugate-paired, equal magnitude
    assert e1 == pytest.approx(e2, rel=1e-12)


def test_z3_branches_1_2_lower_than_branch_0():
    """For branch != 0, the lowest non-trivial eigenvalue is smaller than branch 0
    (the 1/3 phase advance per node gives lower fundamental frequency)."""
    e0, e1, e2 = z3_cover_fundamental_eigenvalue(N=12, gamma_eff=0.0)
    assert e1 < e0
    assert e2 < e0


# ----- Zeta-regularised sum -----------------------------------------------

def test_z3_regularised_sum_at_s_neg3():
    """For s = -3, the regularised sum equals 720 * topological_casimir_coefficient."""
    gamma = 0.4
    total = z3_cover_regularised_zeta_sum(-3, gamma_eff=gamma)
    expected = 720 * float(topological_casimir_coefficient(gamma))
    assert total == pytest.approx(expected, rel=1e-9)
