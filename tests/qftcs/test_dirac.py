"""Dirac field on the Lewis-Papapetrou background tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.qftcs.dirac import (
    LewisPapapetrouTetrad,
    gamma_matrix,
    radial_dirac_system,
    solve_radial_dirac,
    vanstockum_dirac_system,
)


# --- Gamma matrices: Clifford algebra ---


def test_gamma_clifford_algebra():
    """{gamma^a, gamma^b} = 2 eta^{ab} I (Weyl representation)."""
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    for a in range(4):
        for b in range(4):
            ga = gamma_matrix(a)
            gb = gamma_matrix(b)
            anticomm = ga @ gb + gb @ ga
            expected = 2.0 * eta[a, b] * np.eye(4)
            np.testing.assert_allclose(anticomm, expected, atol=1e-12)


def test_gamma_indices_validated():
    with pytest.raises(ValueError):
        gamma_matrix(4)
    with pytest.raises(ValueError):
        gamma_matrix(-1)


# --- Tetrad reproduces the Lewis-Papapetrou metric ---


def test_tetrad_reproduces_minkowski_cylindrical():
    """For F=1, K=0, L=r^2, h=1: tetrad reproduces flat cylindrical Minkowski."""
    r = 2.0
    t = LewisPapapetrouTetrad(F=1.0, K=0.0, L=r ** 2, h=1.0)
    g = t.reproduces_metric()
    expected = np.diag([-1.0, 1.0, r ** 2, 1.0])
    np.testing.assert_allclose(g, expected, atol=1e-12)


def test_tetrad_reproduces_van_stockum_interior():
    """For van Stockum interior at r=R, tetrad reproduces interior metric."""
    omega, R = 0.7, 1.0
    vs = VanStockumInterior(omega=omega, R=R)
    g_interior = vs.metric(R)
    F = -g_interior[0, 0]
    K = g_interior[0, 2]
    L = g_interior[2, 2]
    h = g_interior[1, 1]
    t = LewisPapapetrouTetrad(F=F, K=K, L=L, h=h)
    g_reproduced = t.reproduces_metric()
    expected = np.array([
        [-F, 0.0, K, 0.0],
        [0.0, h, 0.0, 0.0],
        [K, 0.0, L, 0.0],
        [0.0, 0.0, 0.0, h],
    ])
    np.testing.assert_allclose(g_reproduced, expected, atol=1e-12)


def test_tetrad_orthonormality():
    """The tetrad is orthonormal: e^a e^b = eta^{ab} (with metric raising)."""
    t = LewisPapapetrouTetrad(F=1.5, K=0.3, L=2.0, h=1.0)
    e = t.matrix()
    g = t.reproduces_metric()
    g_inv = np.linalg.inv(g)
    # e^a_mu g^{mu nu} e^b_nu should equal eta^{ab}
    eta_recovered = e @ g_inv @ e.T
    eta_expected = np.diag([-1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(eta_recovered, eta_expected, atol=1e-10)


# --- Radial Dirac system runs and produces finite output ---


def test_radial_dirac_system_returns_callable():
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    rhs = radial_dirac_system(F_fn, K_fn, L_fn, h_fn, E=1.0, m=0, k=0, mass=0.5)
    assert callable(rhs)
    result = rhs(2.0, np.array([1.0 + 0j, 0.5 + 0.2j]))
    assert result.shape == (2,)
    assert np.all(np.isfinite(result.real)) and np.all(np.isfinite(result.imag))


def test_solve_radial_dirac_runs_minkowski():
    """Integrate the radial Dirac in flat cylindrical Minkowski."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    sol = solve_radial_dirac(
        F_fn, K_fn, L_fn, h_fn,
        E=1.0, m=0, k=0, mass=0.5,
        r0=1.0, R1_0=1.0 + 0j, R2_0=0.0 + 0j,
        r_max=2.0, n_samples=51,
    )
    assert sol["r"].shape == (51,)
    assert np.all(np.isfinite(sol["R1"].real))
    assert np.all(np.isfinite(sol["R1"].imag))
    assert np.all(np.isfinite(sol["R2"].real))


def test_vanstockum_dirac_system_supercritical():
    """vanstockum_dirac_system requires supercritical input and returns a callable."""
    vs_super = VanStockumInterior(omega=1.0, R=1.0)
    rhs = vanstockum_dirac_system(vs_super, E=1.0, m=0, k=0, mass=0.5)
    assert callable(rhs)
    result = rhs(1.5, np.array([1.0 + 0j, 0.0 + 0j]))
    assert result.shape == (2,)
    assert np.all(np.isfinite(result.real))


def test_vanstockum_dirac_system_rejects_subcritical():
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        vanstockum_dirac_system(vs_sub, E=1.0, m=0, k=0, mass=0.5)
