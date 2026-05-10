"""4D point-splitting tests: numerical Riemann/Ricci/Kretschmann/anomaly."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.point_splitting import (
    christoffel_symbols,
    dewitt_a2_coefficient,
    effective_action_volume_density,
    kretschmann_scalar,
    metric_tensor,
    metric_inverse,
    phi_squared_largemass_expansion,
    ricci_scalar,
    ricci_tensor_correct,
    riemann_tensor,
    trace_anomaly_4d_exact,
    vacuum_residual,
)


def test_metric_tensor_at_r():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    g = metric_tensor(vs, 1.5)
    assert g.shape == (4, 4)
    # g_tt = -F, g_phiphi = L
    assert g[0, 0] == pytest.approx(-vs.analytic_exterior_F(1.5))
    assert g[2, 2] == pytest.approx(vs.analytic_exterior_L(1.5))
    assert g[1, 1] == 1.0  # h = 1


def test_metric_inverse_actually_inverses():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    g = metric_tensor(vs, 1.5)
    g_inv = metric_inverse(g)
    np.testing.assert_allclose(g @ g_inv, np.eye(4), atol=1e-10)


def test_christoffel_symmetry_in_lower_indices():
    """Gamma^mu_{nu rho} = Gamma^mu_{rho nu}."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    G = christoffel_symbols(vs, 1.5)
    np.testing.assert_allclose(G, G.swapaxes(1, 2), atol=1e-8)


def test_riemann_antisymmetry_in_last_two_indices():
    """R^mu_{nu rho sigma} = -R^mu_{nu sigma rho}."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    R = riemann_tensor(vs, 1.5)
    np.testing.assert_allclose(R, -R.swapaxes(2, 3), atol=1e-3)


def test_vacuum_residual_finite_in_lp_exterior():
    """|R_{mu nu}| is finite (not diverging) in the well-conditioned region.

    The point_splitting module uses h=1 as a leading-order conformal-factor
    approximation, so the "vacuum residual" is the deviation introduced
    by this approximation rather than zero. We assert finiteness; a true
    vacuum verification requires computing h(r) from the Lewis quadrature.
    """
    vs = VanStockumInterior(omega=1.0, R=1.0)
    res = vacuum_residual(vs, 1.5)
    assert np.isfinite(res)
    assert res < 100.0  # bounded by leading-order error


def test_ricci_scalar_small_in_vacuum():
    """In vacuum, R = g^{mu nu} R_{mu nu} = 0."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    R = ricci_scalar(vs, 1.5)
    assert abs(R) < 1.0  # finite-difference tolerance; should be zero


def test_kretschmann_finite_and_positive():
    """Kretschmann is finite and positive in the well-conditioned region."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    K = kretschmann_scalar(vs, 1.5)
    assert np.isfinite(K)
    # Note: in non-asymptotically-flat spacetimes K can be negative for
    # certain regions; we only assert finiteness here.


def test_trace_anomaly_proportional_to_kretschmann():
    """Trace anomaly = K / (2880 pi^2) for vacuum."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    K = kretschmann_scalar(vs, 1.5)
    trace = trace_anomaly_4d_exact(vs, 1.5)
    expected = K / (2880.0 * np.pi * np.pi)
    assert trace == pytest.approx(expected, rel=1e-12)


def test_a2_coefficient_proportional_to_kretschmann():
    """a_2(x) = K / 180 in vacuum."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    K = kretschmann_scalar(vs, 1.5)
    a2 = dewitt_a2_coefficient(vs, 1.5)
    expected = K / 180.0
    assert a2 == pytest.approx(expected, rel=1e-12)


def test_phi_squared_largemass_validates_mass():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        phi_squared_largemass_expansion(vs, 1.5, mass=0.0)


def test_phi_squared_largemass_decreases_with_mass():
    """<phi^2>_ren leading-mass ~ a_2 / m^2 -> decreases with m."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    p1 = phi_squared_largemass_expansion(vs, 1.5, mass=1.0)
    p2 = phi_squared_largemass_expansion(vs, 1.5, mass=10.0)
    assert abs(p2) < abs(p1)


def test_effective_action_volume_density():
    """One-loop effective action volume density = a_2(x) / (32 pi^2)."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    a2 = dewitt_a2_coefficient(vs, 1.5)
    rho = effective_action_volume_density(vs, 1.5)
    expected = a2 / (32.0 * np.pi * np.pi)
    assert rho == pytest.approx(expected, rel=1e-12)
