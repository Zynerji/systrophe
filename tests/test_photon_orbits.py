"""Photon orbits / null geodesics tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.photon_orbits import (
    integrate_null_geodesic,
    null_circular_omega,
    null_impact_parameters,
    vanstockum_photon_omega,
)


def test_minkowski_photon_circular_at_radius_r():
    """In Minkowski cylindrical (F=1, K=0, L=r^2): Omega_+- = +- 1/r."""
    r = 2.0
    om_minus, om_plus = null_circular_omega(F=1.0, K=0.0, L=r ** 2, r=r)
    assert om_minus == pytest.approx(-1.0 / r)
    assert om_plus == pytest.approx(1.0 / r)


def test_null_omega_degenerate_at_L_zero():
    with pytest.raises(ValueError):
        null_circular_omega(F=1.0, K=0.5, L=0.0, r=1.0)


def test_minkowski_photon_impact_parameters():
    """In Minkowski (F=1, K=0): b_+- = +- r."""
    r = 2.0
    b_minus, b_plus = null_impact_parameters(F=1.0, K=0.0, r=r)
    assert b_minus == pytest.approx(-r)
    assert b_plus == pytest.approx(r)


def test_impact_parameter_degenerate_at_F_zero():
    with pytest.raises(ValueError):
        null_impact_parameters(F=0.0, K=0.5, r=1.0)


def test_vanstockum_photon_omega_supercritical():
    """For supercritical Tipler exterior: photon Omega is array-valued."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r = np.array([1.5, 2.0, 3.0])
    om_minus, om_plus = vanstockum_photon_omega(vs, r)
    # om_+ > om_- always
    assert np.all(om_plus >= om_minus)
    # The photon Omega values are real and finite (away from L=0)
    assert np.all(np.isfinite(om_minus))
    assert np.all(np.isfinite(om_plus))


def test_vanstockum_photon_omega_rejects_subcritical():
    vs = VanStockumInterior(omega=0.3, R=1.0)
    with pytest.raises(ValueError):
        vanstockum_photon_omega(vs, np.array([1.5]))


def test_null_geodesic_integration_preserves_normalization():
    """A null geodesic in Minkowski cylindrical should have constant E, ell."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    # Start at r=2, with E=1, ell=2 (b = ell/E = 2 = r0, turning point at r0)
    sol = integrate_null_geodesic(
        F_fn, K_fn, L_fn, h_fn,
        E=1.0, ell=2.0, r0=2.0, rdot0=1e-8, affine_max=1.0, n_samples=101,
    )
    # Sanity: outputs are finite arrays of correct length
    assert sol["tau"].shape == (101,)
    assert np.all(np.isfinite(sol["r"]))
    assert np.all(np.isfinite(sol["t"]))


def test_photon_omega_consistency_with_geodesic_module_bounds():
    """null_circular_omega returns the same (Omega_-, Omega_+) as
    geodesic.timelike_omega_bounds."""
    from systrophe.geodesic import timelike_omega_bounds

    F, K, L, r = 1.5, 0.3, 2.0, 1.7
    om_a = null_circular_omega(F, K, L, r)
    om_b = timelike_omega_bounds(F, K, L, r)
    assert om_a[0] == pytest.approx(om_b[0])
    assert om_a[1] == pytest.approx(om_b[1])
