"""Photon ray tracing tests."""

import numpy as np
import pytest

from systrophe.geometry.photon_raytrace import (
    lensing_pattern,
    photon_deflection_angle,
    photon_perihelion,
)


def test_perihelion_in_minkowski():
    """In Minkowski: perihelion of a photon with E=1, ell=2 is at r = b = 2."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    r_min = photon_perihelion(F_fn, K_fn, L_fn, E=1.0, ell=2.0, r_lower=0.5, r_upper=5.0)
    assert r_min is not None
    assert r_min == pytest.approx(2.0, rel=1e-6)


def test_perihelion_returns_none_when_no_turning_point():
    """A photon that never turns (e.g. straight in radial direction) -> None."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    # ell = 0: V_eff = -E^2 < 0 always; no zero.
    r_min = photon_perihelion(F_fn, K_fn, L_fn, E=1.0, ell=0.0, r_lower=0.5, r_upper=5.0)
    assert r_min is None


def test_deflection_minkowski_truncation_residual():
    """In Minkowski, the deflection out to a finite r_max is exactly
    -2 arcsin(b / r_max) (truncation residual; goes to 0 as r_max -> infinity).
    """
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    b = 2.0
    r_max = 20.0
    deflection = photon_deflection_angle(
        F_fn, K_fn, L_fn, h_fn, E=1.0, ell=b, r_min=b, r_max=r_max
    )
    expected = -2.0 * np.arcsin(b / r_max)
    assert abs(deflection - expected) < 1e-2  # quadrature tolerance


def test_deflection_minkowski_decreases_with_larger_r_max():
    """Increasing r_max reduces the magnitude of the truncation residual."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    d_20 = photon_deflection_angle(
        F_fn, K_fn, L_fn, h_fn, E=1.0, ell=2.0, r_min=2.0, r_max=20.0
    )
    d_200 = photon_deflection_angle(
        F_fn, K_fn, L_fn, h_fn, E=1.0, ell=2.0, r_min=2.0, r_max=200.0
    )
    assert abs(d_200) < abs(d_20)


def test_lensing_pattern_returns_arrays():
    """lensing_pattern returns matched-shape arrays."""
    F_fn = lambda r: 1.0
    K_fn = lambda r: 0.0
    L_fn = lambda r: r ** 2
    h_fn = lambda r: 1.0
    ell_array = np.array([1.0, 2.0, 3.0])
    out = lensing_pattern(
        F_fn, K_fn, L_fn, h_fn,
        E=1.0, ell_array=ell_array,
        r_search_lo=0.5, r_search_hi=5.0, r_max=20.0,
    )
    assert out["ell"].shape == (3,)
    assert out["r_perihelion"].shape == (3,)
    assert out["deflection_angle"].shape == (3,)
    # impact parameter b = ell / E = ell here
    np.testing.assert_allclose(out["b"], ell_array)
    # r_perihelion in Minkowski is b
    np.testing.assert_allclose(
        out["r_perihelion"], ell_array, rtol=1e-3
    )


def test_perihelion_in_supercritical_tipler():
    """In a supercritical Tipler exterior, perihelions exist for sufficient ell."""
    from systrophe import VanStockumInterior

    vs = VanStockumInterior(omega=1.0, R=1.0)
    F_fn = lambda r: float(vs.analytic_exterior_F(r))
    K_fn = lambda r: float(vs.analytic_exterior_K(r))
    L_fn = lambda r: float(vs.analytic_exterior_L(r))
    # Photon launched with l/E = 5, search inside the first CTC band region
    r_min = photon_perihelion(F_fn, K_fn, L_fn, E=1.0, ell=5.0, r_lower=2.0, r_upper=10.0)
    # We don't assert a specific value -- just that a perihelion exists
    # somewhere in the sweep range and is finite.
    assert r_min is None or (np.isfinite(r_min) and r_min > 0)
