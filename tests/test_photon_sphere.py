"""Tests for photon sphere module on LP background."""

import numpy as np
import pytest

from systrophe.photon_sphere import (
    PhotonSphere,
    bare_lp_has_photon_sphere,
    effective_b_hybrid,
    engineer_photon_sphere_via_mass,
    hybrid_photon_sphere_omega,
    hybrid_photon_sphere_radii,
    hybrid_photon_sphere_stability,
    impact_parameter_bare,
    photon_sphere_descriptor,
    photon_sphere_shadow_radius,
    schwarzschild_photon_sphere_radius,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- Bare LP: no-go result -------------------------------------------

def test_bare_LP_has_no_photon_sphere_supercritical(vs):
    assert not bare_lp_has_photon_sphere(vs)


def test_bare_LP_has_no_photon_sphere_subcritical():
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)
    assert not bare_lp_has_photon_sphere(vs_sub)


def test_impact_parameter_bare_finite(vs):
    b = impact_parameter_bare(vs, r=2.0, branch="prograde")
    assert np.isfinite(b)


def test_impact_parameter_bare_branches_differ(vs):
    b_p = impact_parameter_bare(vs, r=2.0, branch="prograde")
    b_r = impact_parameter_bare(vs, r=2.0, branch="retrograde")
    assert abs(b_p - b_r) > 1e-6


def test_impact_parameter_bare_invalid_branch(vs):
    with pytest.raises(ValueError):
        impact_parameter_bare(vs, r=2.0, branch="diagonal")


# ----- Hybrid Schwarzschild-cylinder ----------------------------------

def test_effective_b_hybrid_finite_outside_horizon(vs):
    """Hybrid b is finite for r > 2M."""
    M = 0.5
    b = effective_b_hybrid(vs, r=3.0, M=M)
    assert np.isfinite(b)


def test_effective_b_hybrid_nan_inside_horizon(vs):
    """Hybrid b is NaN for r <= 2M."""
    M = 1.0
    b = effective_b_hybrid(vs, r=1.5, M=M)
    assert np.isnan(b)


def test_hybrid_photon_sphere_appears(vs):
    """Adding Schwarzschild mass creates a photon sphere."""
    M = 0.5
    radii = hybrid_photon_sphere_radii(vs, M=M, r_min=1.05, r_max=15.0)
    # For Schwarzschild alone at this M, photon sphere is at r=3M = 1.5
    # The cylinder distorts this; expect at least one
    assert len(radii) >= 1


def test_hybrid_photon_sphere_at_zero_omega_recovers_schwarzschild():
    """At omega -> 0 (very subcritical), bare cylinder vanishes;
    photon sphere should approach Schwarzschild's r = 3M.

    We use omega = 0.01 (deep subcritical) as an approximation.
    """
    vs_weak = VanStockumInterior(omega=0.01, R=1.0)
    M = 1.0
    radii = hybrid_photon_sphere_radii(vs_weak, M=M, r_min=2.5, r_max=10.0)
    if radii:
        # Should be near r = 3M = 3.0 (modulo cylinder perturbation)
        closest = min(radii, key=lambda r: abs(r - 3.0))
        assert abs(closest - 3.0) < 1.0


def test_hybrid_photon_sphere_stability_returns_bool(vs):
    """Hybrid stability classifier returns bool."""
    M = 0.5
    radii = hybrid_photon_sphere_radii(vs, M=M)
    if not radii:
        pytest.skip("No hybrid photon spheres found")
    stable = hybrid_photon_sphere_stability(vs, radii[0], M=M)
    assert isinstance(stable, bool)


def test_hybrid_photon_sphere_omega_finite(vs):
    M = 0.5
    radii = hybrid_photon_sphere_radii(vs, M=M)
    if not radii:
        pytest.skip("No hybrid photon spheres found")
    omega = hybrid_photon_sphere_omega(vs, radii[0], M=M)
    assert np.isfinite(omega)


# ----- Engineer via mass ----------------------------------------------

def test_engineer_photon_sphere_via_mass_returns_M_or_none(vs):
    """Function returns M in range or None."""
    M = engineer_photon_sphere_via_mass(vs, r_target=2.0, M_range=(0.01, 3.0))
    if M is not None:
        assert 0.01 <= M <= 3.0


def test_engineered_M_actually_creates_sphere(vs):
    """If we find an M, the resulting hybrid actually has a photon sphere there."""
    r_target = 2.5
    M = engineer_photon_sphere_via_mass(vs, r_target=r_target, M_range=(0.05, 2.0))
    if M is None:
        pytest.skip("No engineered solution found in range")
    # Compute hybrid db/dr at r_target with this M; should be near 0
    from systrophe.photon_sphere import _db_dr_hybrid
    d = _db_dr_hybrid(vs, r_target, M)
    assert abs(d) < 0.1


# ----- Descriptor + shadow --------------------------------------------

def test_photon_sphere_descriptor_complete(vs):
    """PhotonSphere descriptor at a hybrid photon sphere has all fields."""
    M = 0.5
    radii = hybrid_photon_sphere_radii(vs, M=M)
    if not radii:
        pytest.skip("No hybrid photon spheres found")
    ph = photon_sphere_descriptor(vs, radii[0], M=M)
    assert isinstance(ph, PhotonSphere)
    assert ph.schwarzschild_mass == M
    assert ph.branch in ("prograde", "retrograde")


def test_photon_sphere_shadow_at_infinity(vs):
    M = 0.5
    radii = hybrid_photon_sphere_radii(vs, M=M)
    if not radii:
        pytest.skip("No hybrid photon spheres found")
    ph = photon_sphere_descriptor(vs, radii[0], M=M)
    shadow = photon_sphere_shadow_radius(ph, observer_r=float("inf"))
    assert shadow == pytest.approx(abs(ph.impact_parameter), rel=1e-12)


def test_schwarzschild_photon_sphere_radius():
    """r_ph = 3M for pure Schwarzschild."""
    assert schwarzschild_photon_sphere_radius(M=1.0) == 3.0
    assert schwarzschild_photon_sphere_radius(M=2.5) == 7.5
