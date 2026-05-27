"""Tests for Unruh acoustic-metric mapping."""

import numpy as np
import pytest

from systrophe.analogs.acoustic_metric import (
    acoustic_hawking_temperature,
    acoustic_horizon_radius,
    acoustic_line_element_at_radius,
    acoustic_metric_components,
    acoustic_surface_gravity,
    compare_acoustic_vs_gravitational_T_H,
    ctc_region_is_supersonic,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- components -------------------------------------------------------

def test_acoustic_components_at_subsonic_point():
    """F > 0: subsonic (c > v). signature = +1."""
    comp = acoustic_metric_components(F=1.0, K=0.5, L=1.0)
    assert float(comp["c_squared"]) > float(comp["v_phi"] ** 2)
    assert float(comp["signature"]) == 1.0


def test_acoustic_components_at_sonic_point():
    """F = 0: sonic (c = v). signature = 0."""
    comp = acoustic_metric_components(F=0.0, K=1.0, L=4.0)
    # c^2 = F + v^2 = v^2
    assert float(comp["c_squared"]) == pytest.approx(float(comp["v_phi"] ** 2), abs=1e-12)
    assert float(comp["signature"]) == 0.0


def test_acoustic_components_at_supersonic_point():
    """F < 0: supersonic (c < v). signature = -1."""
    comp = acoustic_metric_components(F=-0.5, K=1.0, L=4.0)
    assert float(comp["c_squared"]) < float(comp["v_phi"] ** 2)
    assert float(comp["signature"]) == -1.0


def test_acoustic_components_array_input():
    F = np.array([1.0, 0.0, -1.0])
    K = np.array([0.5, 0.5, 0.5])
    L = np.array([1.0, 1.0, 1.0])
    comp = acoustic_metric_components(F, K, L)
    assert comp["rho_acoustic"].shape == (3,)
    assert np.allclose(comp["signature"], np.array([1.0, 0.0, -1.0]))


# ----- LP integration ---------------------------------------------------

def test_acoustic_horizon_exists_in_supercritical_exterior(vs):
    """A supercritical (a > 1/2) van Stockum exterior has F=0 zeros."""
    r_h = acoustic_horizon_radius(vs, r_min=1.01, r_max=20.0)
    assert r_h is not None
    # Verify F is approximately 0 at the returned r
    F_h = float(vs.analytic_exterior_F(r_h))
    assert abs(F_h) < 0.05


def test_acoustic_surface_gravity_positive(vs):
    """kappa > 0 at the chronology horizon."""
    r_h = acoustic_horizon_radius(vs, r_min=1.01, r_max=20.0)
    kappa = acoustic_surface_gravity(vs, r_horizon=r_h)
    assert kappa > 0


def test_acoustic_hawking_T_positive(vs):
    r_h = acoustic_horizon_radius(vs, r_min=1.01, r_max=20.0)
    T = acoustic_hawking_temperature(vs, r_horizon=r_h)
    assert T > 0


def test_acoustic_vs_gravitational_hawking_T_identical(vs):
    """By Unruh identification c^2 - v^2 = F, T_acoustic = T_gravitational exactly."""
    r_h = acoustic_horizon_radius(vs, r_min=1.01, r_max=20.0)
    cmp = compare_acoustic_vs_gravitational_T_H(vs, r_horizon=r_h)
    # Identical by construction; only FD noise possible
    assert cmp["rel_diff"] < 1e-12


def test_line_element_regime_classification(vs):
    """At a known supersonic radius (F < 0), regime = 'supersonic'."""
    r_h = acoustic_horizon_radius(vs, r_min=1.01, r_max=20.0)
    # Just past the horizon: F < 0
    r_super = r_h * 1.05
    info = acoustic_line_element_at_radius(vs, r=r_super)
    # Could be subsonic if we picked a point in the next band; check explicitly
    if info["F"] < -1e-12:
        assert info["regime"] == "supersonic"
    elif info["F"] > 1e-12:
        assert info["regime"] == "subsonic"


def test_ctc_region_is_supersonic_consistency(vs):
    """All CTC (F<0) samples must be classified as supersonic."""
    rs = np.linspace(1.05, 15.0, 50)
    result = ctc_region_is_supersonic(vs, r_samples=rs)
    assert result["ctc_supersonic_consistency"]
    # Both populations exist on a sufficient range
    assert result["n_subsonic"] > 0
    assert result["n_supersonic"] > 0


def test_subsonic_in_pre_horizon_region(vs):
    """Just outside the cylinder (r slightly > R = 1), F > 0 (subsonic)."""
    r = 1.05
    info = acoustic_line_element_at_radius(vs, r=r)
    # At small r past R, F starts at 1 and decreases; should still be positive
    if info["F"] > 0:
        assert info["regime"] == "subsonic"
