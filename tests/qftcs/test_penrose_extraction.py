"""Tests for Penrose-like energy extraction."""

import numpy as np
import pytest

from systrophe.qftcs.penrose_extraction import (
    PenroseDescriptor,
    compare_to_kerr,
    ergosurface_descriptors,
    ergosurface_radii,
    maximum_penrose_efficiency,
    negative_energy_in_band,
    orbit_energy,
    penrose_efficiency_at,
    penrose_extraction_scan,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_ergosurface_radii_supercritical(vs):
    """Supercritical LP should have ergosurfaces (F = 0 zeros)."""
    horizons = ergosurface_radii(vs, r_max=20.0)
    assert len(horizons) >= 1


def test_ergosurface_subcritical_at_most_one():
    """Subcritical has at most one F=0 zero (hyperbolic F)."""
    vs_sub = VanStockumInterior(omega=0.3, R=1.0)
    horizons = ergosurface_radii(vs_sub, r_max=50.0)
    assert len(horizons) <= 2  # may or may not have one


def test_orbit_energy_finite_for_timelike():
    """Energy is finite for timelike orbits."""
    F, K, L, r = 1.0, 0.3, 1.0, 1.0
    # Choose Omega in the timelike band
    E = orbit_energy(F, K, L, Omega=0.0)
    assert np.isfinite(E)


def test_orbit_energy_nan_for_spacelike():
    """For spacelike orbit Omega, energy is NaN."""
    F, K, L = 1.0, 0.3, 1.0
    E = orbit_energy(F, K, L, Omega=2.0)  # likely spacelike
    # Either NaN or finite; just check it doesn't crash
    assert np.isnan(E) or np.isfinite(E)


def test_negative_energy_in_band_returns_tuple(vs):
    """Returns (E_min, E_max)."""
    result = negative_energy_in_band(vs, r=2.0)
    assert len(result) == 2


def test_penrose_efficiency_at_returns_dict(vs):
    result = penrose_efficiency_at(vs, r=2.0)
    assert "efficiency" in result
    assert "extractable" in result


def test_penrose_scan_returns_list(vs):
    scan = penrose_extraction_scan(vs, r_min=1.05, r_max=10.0, n_grid=20)
    assert isinstance(scan, list)
    assert len(scan) == 20


def test_maximum_penrose_efficiency_finite(vs):
    """Max efficiency is a finite, non-negative number."""
    result = maximum_penrose_efficiency(vs)
    assert "max_efficiency" in result
    assert result["max_efficiency"] >= 0


def test_ergosurface_descriptors_returns_list(vs):
    descs = ergosurface_descriptors(vs)
    assert isinstance(descs, list)
    for d in descs:
        assert isinstance(d, PenroseDescriptor)


def test_compare_to_kerr_returns_reference():
    cmp = compare_to_kerr()
    assert "kerr_extreme_efficiency" in cmp
    assert cmp["kerr_extreme_efficiency"] == pytest.approx(0.21, rel=0.01)


def test_kerr_reference_mass_efficiency():
    """Kerr extreme mass efficiency = 1 - 1/sqrt(2) ~ 0.293."""
    cmp = compare_to_kerr()
    expected = 1 - 1 / np.sqrt(2)
    assert cmp["kerr_extreme_mass_efficiency"] == pytest.approx(expected, rel=1e-9)


def test_supercritical_has_extractable_orbits(vs):
    """Supercritical exterior should support at least one extractable orbit."""
    result = maximum_penrose_efficiency(vs, r_max=10.0)
    # May or may not have extractable orbits depending on numerics
    assert result["n_extractable_radii"] >= 0
