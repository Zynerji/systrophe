"""Tests for DM scalar coupling module."""

import math

import numpy as np
import pytest

from systrophe.dm_scalar_coupling import (
    DM_density_profile_around_cylinder,
    DM_drag_on_orbits,
    DM_induced_CTC_lifetime,
    DM_superradiance_growth_rate,
    compton_wavelength,
    effective_alpha_with_DM_pressure,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_compton_wavelength_inverse_mass():
    l1 = compton_wavelength(1.0)
    l2 = compton_wavelength(2.0)
    assert l2 == pytest.approx(l1 / 2.0, rel=1e-12)


def test_compton_wavelength_zero_mass_infinite():
    assert compton_wavelength(0.0) == float("inf")


def test_DM_density_profile_returns_array(vs):
    r = np.linspace(1.0, 10.0, 20)
    profile = DM_density_profile_around_cylinder(vs, r)
    assert len(profile["rho_DM"]) == 20


def test_DM_density_decreases_with_r(vs):
    """Density should peak at small r and decrease outward."""
    r = np.linspace(0.1, 10.0, 100)
    profile = DM_density_profile_around_cylinder(vs, r)
    assert profile["rho_DM"][0] >= profile["rho_DM"][-1]


def test_DM_drag_returns_dict(vs):
    res = DM_drag_on_orbits(vs, r_orbit=2.0)
    assert "delta_Omega" in res
    assert "relative_shift" in res


def test_DM_drag_relative_shift_tiny_for_ULDM(vs):
    """For ultra-light DM, relative shift should be very small."""
    res = DM_drag_on_orbits(vs, r_orbit=2.0, DM_mass_eV=1e-22, DM_density=1e-30)
    assert res["relative_shift"] < 1.0


def test_DM_superradiance_returns_dict(vs):
    res = DM_superradiance_growth_rate(vs)
    assert "is_superradiant" in res
    assert "growth_rate_estimate" in res


def test_DM_superradiance_growth_rate_non_negative(vs):
    res = DM_superradiance_growth_rate(vs)
    assert res["growth_rate_estimate"] >= 0


def test_effective_alpha_with_DM_pressure_returns_dict(vs):
    res = effective_alpha_with_DM_pressure(vs, DM_pressure=0.0)
    assert "alpha_corrected" in res
    assert "still_supercritical" in res


def test_effective_alpha_zero_pressure_recovers_bare(vs):
    """Zero pressure should recover bare alpha."""
    res = effective_alpha_with_DM_pressure(vs, DM_pressure=0.0)
    if res["still_supercritical"]:
        assert res["alpha_corrected"] == pytest.approx(res["alpha_bare"], rel=1e-9)


def test_effective_alpha_negative_pressure_can_reduce(vs):
    """Sufficiently negative pressure can drop a below 0.5."""
    res = effective_alpha_with_DM_pressure(vs, DM_pressure=-1.5)
    # With strong negative pressure, becomes subcritical
    assert res["still_supercritical"] is False


def test_DM_lifetime_returns_dict(vs):
    res = DM_induced_CTC_lifetime(vs)
    assert "lifetime_estimate" in res
    assert "lifetime_exceeds_Hubble" in res


def test_DM_lifetime_ultra_light_exceeds_Hubble(vs):
    """For ULDM with tiny sigma_xs, lifetime should exceed Hubble time."""
    res = DM_induced_CTC_lifetime(vs, DM_density=1e-30, DM_mass_eV=1e-22)
    assert res["lifetime_exceeds_Hubble"] is True


def test_DM_lifetime_non_negative_rate(vs):
    res = DM_induced_CTC_lifetime(vs)
    assert res["rate_estimate"] >= 0
