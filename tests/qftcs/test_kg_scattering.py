"""Tests for KG scattering on LP background."""

import math

import pytest

from systrophe.qftcs.kg_scattering import (
    bound_states_between_CHs,
    cascade_transmission_total,
    effective_potential,
    reflection_coefficient,
    superradiance_test,
    transmission_at_multiple_CH,
    transmission_coefficient,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_effective_potential_finite(vs_super):
    V = effective_potential(vs_super, r=1.5, omega=1.0)
    assert math.isfinite(V)


def test_effective_potential_mass_dependent(vs_super):
    V0 = effective_potential(vs_super, r=1.5, omega=1.0, mass=0.0)
    V1 = effective_potential(vs_super, r=1.5, omega=1.0, mass=1.0)
    # V should differ when mass is varied
    assert V0 != V1


def test_transmission_coefficient_in_unit_interval(vs_super):
    T = transmission_coefficient(vs_super, r_left=1.5, r_right=2.5, omega=1.0)
    assert 0 <= T <= 1


def test_transmission_invalid_range_raises(vs_super):
    with pytest.raises(ValueError):
        transmission_coefficient(vs_super, r_left=2.5, r_right=1.5, omega=1.0)


def test_reflection_coefficient_complements_transmission(vs_super):
    T = transmission_coefficient(vs_super, r_left=1.5, r_right=2.5, omega=1.0)
    R = reflection_coefficient(vs_super, r_left=1.5, r_right=2.5, omega=1.0)
    assert abs(T + R - 1.0) < 1e-9


def test_transmission_at_multiple_CH_returns_list(vs_super):
    res = transmission_at_multiple_CH(vs_super, omega=1.0, n_horizons=2)
    assert isinstance(res, list)


def test_transmission_at_multiple_CH_subcritical_empty(vs_sub):
    res = transmission_at_multiple_CH(vs_sub, omega=1.0)
    assert res == []


def test_transmission_cumulative_decreasing(vs_super):
    res = transmission_at_multiple_CH(vs_super, omega=1.0, n_horizons=3)
    if len(res) > 1:
        Ts = [r["cumulative_T"] for r in res]
        for t1, t2 in zip(Ts[:-1], Ts[1:]):
            assert t2 <= t1 + 1e-12


def test_bound_states_returns_list(vs_super):
    omegas = bound_states_between_CHs(vs_super, r_left=2.0, r_right=10.0)
    assert isinstance(omegas, list)


def test_superradiance_test_returns_dict(vs_super):
    res = superradiance_test(vs_super, r_inside_CH=3.35, omega=0.1, n_phi=1)
    assert "is_superradiant" in res
    assert "Omega_H" in res


def test_superradiance_high_omega_not_superradiant(vs_super):
    """For omega far above threshold, should not be superradiant."""
    res = superradiance_test(vs_super, r_inside_CH=3.35, omega=1e6, n_phi=1)
    assert res["is_superradiant"] is False


def test_cascade_transmission_total_in_unit_interval(vs_super):
    T = cascade_transmission_total(vs_super, omega=1.0, n_horizons=2)
    assert 0 <= T <= 1


def test_cascade_transmission_subcritical_unity(vs_sub):
    T = cascade_transmission_total(vs_sub, omega=1.0)
    assert T == 1.0
