"""Tests for 5D Kaluza-Klein embedding module."""

import math

import numpy as np
import pytest

from systrophe.kk_embedding import (
    compactification_radius_from_observations,
    five_d_geodesic_drift_in_xi,
    five_d_metric,
    kk_monopole_charge,
    kk_reduced_alpha,
    kk_tower_masses,
    xi_circulation_evades_CTC,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_five_d_metric_shape(vs_super):
    g5 = five_d_metric(vs_super, r=2.0)
    assert g5.shape == (5, 5)


def test_five_d_metric_xi_xi_positive(vs_super):
    g5 = five_d_metric(vs_super, r=2.0, xi_radius=2.5)
    assert g5[4, 4] == pytest.approx(2.5 ** 2, rel=1e-12)


def test_kk_tower_masses_length(vs_super):
    m = kk_tower_masses(L_xi=1.0, n_max=5)
    assert len(m) == 5


def test_kk_tower_masses_monotone():
    m = kk_tower_masses(L_xi=1.0, n_max=5)
    for m1, m2 in zip(m[:-1], m[1:]):
        assert m2 > m1


def test_kk_tower_masses_invalid_L_xi():
    with pytest.raises(ValueError):
        kk_tower_masses(L_xi=-1.0)


def test_kk_reduced_alpha_increases_with_small_L_xi(vs_super):
    """Small L_xi (large KK contribution) should increase alpha_eff."""
    alpha1 = kk_reduced_alpha(vs_super, L_xi=10.0)
    alpha2 = kk_reduced_alpha(vs_super, L_xi=0.5)
    assert alpha2 > alpha1


def test_kk_reduced_alpha_subcritical_zero(vs_sub):
    alpha = kk_reduced_alpha(vs_sub, L_xi=1.0)
    assert alpha == 0.0


def test_xi_circulation_evades_CTC_returns_dict(vs_super):
    res = xi_circulation_evades_CTC(vs_super, r_in_CTC=3.35, L_xi=1.0)
    assert "evadable" in res
    assert "min_dxi_dphi" in res


def test_xi_circulation_evades_in_CTC_band(vs_super):
    """In CTC band, evadable should be True."""
    res = xi_circulation_evades_CTC(vs_super, r_in_CTC=3.35, L_xi=1.0)
    assert res["regime"] == "CTC"
    assert res["evadable"] is True


def test_xi_circulation_not_evadable_outside_CTC(vs_super):
    """Outside CTC (F > 0), no CTC to evade."""
    res = xi_circulation_evades_CTC(vs_super, r_in_CTC=1.5, L_xi=1.0)
    assert res["regime"] == "non-CTC"
    assert res["evadable"] is False


def test_kk_monopole_charge_returns_dict(vs_super):
    res = kk_monopole_charge(vs_super, L_xi=1.0, n_twist=1)
    assert "magnetic_charge" in res


def test_kk_monopole_charge_scales_with_n(vs_super):
    res1 = kk_monopole_charge(vs_super, L_xi=1.0, n_twist=1)
    res2 = kk_monopole_charge(vs_super, L_xi=1.0, n_twist=3)
    assert res2["magnetic_charge"] == pytest.approx(3 * res1["magnetic_charge"], rel=1e-12)


def test_kk_monopole_invalid_L_xi(vs_super):
    with pytest.raises(ValueError):
        kk_monopole_charge(vs_super, L_xi=-1.0)


def test_compactification_radius_returns_dict():
    res = compactification_radius_from_observations(bound_KK_mass_eV=1e-3)
    assert "L_xi_max_natural_units" in res


def test_compactification_radius_invalid_bound():
    with pytest.raises(ValueError):
        compactification_radius_from_observations(bound_KK_mass_eV=0.0)


def test_five_d_geodesic_drift_returns_dict(vs_super):
    res = five_d_geodesic_drift_in_xi(vs_super, r=3.35, L_xi=1.0, dxi_dphi=0.1)
    assert "drift_per_revolution" in res
    assert math.isfinite(res["drift_per_revolution"])


def test_five_d_geodesic_drift_proportional_to_velocity(vs_super):
    r1 = five_d_geodesic_drift_in_xi(vs_super, r=2.0, L_xi=1.0, dxi_dphi=0.1)
    r2 = five_d_geodesic_drift_in_xi(vs_super, r=2.0, L_xi=1.0, dxi_dphi=0.2)
    assert r2["drift_per_revolution"] == pytest.approx(2 * r1["drift_per_revolution"], rel=1e-12)
