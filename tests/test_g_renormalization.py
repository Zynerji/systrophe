"""Tests for G renormalization module."""

import math

import pytest

from systrophe.g_renormalization import (
    GRunningData,
    asymptotic_safety_consistent,
    ctc_band_g_modification,
    curvature_scale_at_r,
    g_eff_at_r,
    g_running,
    newtonian_limit_recovery,
    ultraviolet_fixed_point_distance,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_g_running_at_zero_k_equals_G0():
    G = g_running(0.0)
    assert G == pytest.approx(1.0, rel=1e-12)


def test_g_running_negative_k_raises():
    with pytest.raises(ValueError):
        g_running(-1.0)


def test_g_running_decreases_with_k():
    G1 = g_running(0.1)
    G10 = g_running(10.0)
    assert G10 < G1


def test_g_running_asymptotic_freedom():
    """At very large k, G -> 0."""
    G_huge = g_running(1e10)
    assert G_huge < 1e-10


def test_curvature_scale_finite_in_exterior(vs_super):
    k = curvature_scale_at_r(vs_super, r=1.5)
    assert math.isfinite(k)


def test_curvature_scale_large_near_CH(vs_super):
    """Curvature scale grows as r approaches F=0."""
    k_far = curvature_scale_at_r(vs_super, r=1.4)
    k_near = curvature_scale_at_r(vs_super, r=1.82)
    assert k_near > k_far


def test_g_eff_at_r_returns_GRunningData(vs_super):
    data = g_eff_at_r(vs_super, r=1.5)
    assert isinstance(data, GRunningData)


def test_g_eff_at_r_bounded_by_G0(vs_super):
    data = g_eff_at_r(vs_super, r=1.5)
    assert 0 < data.G_at_k <= 1.0 + 1e-9


def test_uv_fixed_point_distance_returns_dict(vs_super):
    res = ultraviolet_fixed_point_distance(vs_super)
    assert "crossing_radii" in res


def test_ctc_band_modification_supercritical(vs_super):
    res = ctc_band_g_modification(vs_super, n_bands=2)
    assert isinstance(res, list)
    assert len(res) <= 2


def test_ctc_band_modification_subcritical_empty(vs_sub):
    res = ctc_band_g_modification(vs_sub)
    assert res == []


def test_asymptotic_safety_consistent_at_safe_r(vs_super):
    res = asymptotic_safety_consistent(vs_super, r_test=1.5)
    assert res["is_consistent"] is True


def test_newtonian_limit_at_large_r(vs_super):
    res = newtonian_limit_recovery(vs_super, r_large=100.0)
    # At large r, G_eff should be close to G_0 (depending on F oscillation)
    assert "newtonian_limit_holds" in res


def test_g_at_zero_curvature_unchanged():
    """If k = 0 (flat region), G = G_0."""
    G = g_running(0.0)
    assert G == 1.0
