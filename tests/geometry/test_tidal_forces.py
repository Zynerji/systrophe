"""Tests for tidal forces module."""

import math

import pytest

from systrophe.geometry.tidal_forces import (
    angular_squeeze_to_radial_stretch_ratio,
    geodesic_deviation_evolution,
    multi_band_tidal_signature,
    riemann_scalar_radial,
    safety_radius_for_probe,
    spaghettification_proxy,
    tidal_squeeze_at_radius,
    tidal_stretch_at_radius,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_riemann_scalar_radial_finite(vs_super):
    T = riemann_scalar_radial(vs_super, r=2.0)
    assert math.isfinite(T)


def test_tidal_stretch_scales_with_probe_length(vs_super):
    T1 = tidal_stretch_at_radius(vs_super, r=2.0, probe_length=1.0)
    T2 = tidal_stretch_at_radius(vs_super, r=2.0, probe_length=2.0)
    if abs(T1) > 1e-10:
        assert T2 == pytest.approx(2 * T1, rel=1e-10)


def test_tidal_squeeze_returns_value(vs_super):
    sq = tidal_squeeze_at_radius(vs_super, r=2.0)
    assert math.isfinite(sq)


def test_geodesic_deviation_returns_dict(vs_super):
    res = geodesic_deviation_evolution(vs_super, r_start=1.1, r_end=2.0)
    assert "final_xi" in res
    assert "growth_ratio" in res


def test_geodesic_deviation_invalid_range_raises(vs_super):
    with pytest.raises(ValueError):
        geodesic_deviation_evolution(vs_super, r_start=2.0, r_end=1.0)


def test_geodesic_deviation_max_xi_geq_initial(vs_super):
    res = geodesic_deviation_evolution(vs_super, r_start=1.1, r_end=2.0,
                                       deviation_initial=1.0)
    assert res["max_xi"] >= 1.0 - 1e-9


def test_spaghettification_proxy_non_negative(vs_super):
    s = spaghettification_proxy(vs_super, r=2.0)
    assert s >= 0


def test_multi_band_tidal_returns_list(vs_super):
    bands = multi_band_tidal_signature(vs_super, n_bands=3)
    assert isinstance(bands, list)
    assert len(bands) <= 3


def test_multi_band_tidal_subcritical_empty(vs_sub):
    bands = multi_band_tidal_signature(vs_sub)
    assert bands == []


def test_multi_band_tidal_entries_complete(vs_super):
    bands = multi_band_tidal_signature(vs_super, n_bands=2)
    for b in bands:
        assert "band_index" in b
        assert "tidal_scalar" in b
        assert "r_inner" in b
        assert "r_outer" in b


def test_safety_radius_returns_dict(vs_super):
    res = safety_radius_for_probe(vs_super, material_strength=0.01)
    assert "material_strength" in res
    assert "safe" in res


def test_safety_radius_high_strength_safe(vs_super):
    """With very strong material, probe is safe everywhere."""
    res = safety_radius_for_probe(vs_super, material_strength=1e10)
    assert res["safe"] is True


def test_angular_to_radial_ratio_finite(vs_super):
    ratio = angular_squeeze_to_radial_stretch_ratio(vs_super, r=2.0)
    assert math.isfinite(ratio) or ratio == float("inf")


def test_tidal_scalar_subcritical_finite(vs_sub):
    T = riemann_scalar_radial(vs_sub, r=2.0)
    assert math.isfinite(T)
