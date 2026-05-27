"""Tests for Hawking radiation budget module."""

import math

import pytest

from systrophe.qftcs.hawking_budget import (
    CHRadiationBudget,
    bekenstein_hawking_entropy_at_CH,
    evaporation_time_estimate,
    hawking_temperature_at_CH,
    horizon_area_at_CH,
    log_periodic_temperature_pattern,
    per_CH_radiation_budget,
    surface_gravity_at_CH,
    total_radiation_lifetime,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_surface_gravity_positive(vs_super):
    kappa = surface_gravity_at_CH(vs_super, r_CH=1.83)
    assert kappa > 0


def test_hawking_temperature_positive(vs_super):
    T = hawking_temperature_at_CH(vs_super, r_CH=1.83)
    assert T > 0


def test_hawking_temperature_kappa_relation(vs_super):
    """T = kappa / (2 pi)."""
    kappa = surface_gravity_at_CH(vs_super, r_CH=1.83)
    T = hawking_temperature_at_CH(vs_super, r_CH=1.83)
    assert T == pytest.approx(kappa / (2 * math.pi), rel=1e-12)


def test_horizon_area_positive(vs_super):
    A = horizon_area_at_CH(vs_super, r_CH=1.83)
    assert A > 0


def test_horizon_area_scales_with_length(vs_super):
    A1 = horizon_area_at_CH(vs_super, r_CH=1.83, cylinder_length=1.0)
    A2 = horizon_area_at_CH(vs_super, r_CH=1.83, cylinder_length=2.0)
    assert A2 == pytest.approx(2 * A1, rel=1e-12)


def test_bekenstein_hawking_quarter_area(vs_super):
    A = horizon_area_at_CH(vs_super, r_CH=1.83)
    S = bekenstein_hawking_entropy_at_CH(vs_super, r_CH=1.83)
    assert S == pytest.approx(A / 4.0, rel=1e-12)


def test_evaporation_time_finite(vs_super):
    t = evaporation_time_estimate(vs_super, r_CH=1.83)
    assert math.isfinite(t)
    assert t > 0


def test_per_CH_returns_list(vs_super):
    bs = per_CH_radiation_budget(vs_super, n_horizons=3)
    assert isinstance(bs, list)
    assert all(isinstance(b, CHRadiationBudget) for b in bs)


def test_per_CH_subcritical_empty(vs_sub):
    bs = per_CH_radiation_budget(vs_sub)
    assert bs == []


def test_per_CH_entries_have_all_fields(vs_super):
    bs = per_CH_radiation_budget(vs_super, n_horizons=2)
    for b in bs:
        assert b.r_CH > 0
        assert b.surface_gravity > 0
        assert b.hawking_temperature > 0
        assert b.horizon_area > 0
        assert b.bekenstein_hawking_entropy > 0


def test_total_radiation_lifetime_positive(vs_super):
    L = total_radiation_lifetime(vs_super, n_horizons=3)
    assert L > 0
    assert math.isfinite(L)


def test_log_periodic_pattern_returns_dict(vs_super):
    res = log_periodic_temperature_pattern(vs_super, n_horizons=4)
    assert "ratios" in res
    assert "predicted_ratio" in res


def test_log_periodic_predicted_ratio_supercritical(vs_super):
    """predicted_ratio = exp(-pi/alpha) for supercritical."""
    res = log_periodic_temperature_pattern(vs_super, n_horizons=4)
    expected = math.exp(-math.pi / vs_super.alpha)
    assert res["predicted_ratio"] == pytest.approx(expected, rel=1e-12)


def test_log_periodic_subcritical_returns_nan(vs_sub):
    res = log_periodic_temperature_pattern(vs_sub)
    assert math.isnan(res["predicted_ratio"])
