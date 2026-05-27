"""Tests for BH pair production module."""

import math

import pytest

from systrophe.qftcs.bh_pair_production import (
    BHPairRate,
    field_strength_at_r,
    novelty_scan,
    pair_extinction_modification,
    production_locus,
    schwinger_analog_rate,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_field_strength_at_r_finite(vs_super):
    E = field_strength_at_r(vs_super, r=1.5)
    assert math.isfinite(E)
    assert E >= 0


def test_schwinger_rate_returns_BHPairRate(vs_super):
    res = schwinger_analog_rate(vs_super, r=1.5)
    assert isinstance(res, BHPairRate)


def test_schwinger_rate_in_unit_interval(vs_super):
    res = schwinger_analog_rate(vs_super, r=1.5)
    assert 0 <= res.production_rate <= 1


def test_schwinger_rate_decreases_with_mass(vs_super):
    """Heavier BHs are exponentially suppressed."""
    r_light = schwinger_analog_rate(vs_super, r=1.5, bh_mass=0.1)
    r_heavy = schwinger_analog_rate(vs_super, r=1.5, bh_mass=10.0)
    assert r_heavy.production_rate <= r_light.production_rate


def test_production_locus_returns_dict(vs_super):
    res = production_locus(vs_super, bh_mass=0.5)
    assert "r_max_rate" in res
    assert "max_rate" in res


def test_pair_extinction_at_pi_zero(vs_super):
    res = pair_extinction_modification(vs_super, delta=math.pi, r=1.5)
    assert res["rate_pair"] == 0.0


def test_pair_extinction_at_zero_unchanged(vs_super):
    res = pair_extinction_modification(vs_super, delta=0.0, r=1.5)
    assert res["rate_pair"] == pytest.approx(res["rate_single"], rel=1e-9)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_radii=20)
    assert "verdict" in res
