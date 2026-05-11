"""Tests for holographic complexity module."""

import math

import pytest

from systrophe.holographic_complexity import (
    action_proxy_C,
    complexity_growth_rate,
    novelty_scan,
    pair_extinction_complexity,
    volume_proxy_C,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_volume_proxy_positive(vs):
    C = volume_proxy_C(vs, r_inner=1.83, r_outer=11.23)
    assert C > 0


def test_action_proxy_finite(vs):
    C = action_proxy_C(vs, r_inner=1.83, r_outer=11.23)
    assert math.isfinite(C)


def test_complexity_growth_rate_positive(vs):
    rate = complexity_growth_rate(vs, r_inner=1.83, r_outer=11.23)
    assert rate > 0


def test_pair_extinction_at_pi_zero(vs):
    res = pair_extinction_complexity(vs, r_inner=1.83, r_outer=11.23, delta=math.pi)
    assert res["C_V_pair"] == 0.0
    assert res["C_A_pair"] == 0.0


def test_pair_extinction_at_zero_unchanged(vs):
    res = pair_extinction_complexity(vs, r_inner=1.83, r_outer=11.23, delta=0.0)
    assert res["C_V_pair"] == pytest.approx(res["C_V_single"], rel=1e-12)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_band_widths=10)
    assert "verdict" in res
