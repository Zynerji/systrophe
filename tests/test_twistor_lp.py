"""Tests for twistor LP module."""

import math

import pytest

from systrophe.twistor_lp import (
    TwistorAtPoint,
    alpha_plane_at_r,
    novelty_scan,
    pair_extinction_twistor,
    self_dual_test,
    twistor_inner_product,
    twistor_norm,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_alpha_plane_returns_TwistorAtPoint(vs):
    Z = alpha_plane_at_r(vs, r=1.5)
    assert isinstance(Z, TwistorAtPoint)


def test_alpha_plane_norm_positive(vs):
    Z = alpha_plane_at_r(vs, r=1.5)
    assert Z.norm_squared > 0


def test_twistor_norm_finite(vs):
    n = twistor_norm(vs, r=1.5)
    assert math.isfinite(n)


def test_inner_product_with_self_real(vs):
    """<Z, Z> should be real and positive."""
    ip = twistor_inner_product(vs, 1.5, 1.5)
    assert abs(ip.imag) < 1e-9
    assert ip.real > 0


def test_self_dual_test_returns_dict(vs):
    res = self_dual_test(vs, r=1.5)
    assert "is_self_dual" in res


def test_pair_extinction_at_pi_collapses(vs):
    res = pair_extinction_twistor(vs, r=1.5, delta=math.pi)
    assert res["norm_pair"] == 0.0
    assert res["twistor_collapsed"] is True


def test_pair_extinction_at_zero_unchanged(vs):
    res = pair_extinction_twistor(vs, r=1.5, delta=0.0)
    assert res["norm_pair"] == pytest.approx(res["norm_single"], rel=1e-12)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_radii=10)
    assert "verdict" in res
