"""Tests for Unruh effect module."""

import math

import pytest

from systrophe.unruh_effect import (
    centripetal_acceleration_at_orbit,
    combined_unruh_hawking_T,
    comparison_to_Unruh_DeWitt,
    frame_drag_modified_T,
    novelty_scan,
    unruh_temperature,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_unruh_temperature_zero_at_zero_a():
    assert unruh_temperature(0.0) == 0.0


def test_unruh_temperature_proportional_to_a():
    T1 = unruh_temperature(1.0)
    T2 = unruh_temperature(2.0)
    assert T2 == pytest.approx(2 * T1, rel=1e-12)


def test_unruh_temperature_formula():
    """T = a / (2 pi)."""
    T = unruh_temperature(2 * math.pi)
    assert T == pytest.approx(1.0, rel=1e-12)


def test_centripetal_acceleration_finite(vs_super):
    a = centripetal_acceleration_at_orbit(vs_super, r=1.5)
    assert math.isfinite(a)
    assert a > 0


def test_combined_unruh_hawking_returns_dict(vs_super):
    res = combined_unruh_hawking_T(vs_super, r=1.5)
    assert "T_Unruh" in res
    assert "T_Hawking" in res
    assert "T_combined" in res


def test_combined_T_geq_individual(vs_super):
    res = combined_unruh_hawking_T(vs_super, r=1.5)
    if math.isfinite(res["T_combined"]):
        assert res["T_combined"] >= res["T_Unruh"] - 1e-12
        assert res["T_combined"] >= res["T_Hawking"] - 1e-12


def test_frame_drag_modified_T_finite(vs_super):
    T = frame_drag_modified_T(vs_super, r=1.5)
    assert math.isfinite(T)


def test_unruh_dewitt_rate_zero_at_zero_a():
    res = comparison_to_Unruh_DeWitt(acceleration=0.0)
    assert res["rate"] == 0.0


def test_unruh_dewitt_rate_grows_with_a():
    r_low = comparison_to_Unruh_DeWitt(acceleration=0.1)
    r_high = comparison_to_Unruh_DeWitt(acceleration=10.0)
    assert r_high["rate"] > r_low["rate"]


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_r_values=10)
    assert "verdict" in res
