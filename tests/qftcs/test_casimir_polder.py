"""Tests for Casimir-Polder force module."""

import math

import pytest

from systrophe.qftcs.casimir_polder import (
    casimir_polder_force_static,
    casimir_polder_with_rotation,
    distance_dependence_exponent_in_band,
    frame_drag_correction,
    novelty_scan,
    thermal_correction_at_T,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_casimir_polder_static_attractive():
    F = casimir_polder_force_static(1.0)
    assert F < 0


def test_casimir_polder_static_inverse_5th_power():
    F1 = casimir_polder_force_static(1.0)
    F2 = casimir_polder_force_static(2.0)
    assert F2 == pytest.approx(F1 / 32, rel=1e-12)


def test_frame_drag_correction_finite(vs):
    c = frame_drag_correction(vs, r=1.5)
    assert math.isfinite(c)


def test_casimir_polder_with_rotation_returns_dict(vs):
    res = casimir_polder_with_rotation(vs, distance=0.5)
    assert "F_modified" in res
    assert "frame_drag_correction" in res


def test_distance_dependence_exponent_finite(vs):
    n = distance_dependence_exponent_in_band(vs, r_inner=0.1, r_outer=1.0)
    assert math.isfinite(n)


def test_thermal_correction_zero_T_no_correction(vs):
    res = thermal_correction_at_T(vs, distance=1.0, temperature=0.0)
    assert res["correction_factor"] == 1.0


def test_thermal_correction_high_T_dominates(vs):
    res_low = thermal_correction_at_T(vs, distance=1.0, temperature=0.01)
    res_high = thermal_correction_at_T(vs, distance=1.0, temperature=100.0)
    assert res_high["correction_factor"] > res_low["correction_factor"]


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_d_values=10)
    assert "verdict" in res
