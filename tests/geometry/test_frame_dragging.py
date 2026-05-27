"""Tests for frame-dragging signatures."""

import numpy as np
import pytest

from systrophe.geometry.frame_dragging import (
    FrameDragging,
    compare_to_kerr_LT,
    frame_dragging_pattern,
    gyroscope_precession_angle,
    inertial_drag_zero_locus,
    kerr_lense_thirring,
    lense_thirring_frequency,
    total_drag_per_revolution,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_lense_thirring_finite(vs):
    om = lense_thirring_frequency(vs, r=2.0)
    assert np.isfinite(om)


def test_lense_thirring_supercritical_nonzero(vs):
    """Supercritical cylinder has nonzero LT precession."""
    om = lense_thirring_frequency(vs, r=2.0)
    assert abs(om) > 1e-6


def test_gyroscope_precession_proportional_to_time(vs):
    angle_1 = gyroscope_precession_angle(vs, r=2.0, time=1.0)
    angle_2 = gyroscope_precession_angle(vs, r=2.0, time=2.0)
    assert angle_2 == pytest.approx(2 * angle_1, rel=1e-12)


def test_frame_dragging_pattern_returns_dict(vs):
    pattern = frame_dragging_pattern(vs, r_min=1.05, r_max=10.0, n_grid=50)
    assert "r_grid" in pattern
    assert "Omega_LT_array" in pattern
    assert len(pattern["r_grid"]) == 50


def test_kerr_LT_zero_at_zero_a():
    """Kerr LT vanishes if a = 0 (non-rotating BH)."""
    om = kerr_lense_thirring(a=0.0, M=1.0, r=10.0)
    assert om == 0.0


def test_kerr_LT_positive_for_positive_a():
    """Kerr LT positive for a > 0 (prograde rotation)."""
    om = kerr_lense_thirring(a=0.5, M=1.0, r=10.0)
    assert om > 0


def test_compare_to_kerr_returns_dict(vs):
    cmp = compare_to_kerr_LT(vs, r=2.0)
    assert "Omega_LT_LP" in cmp
    assert "Omega_LT_Kerr" in cmp
    assert "ratio_LP_over_Kerr" in cmp


def test_zero_crossings_in_supercritical(vs):
    """Supercritical exterior may have zero-crossings of Omega_LT."""
    zeros = inertial_drag_zero_locus(vs, r_min=1.05, r_max=20.0)
    assert isinstance(zeros, list)


def test_total_drag_per_revolution(vs):
    drag = total_drag_per_revolution(vs, r=2.0, Omega_obs=1.0)
    assert np.isfinite(drag)


def test_total_drag_scales_inverse_Omega(vs):
    """Doubling Omega_obs halves the drag per revolution (smaller T)."""
    drag_1 = total_drag_per_revolution(vs, r=2.0, Omega_obs=1.0)
    drag_2 = total_drag_per_revolution(vs, r=2.0, Omega_obs=2.0)
    assert drag_2 == pytest.approx(drag_1 / 2, rel=1e-12)
