"""Tests for optical-fiber analog."""

import numpy as np
import pytest

from systrophe.optical_fiber_analog import (
    FiberEffectiveMetric,
    compare_to_steinhauer_2010,
    fiber_analog_hawking_T,
    fiber_analog_horizon,
    fiber_effective_metric,
    gaussian_pump_profile,
    linear_pump_profile,
    pulse_collision_radiation,
)


def test_fiber_metric_subsonic():
    """v < c_probe: subsonic (F > 0)."""
    m = fiber_effective_metric(n_pump=1.5, v_group=0.3, n_probe=1.5)
    assert m.F > 0
    assert not m.is_supersonic


def test_fiber_metric_supersonic():
    """v > c_probe: supersonic (F < 0)."""
    c_probe = 1.0 / 1.5
    m = fiber_effective_metric(n_pump=1.5, v_group=c_probe + 0.1, n_probe=1.5)
    assert m.F < 0
    assert m.is_supersonic


def test_fiber_metric_at_horizon():
    """v = c_probe: F = 0 (horizon)."""
    c_probe = 1.0 / 1.5
    m = fiber_effective_metric(n_pump=1.5, v_group=c_probe, n_probe=1.5)
    assert abs(m.F) < 1e-12


def test_gaussian_pump_profile_at_center():
    p = gaussian_pump_profile(amplitude=0.9, sigma=1.0, x0=5.0)
    assert p(5.0) == pytest.approx(0.9, rel=1e-12)


def test_gaussian_pump_profile_at_infinity():
    p = gaussian_pump_profile(amplitude=0.9, sigma=1.0, x0=5.0)
    assert p(100.0) == pytest.approx(0.0, abs=1e-30)


def test_linear_pump_profile_endpoints():
    p = linear_pump_profile(v_start=0.4, v_end=0.9, x_start=0.0, x_end=10.0)
    assert p(0.0) == 0.4
    assert p(10.0) == 0.9


def test_fiber_analog_horizon_with_linear_ramp():
    """Linear pump ramping past c_probe ~ 0.667 should give a horizon."""
    p = linear_pump_profile(v_start=0.4, v_end=0.9)
    result = fiber_analog_horizon(p, probe_index=1.5)
    assert result["n_horizons"] == 1


def test_fiber_analog_horizon_no_crossing():
    """Pump always subsonic gives no horizon."""
    p = gaussian_pump_profile(amplitude=0.3, sigma=1.0)
    result = fiber_analog_horizon(p, probe_index=1.5)
    assert result["n_horizons"] == 0


def test_fiber_hawking_T_positive():
    p = linear_pump_profile(v_start=0.4, v_end=0.9)
    horizon = fiber_analog_horizon(p, probe_index=1.5)
    if not horizon["horizons"]:
        pytest.skip("no horizon")
    T_H = fiber_analog_hawking_T(p, horizon["horizons"][0], probe_index=1.5)
    assert T_H > 0


def test_pulse_collision_no_horizon_when_subsonic():
    """Two slow pulses: no horizon between them."""
    result = pulse_collision_radiation(v_pump_1=0.2, v_pump_2=0.3,
                                            n_probe=1.5)
    assert not result["horizon_present"]


def test_pulse_collision_horizon_when_relative_supersonic():
    """Two fast pulses with large relative v: horizon present."""
    result = pulse_collision_radiation(v_pump_1=0.1, v_pump_2=0.9,
                                            n_probe=1.5)
    assert result["horizon_present"]


def test_steinhauer_comparison_returns_dict():
    cmp = compare_to_steinhauer_2010(T_H_predicted=1e-5)
    assert "consistent" in cmp
    assert cmp["consistent"]
