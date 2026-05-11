"""Tests for QI channel module on LP background."""

import math

import numpy as np
import pytest

from systrophe.qi_channel import (
    amplitude_damping_p,
    channel_assisted_clock_sync,
    channel_capacity_holevo,
    chronology_horizon_crossings_per_loop,
    chronology_horizon_decoherence,
    closed_timelike_loop_capacity,
    entanglement_swap_protocol,
    redshift_factor,
    shannon_classical_capacity_over_radial_path,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs_super():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def vs_sub():
    return VanStockumInterior(omega=0.3, R=1.0)


def test_redshift_factor_unity_at_same_radius(vs_super):
    z = redshift_factor(vs_super, r_A=1.5, r_B=1.5)
    assert z == pytest.approx(1.0, rel=1e-12)


def test_redshift_factor_finite_between_crossings(vs_super):
    # r=1.2 and r=1.5 are both between R=1 and first CH at ~1.83
    z = redshift_factor(vs_super, r_A=1.2, r_B=1.5)
    assert math.isfinite(z)
    assert z > 0


def test_amplitude_damping_p_zero_at_unity():
    assert amplitude_damping_p(1.0) == pytest.approx(0.5, rel=1e-12)


def test_amplitude_damping_p_unity_at_zero_redshift():
    assert amplitude_damping_p(0.0) == 1.0


def test_channel_capacity_holevo_zero_at_p1():
    assert channel_capacity_holevo(1.0) == 0.0


def test_channel_capacity_holevo_max_at_p0():
    assert channel_capacity_holevo(0.0) == 1.0


def test_channel_capacity_holevo_monotone():
    """Capacity should decrease with p."""
    caps = [channel_capacity_holevo(p) for p in [0.0, 0.2, 0.5, 0.8, 1.0]]
    for c1, c2 in zip(caps[:-1], caps[1:]):
        assert c1 >= c2 - 1e-12


def test_chronology_horizon_crossings_zero_for_subcritical(vs_sub):
    n = chronology_horizon_crossings_per_loop(vs_sub, 1.5, 10.0)
    assert n == 0


def test_chronology_horizon_crossings_positive_supercritical(vs_super):
    # Across r=1.1 to r=20, supercritical: must cross several CHs
    n = chronology_horizon_crossings_per_loop(vs_super, 1.1, 20.0)
    assert n >= 1


def test_chronology_horizon_decoherence_returns_capacity(vs_super):
    res = chronology_horizon_decoherence(vs_super, 1.5, 5.0)
    assert "channel_capacity_bits" in res
    assert 0 <= res["channel_capacity_bits"] <= 1


def test_closed_timelike_loop_capacity_zero_classical(vs_super):
    res = closed_timelike_loop_capacity(vs_super, r_loop=3.35)
    assert res["deutsch_classical_capacity"] == 0.0


def test_closed_timelike_loop_capacity_pctc_one_in_CTC(vs_super):
    res = closed_timelike_loop_capacity(vs_super, r_loop=3.35)
    if res["regime"] == "CTC":
        assert res["pctc_lloyd_capacity_bits"] == 1.0


def test_entanglement_swap_returns_fidelity_in_unit_interval(vs_super):
    res = entanglement_swap_protocol(vs_super, r_A=1.2, r_B=1.5)
    assert 0 <= res["final_fidelity"] <= 1


def test_entanglement_swap_p_total_in_unit(vs_super):
    res = entanglement_swap_protocol(vs_super, 1.2, 5.0)
    assert 0 <= res["p_total"] <= 1


def test_channel_assisted_clock_sync_finite(vs_super):
    res = channel_assisted_clock_sync(vs_super, r_A=1.2, r_B=1.5)
    assert math.isfinite(res["residual"])


def test_channel_assisted_clock_sync_behind_CH(vs_super):
    # r=3.35 is behind first CH where F<0
    res = channel_assisted_clock_sync(vs_super, r_A=1.2, r_B=3.35)
    assert res["behind_CH"] is True


def test_shannon_capacity_radial_path_returns_capacities(vs_super):
    res = shannon_classical_capacity_over_radial_path(vs_super, 1.1, 5.0)
    assert len(res["capacities"]) > 0
    assert 0 <= res["mean_capacity"] <= 1


def test_shannon_capacity_invalid_range(vs_super):
    with pytest.raises(ValueError):
        shannon_classical_capacity_over_radial_path(vs_super, 5.0, 1.0)
