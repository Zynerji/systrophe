"""Tests for solitons on LP module."""

import math

import numpy as np
import pytest

from systrophe.lp.solitons_on_lp import (
    bound_state_spectrum_in_band,
    kink_mass_on_lp,
    kink_profile,
    novelty_scan,
    pair_extinction_soliton_decay,
    topological_charge,
    vortex_winding_number,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_kink_profile_asymptotes():
    r = np.linspace(-10, 10, 100)
    phi = kink_profile(r, r_0=0.0, width=1.0)
    assert phi[0] == pytest.approx(-1.0, abs=1e-3)
    assert phi[-1] == pytest.approx(1.0, abs=1e-3)


def test_kink_mass_finite(vs):
    M = kink_mass_on_lp(vs, r_0=1.5)
    assert math.isfinite(M)
    assert M > 0


def test_kink_mass_inf_in_CTC(vs):
    # Inside CTC band F<0 -> mass undefined
    M = kink_mass_on_lp(vs, r_0=3.35)
    assert M == float("inf")


def test_vortex_winding_zero_for_constant():
    theta = np.zeros(10)
    assert vortex_winding_number(theta) == 0


def test_vortex_winding_one_for_2pi_loop():
    theta = np.linspace(0, 2 * math.pi, 100)
    assert vortex_winding_number(theta) == 1


def test_bound_state_spectrum_correct(vs):
    levels = bound_state_spectrum_in_band(vs, r_inner=1.0, r_outer=2.0, n_modes=3)
    assert len(levels) == 3
    # E_n / E_1 = n
    assert levels[1] == pytest.approx(2 * levels[0], rel=1e-12)
    assert levels[2] == pytest.approx(3 * levels[0], rel=1e-12)


def test_bound_state_invalid_band(vs):
    levels = bound_state_spectrum_in_band(vs, r_inner=5.0, r_outer=2.0)
    assert levels == []


def test_pair_extinction_at_pi_zero_mass(vs):
    res = pair_extinction_soliton_decay(vs, r_0=1.5, delta=math.pi)
    assert res["soliton_dissolved"] is True
    assert res["M_pair"] == 0.0


def test_pair_extinction_at_zero_unchanged(vs):
    res = pair_extinction_soliton_decay(vs, r_0=1.5, delta=0.0)
    assert res["M_pair"] == pytest.approx(res["M_single"], rel=1e-12)


def test_topological_charge_kink_is_one():
    r = np.linspace(-10, 10, 1000)
    phi = kink_profile(r, r_0=0.0, width=1.0)
    Q = topological_charge(phi, r)
    assert Q == 1


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_r0_values=10)
    assert "verdict" in res
