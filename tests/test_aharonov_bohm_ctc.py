"""Tests for Aharonov-Bohm CTC module."""

import math

import pytest

from systrophe.aharonov_bohm_ctc import (
    aharonov_bohm_phase,
    detector_interference_amplitude,
    enclosed_flux,
    novelty_scan,
    pair_extinction_AB,
    phase_jump_across_CH,
    topological_invariant_winding,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_aharonov_bohm_phase_finite(vs):
    phase = aharonov_bohm_phase(vs, r=1.5)
    assert math.isfinite(phase)


def test_aharonov_bohm_phase_proportional_to_K(vs):
    phase = aharonov_bohm_phase(vs, r=1.5)
    K = enclosed_flux(vs, 1.5)
    assert phase == pytest.approx(2 * math.pi * K, rel=1e-12)


def test_enclosed_flux_finite(vs):
    K = enclosed_flux(vs, r=1.5)
    assert math.isfinite(K)


def test_phase_jump_returns_dict(vs):
    res = phase_jump_across_CH(vs, r_CH=1.83)
    assert "phase_jump" in res


def test_topological_winding_returns_int(vs):
    w = topological_invariant_winding(vs)
    assert isinstance(w, int)


def test_pair_extinction_at_pi_zero(vs):
    res = pair_extinction_AB(vs, r=1.5, delta=math.pi)
    assert res["phi_pair"] == 0.0
    assert res["trivial_at_pi"] is True


def test_pair_extinction_at_zero_unchanged(vs):
    res = pair_extinction_AB(vs, r=1.5, delta=0.0)
    assert res["phi_pair"] == pytest.approx(res["phi_single"], rel=1e-12)


def test_detector_amplitude_in_unit_interval(vs):
    A = detector_interference_amplitude(vs, r=1.5, n_loops=1)
    assert -1 <= A <= 1


def test_detector_amplitude_zero_loops_one(vs):
    A = detector_interference_amplitude(vs, r=1.5, n_loops=0)
    assert A == pytest.approx(1.0, rel=1e-12)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_radii=10)
    assert "verdict" in res
