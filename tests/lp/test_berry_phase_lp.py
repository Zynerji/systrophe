"""Tests for Berry phase LP module."""

import math

import pytest

from systrophe.lp.berry_phase_lp import (
    berry_connection,
    berry_curvature,
    berry_phase_per_revolution,
    novelty_scan,
    pair_extinction_berry,
    topological_chern_number,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_berry_connection_finite(vs):
    A = berry_connection(vs, r=1.5)
    assert math.isfinite(A)


def test_berry_phase_per_revolution_finite(vs):
    gamma = berry_phase_per_revolution(vs, r=1.5)
    assert math.isfinite(gamma)


def test_berry_phase_2pi_times_connection(vs):
    A = berry_connection(vs, r=1.5)
    gamma = berry_phase_per_revolution(vs, r=1.5)
    assert gamma == pytest.approx(2 * math.pi * A, rel=1e-12)


def test_berry_curvature_finite(vs):
    F = berry_curvature(vs, r=1.5)
    assert math.isfinite(F)


def test_chern_number_returns_value(vs):
    C = topological_chern_number(vs, r_inner=1.05, r_outer=1.5)
    assert math.isfinite(C)


def test_pair_extinction_at_pi_zero(vs):
    res = pair_extinction_berry(vs, r=1.5, delta=math.pi)
    if math.isfinite(res["gamma_pair"]):
        assert res["gamma_pair"] == 0.0
    assert res["phase_trivial_at_pi"] is True


def test_pair_extinction_at_zero_unchanged(vs):
    res = pair_extinction_berry(vs, r=1.5, delta=0.0)
    if math.isfinite(res["gamma_single"]):
        assert res["gamma_pair"] == pytest.approx(res["gamma_single"], rel=1e-12)


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_radii=10)
    assert "verdict" in res
