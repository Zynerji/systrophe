"""Tests for the Krasnikov-Pair anti-phase extinction."""

import math

import numpy as np

from systrophe.krasnikov_pair import (
    krasnikov_pair_NEC_radial,
    krasnikov_pair_total_negative_energy,
    krasnikov_pair_extinction_curve,
    novelty_scan,
)


def test_NEC_at_delta_zero_is_4x_single():
    """At delta=0 the pair adds coherently; pair = 4 x single."""
    from systrophe.krasnikov_tube import krasnikov_NEC_radial
    single = krasnikov_NEC_radial(0.0, t=1.0, alpha=4.0)
    pair = krasnikov_pair_NEC_radial(0.0, t=1.0, alpha=4.0, delta=0.0)
    assert abs(pair - 4.0 * single) < 1e-10


def test_NEC_at_delta_pi_is_zero():
    """At delta=pi (anti-phase) the pair extinguishes -> wall NEC = 0."""
    pair = krasnikov_pair_NEC_radial(0.0, t=1.0, alpha=4.0, delta=math.pi)
    assert abs(pair) < 1e-10


def test_NEC_at_delta_2pi_is_4x_single():
    """delta = 2pi recovers the in-phase amplitude."""
    from systrophe.krasnikov_tube import krasnikov_NEC_radial
    single = krasnikov_NEC_radial(0.0, t=1.0, alpha=4.0)
    pair = krasnikov_pair_NEC_radial(0.0, t=1.0, alpha=4.0, delta=2 * math.pi)
    assert abs(pair - 4.0 * single) < 1e-10


def test_total_negative_energy_zero_at_delta_pi():
    e = krasnikov_pair_total_negative_energy(delta=math.pi, alpha=4.0)
    assert abs(e) < 1e-8


def test_total_negative_energy_negative_at_delta_zero():
    e = krasnikov_pair_total_negative_energy(delta=0.0, alpha=4.0)
    assert e < 0


def test_extinction_curve_minimum_near_pi():
    delta_grid, E = krasnikov_pair_extinction_curve(alpha=4.0)
    idx_min = int(np.argmin(np.abs(E)))
    assert abs(delta_grid[idx_min] - math.pi) < 0.2


def test_novelty_scan_returns_verdict():
    res = novelty_scan(n_delta=21)
    assert "novelty_verdict" in res
    assert res["novelty_verdict"] in ("smooth", "uniform", "novel_structure")


def test_extinction_delta_reported():
    res = novelty_scan(n_delta=51)
    assert "extinction_delta" in res
    # Should be within 0.2 of pi
    assert abs(res["extinction_delta"] - math.pi) < 0.2
