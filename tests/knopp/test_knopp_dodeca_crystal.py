"""Tests for the crystal-lock sweep (knopp_dodeca_crystal)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_alignment import FACE_LOCK_DEG
from systrophe.knopp.knopp_dodeca_crystal import (
    SQRT5,
    crystal_lock_sweep,
    crystal_order,
    pump_ring,
    registry_ladder,
    source_registry,
    summarise_crystal,
)


def test_source_registry_closed_form():
    assert math.isclose(float(source_registry(0.0)), 1.0, abs_tol=1e-12)
    x = 1.7
    expect = (math.cos(x) ** 2 + 5 * math.cos(x / SQRT5) ** 2) / 6
    assert math.isclose(float(source_registry(x)), expect, rel_tol=1e-12)


def test_perfect_registry_impossible_for_x_positive():
    # sqrt(5) incommensuration: Lambda < 1 strictly on x > 0
    x = np.linspace(0.5, 60.0, 30000)
    assert float(np.max(source_registry(x))) < 1.0 - 1e-4


def test_ladder_rungs_are_sqrt5_approximants():
    ladder = registry_ladder()
    assert len(ladder) >= 4
    # each rung: x/pi and x/(sqrt5 pi) both near integers (m, n) with m/n ~ sqrt5
    best = max(ladder, key=lambda d: d["lambda"])
    m = round(best["m_over_pi"])
    n = round(best["n_over_pi"])
    assert abs(best["m_over_pi"] - m) < 0.05
    assert abs(best["n_over_pi"] - n) < 0.05
    assert math.isclose(m / n, SQRT5, rel_tol=0.01)   # 9/4 = 2.25 ~ 2.236
    assert best["lambda"] > 0.99


def test_pump_ring_geometry():
    ring = pump_ring(0.21)
    assert ring.shape == (64, 3)
    assert np.allclose(ring[:, 1], 0.7946544723 * 0.21)
    with pytest.raises(ValueError):
        pump_ring(-0.1)


def test_ring_coherence_locks_at_face_angle():
    on = crystal_order(FACE_LOCK_DEG, 0.212)
    off = crystal_order(FACE_LOCK_DEG + 4.0, 0.212)
    assert on.ring_coherence > 0.99
    assert off.ring_coherence < on.ring_coherence - 0.02
    assert on.order > off.order


def test_order_gated_by_full_spectrum():
    locked = crystal_order(FACE_LOCK_DEG, 0.212)
    point = crystal_order(0.0, 0.212)
    assert locked.full_spectrum
    assert not point.full_spectrum
    assert locked.order > point.order


def test_crystal_lock_sweep_finds_face_angle_lock():
    r = crystal_lock_sweep(
        scales=np.linspace(0.16, 0.30, 8),
        betas=np.linspace(30.0, 45.0, 7),
    )
    assert r.lock_at_face_angle
    assert math.isclose(r.lock_beta_deg, FACE_LOCK_DEG, abs_tol=2.5)
    assert 0.18 <= r.lock_scale <= 0.26
    assert r.lock_order > 0.6
    assert r.lock.ring_coherence > 0.99
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    text = summarise_crystal(r)
    assert "LOCK" in text and "approximant ladder" in text


def test_lock_is_q_independent():
    a = crystal_order(FACE_LOCK_DEG, 0.212, Q=30.0)
    b = crystal_order(FACE_LOCK_DEG, 0.212, Q=240.0)
    assert math.isclose(a.order, b.order, rel_tol=1e-6)
