"""Geodesic module tests."""

import numpy as np
import pytest

from systrophe.geometry.geodesic import (
    CircularOrbit,
    is_omega_timelike,
    omega_for_target_coord_time,
    timelike_omega_bounds,
)


def test_minkowski_circular_orbit_timelike_for_subluminal_omega():
    """In flat Minkowski: F = 1, K = 0, L = r^2. Timelike iff |Omega r| < 1."""
    r = 2.0
    F, K, L = 1.0, 0.0, r ** 2
    assert is_omega_timelike(F, K, L, r, Omega=0.4)  # v = 0.8 < c
    assert not is_omega_timelike(F, K, L, r, Omega=0.6)  # v = 1.2 > c


def test_omega_bounds_recover_minkowski_lightcone():
    """At F=1, K=0, L=r^2: Omega_+- = +- 1/r."""
    r = 2.0
    om_lo, om_hi = timelike_omega_bounds(F=1.0, K=0.0, L=r ** 2, r=r)
    assert om_lo == pytest.approx(-1.0 / r)
    assert om_hi == pytest.approx(1.0 / r)


def test_circular_orbit_revolution_periods():
    """At Omega = 1.0: dt = 2 pi per revolution."""
    orb = CircularOrbit(r=2.0, F=1.0, K=0.0, L=4.0, Omega=0.4)
    assert orb.coord_dt_per_revolution == pytest.approx(2.0 * np.pi / 0.4)
    # tau / t = sqrt(1 - 0.4^2 * 4) = sqrt(0.36) = 0.6
    expected_dtau = 2.0 * np.pi / 0.4 * np.sqrt(1.0 - 0.4 ** 2 * 4.0)
    assert orb.proper_dtau_per_revolution == pytest.approx(expected_dtau)


def test_circular_orbit_in_ctc_region_flag():
    """L < 0 -> is_in_ctc_region True."""
    orb_ctc = CircularOrbit(r=3.0, F=1.0, K=0.0, L=-1.0, Omega=2.0)
    assert orb_ctc.is_in_ctc_region
    orb_no = CircularOrbit(r=3.0, F=1.0, K=0.0, L=+1.0, Omega=0.1)
    assert not orb_no.is_in_ctc_region


def test_omega_for_target_coord_time_inverse():
    """Omega -> 2 pi / Omega is the coord-time-per-rev relationship."""
    target_dt = 5.0
    Omega = omega_for_target_coord_time(target_dt)
    assert Omega == pytest.approx(2.0 * np.pi / target_dt)


def test_negative_target_dt_gives_negative_omega():
    """Backward time travel requires Omega < 0."""
    Omega = omega_for_target_coord_time(target_dt=-1.0)
    assert Omega < 0.0
