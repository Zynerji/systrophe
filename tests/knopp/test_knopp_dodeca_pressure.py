"""Tests for the interior pressure/gradient derivation (knopp_dodeca_pressure)."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_dodeca_alignment import FACE_LOCK_DEG, horn_torus_sdf
from systrophe.knopp.knopp_dodeca_pressure import (
    axis_amplitudes,
    field_axes,
    gorkov_gradient,
    gorkov_potential,
    pressure_report,
    radiation_pressure,
    summarise_pressure,
    trap_stiffness,
    tube_interior_points,
    wall_pressure_dipole,
    wavenumber,
)


def test_wavenumber_demo_parity_and_validation():
    assert wavenumber(0.0) == 14.0
    assert wavenumber(1.0) == 38.0
    with pytest.raises(ValueError):
        wavenumber(1.5)


def test_gradient_analytic_matches_finite_difference():
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    ax = field_axes(FACE_LOCK_DEG)
    k = wavenumber(sat)
    pts = tube_interior_points(n=8)[:6]
    g_an = gorkov_gradient(pts, ax, a2, k)
    h = 1e-5
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        g_num = (gorkov_potential(pts + e, ax, a2, k)
                 - gorkov_potential(pts - e, ax, a2, k)) / (2 * h)
        assert np.allclose(g_an[:, i], g_num, atol=1e-4)


def test_pressure_bounded_by_saturation_ceiling():
    a2, sat = axis_amplitudes(FACE_LOCK_DEG)
    ax = field_axes(FACE_LOCK_DEG)
    pts = tube_interior_points(n=24)
    I = radiation_pressure(pts, ax, a2, wavenumber(sat))
    ceiling = float(np.sum(a2) / 2.0)
    assert np.all(I >= 0.0)
    assert np.all(I <= ceiling + 1e-12)


def test_mean_interior_pressure_is_half_ceiling():
    # spatial average of cos^2 over many wavelengths is 1/2
    r = pressure_report(FACE_LOCK_DEG)
    assert math.isclose(r.mean_interior_pressure, 0.5 * r.pressure_ceiling,
                        rel_tol=0.05)


def test_gradient_linear_and_stiffness_quadratic_in_k():
    r = pressure_report(FACE_LOCK_DEG)
    assert 0.9 < r.gradient_k_exponent < 1.05
    assert math.isclose(r.stiffness_k_exponent, 2.0, abs_tol=1e-6)
    a2, _ = axis_amplitudes(FACE_LOCK_DEG)
    assert math.isclose(trap_stiffness(a2, 20.0) / trap_stiffness(a2, 10.0),
                        4.0, rel_tol=1e-12)


def test_face_lock_dominates_point_lock():
    rf = pressure_report(FACE_LOCK_DEG)
    rv = pressure_report(0.0)
    assert rf.pressure_ceiling > 4 * rv.pressure_ceiling
    assert rf.max_gradient > 5 * rv.max_gradient
    assert rf.fill_fraction > 0.4
    assert rv.fill_fraction < 0.05


def test_wall_dipole_needs_steering_lobe():
    assert abs(wall_pressure_dipole(FACE_LOCK_DEG, eps=0.0)) < 1e-10
    d1 = wall_pressure_dipole(FACE_LOCK_DEG, eps=0.22)
    d2 = wall_pressure_dipole(FACE_LOCK_DEG, eps=0.44)
    assert d1 > 0
    assert math.isclose(d2 / d1, 2.0, rel_tol=1e-6)
    with pytest.raises(ValueError):
        wall_pressure_dipole(FACE_LOCK_DEG, eps=-0.1)


def test_interior_grid_is_inside_the_tube():
    pts = tube_interior_points(n=20)
    assert len(pts) > 100
    assert np.all(horn_torus_sdf(pts) < 0.0)


def test_pressure_report_fields_and_summary():
    r = pressure_report(FACE_LOCK_DEG)
    assert 0.0 <= r.saturation <= 1.0
    assert r.max_interior_pressure <= r.pressure_ceiling + 1e-12
    assert r.catcher_verdict in ("novel_structure", "smooth", "uniform")
    text = summarise_pressure(r)
    assert "pressure ceiling" in text and "catcher" in text
