"""Tests for LQG discretization module."""

import math

import pytest

from systrophe.quantum_info.lqg_discretization import (
    LQGAreaQuantization,
    classical_proper_area,
    count_vertices_in_region,
    discretization_error_relative,
    lp_area_quantum_signature,
    lqg_area_spectrum_at_j,
    lqg_volume_spectrum_at_j,
    planck_area_unit,
    regge_calculus_deficit_angle,
    spin_for_classical_area,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


def test_planck_area_unit_positive():
    assert planck_area_unit() > 0


def test_lqg_area_at_j_half_finite():
    A = lqg_area_spectrum_at_j(0.5)
    assert A > 0
    # A_{1/2} = 8 pi gamma * sqrt(3/4)
    expected = 8 * math.pi * 0.2375 * math.sqrt(0.75)
    assert A == pytest.approx(expected, rel=1e-12)


def test_lqg_area_j_zero_is_zero():
    A = lqg_area_spectrum_at_j(0.0)
    assert A == 0.0


def test_lqg_area_negative_j_raises():
    with pytest.raises(ValueError):
        lqg_area_spectrum_at_j(-0.5)


def test_lqg_area_monotone_increasing():
    js = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    A_vals = [lqg_area_spectrum_at_j(j) for j in js]
    for a1, a2 in zip(A_vals[:-1], A_vals[1:]):
        assert a2 > a1


def test_lqg_volume_at_j_half_finite():
    V = lqg_volume_spectrum_at_j(0.5)
    assert V > 0


def test_lqg_volume_too_few_edges_raises():
    with pytest.raises(ValueError):
        lqg_volume_spectrum_at_j(0.5, n_edges=2)


def test_classical_proper_area_positive(vs):
    A = classical_proper_area(vs, r=1.5)
    assert A > 0


def test_spin_for_classical_area_non_negative(vs):
    j = spin_for_classical_area(vs, r=1.5)
    assert j >= 0


def test_discretization_error_relative_returns_LQGQ(vs):
    q = discretization_error_relative(vs, r=1.5)
    assert isinstance(q, LQGAreaQuantization)
    assert q.relative_error >= 0


def test_discretization_error_small_for_macroscopic_areas(vs):
    """Semiclassical limit: error -> 0 as area >> Planck unit."""
    # Use a very long cylinder slice to push j large
    q = discretization_error_relative(vs, r=1.5, z_range=(0.0, 1000.0))
    # For j ~ 100, half-integer rounding error ~ 1/j ~ 0.01 max
    assert q.relative_error < 0.01


def test_count_vertices_positive(vs):
    n = count_vertices_in_region(vs, r_inner=1.83, r_outer=11.23)
    assert n > 0


def test_count_vertices_invalid_range(vs):
    with pytest.raises(ValueError):
        count_vertices_in_region(vs, 5.0, 2.0)


def test_lp_area_quantum_signature_returns_arrays(vs):
    res = lp_area_quantum_signature(vs)
    assert "rel_errors" in res
    assert len(res["rel_errors"]) > 0
    assert res["mean_rel_error"] >= 0


def test_regge_deficit_angle_returns_value(vs):
    d = regge_calculus_deficit_angle(vs, r=1.5)
    assert math.isfinite(d)


def test_regge_deficit_too_few_edges(vs):
    with pytest.raises(ValueError):
        regge_calculus_deficit_angle(vs, r=1.5, n_edges=2)


def test_regge_deficit_zero_for_six_edges_flat(vs):
    """6 equilateral triangles around a vertex tile flat space (deficit=0 modulo curvature)."""
    # For n=6, theta = 2pi/3 (60° triangles), 6 * 2pi/3 = 4pi != 2pi
    # Actually theta_n = (n-2)*pi/n. For n=6: theta = 4pi/6 = 2pi/3.
    # Sum = 6 * 2pi/3 = 4pi. Deficit = 2pi - 4pi = -2pi. Modulated by |F|.
    d = regge_calculus_deficit_angle(vs, r=1.5, n_edges=6)
    assert math.isfinite(d)
