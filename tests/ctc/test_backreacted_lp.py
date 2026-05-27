"""Tests for the toy iterative semiclassical back-reaction scaffold."""

import math

import numpy as np
import pytest

from systrophe.ctc.backreacted_lp import (
    classical_F,
    classical_horizon,
    find_chronology_horizon,
    horizon_shift,
    iterate_backreaction,
    perturbed_F,
    polyakov_T_tt_general,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- classical -----------------------------------------------------------


def test_classical_F_callable_matches_analytic(vs):
    F = classical_F(vs)
    # Compare to VanStockum's analytic_exterior_F at a few sample points.
    for r in [1.1, 1.3, 1.5, 1.7]:
        expected = float(vs.analytic_exterior_F(np.array([r]))[0])
        assert F(r) == pytest.approx(expected, rel=1e-12)


def test_classical_horizon_omega_R_one(vs):
    # alpha = sqrt(3), r_H = exp(atan(sqrt(3))/sqrt(3)) = exp(pi/(3 sqrt(3))).
    expected = math.exp(math.pi / (3.0 * math.sqrt(3.0)))
    assert classical_horizon(vs) == pytest.approx(expected, rel=1e-12)


def test_classical_horizon_subcritical_rejected():
    vs_sub = VanStockumInterior(omega=0.4, R=1.0)
    with pytest.raises(ValueError):
        classical_horizon(vs_sub)


# ----- root finding --------------------------------------------------------


def test_find_horizon_recovers_classical(vs):
    F = classical_F(vs)
    r_H = find_chronology_horizon(F, r_min=1.01, r_max=3.0)
    assert r_H is not None
    assert r_H == pytest.approx(classical_horizon(vs), abs=1e-3)


def test_find_horizon_returns_none_when_F_positive(vs):
    # constant positive F has no chronology horizon
    F = lambda r: 1.0
    assert find_chronology_horizon(F, r_min=1.01, r_max=3.0) is None


# ----- Polyakov stress on a generic F --------------------------------------


def test_polyakov_finite_in_subhorizon_region(vs):
    F = classical_F(vs)
    rH = classical_horizon(vs)
    for r in np.linspace(1.05, 0.9 * rH, 5):
        T = polyakov_T_tt_general(F, float(r))
        assert math.isfinite(T)


def test_polyakov_nan_in_F_negative_region(vs):
    F = classical_F(vs)
    rH = classical_horizon(vs)
    # Choose r well past r_H where F is clearly negative.
    T = polyakov_T_tt_general(F, float(rH + 0.5))
    assert math.isnan(T)


# ----- iteration -----------------------------------------------------------


def test_iteration_zero_ell2_recovers_classical(vs):
    res = iterate_backreaction(vs, ell2=0.0, n_iter=5)
    assert res.verdict == "displaced"
    s = horizon_shift(res, vs)
    assert s["r_H_backreacted"] == pytest.approx(
        classical_horizon(vs), abs=1e-3,
    )
    # shift should be at the numerical-noise floor
    assert abs(s["shift"]) < 1e-3


def test_iteration_positive_ell2_pulls_horizon_inward(vs):
    # First-order backreaction shifts r_H *inward* (toward R).
    res = iterate_backreaction(
        vs, ell2=1e-4, n_iter=50, damping=0.2,
    )
    s = horizon_shift(res, vs)
    # Horizon either survives but is pulled inward, or gets swallowed.
    if s["r_H_backreacted"] is not None:
        assert s["shift"] < -1e-3  # strictly inward
    assert res.verdict in ("displaced", "unconverged", "swallowed")


def test_iteration_large_ell2_swallows_horizon(vs):
    res = iterate_backreaction(
        vs, ell2=1e-2, n_iter=20, damping=0.3,
    )
    assert res.verdict in ("swallowed", "censored")


def test_iteration_subcritical_rejected():
    vs_sub = VanStockumInterior(omega=0.4, R=1.0)
    with pytest.raises(ValueError):
        iterate_backreaction(vs_sub, ell2=1e-4)


def test_iteration_bad_damping(vs):
    with pytest.raises(ValueError):
        iterate_backreaction(vs, damping=0.0)
    with pytest.raises(ValueError):
        iterate_backreaction(vs, damping=1.5)


def test_iteration_steps_and_history_lengths_match(vs):
    res = iterate_backreaction(vs, ell2=1e-5, n_iter=10, damping=0.5)
    assert len(res.steps) == len(res.r_H_history)
    for step in res.steps:
        assert step.F_grid.shape == res.r_grid.shape
        assert step.delta_F_grid.shape == res.r_grid.shape
        assert step.T_tt_grid.shape == res.r_grid.shape


def test_horizon_shift_reports_classical(vs):
    res = iterate_backreaction(vs, ell2=0.0, n_iter=3)
    s = horizon_shift(res, vs)
    assert s["r_H_classical"] == pytest.approx(
        classical_horizon(vs), rel=1e-12,
    )


# ----- perturbed_F helper --------------------------------------------------


def test_perturbed_F_adds_delta():
    base = lambda r: r ** 2
    delta = lambda r: 0.5
    F = perturbed_F(base, delta)
    for r in [0.0, 1.0, 2.0]:
        assert F(r) == pytest.approx(r ** 2 + 0.5, abs=1e-12)
