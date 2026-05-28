"""Tests for the Bowen-York NR initial data scaffold."""

import math

import numpy as np
import pytest

from systrophe.knopp.knopp_toroidal_nr_initial_data import (
    BowenYorkBinary,
    NRInitialDataReport,
    bowen_york_A_squared_at_point,
    hamilton_constraint_residual,
    initial_data_diagnostic,
    puncture_conformal_psi,
    scan_separation,
    summarise_initial_data,
)


# ----- BowenYorkBinary -----------------------------------------------------


def test_construction_defaults():
    b = BowenYorkBinary()
    assert b.M == 1.0 and b.d == 2.0 and b.chi == 1.0


def test_kepler_P_formula():
    b = BowenYorkBinary(M=1.0, d=4.0)
    # P = M sqrt(2M/d) = 1 * sqrt(0.5) = 1/sqrt(2)
    assert b.kepler_P == pytest.approx(1.0 / math.sqrt(2.0), rel=1e-12)


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        BowenYorkBinary(M=-1.0)
    with pytest.raises(ValueError):
        BowenYorkBinary(d=0.0)
    with pytest.raises(ValueError):
        BowenYorkBinary(chi=1.5)


# ----- A^2 (Bowen-York extrinsic curvature) ------------------------------


def test_A_squared_at_punctures_diverges():
    """At a BH centre, r -> 0 -> A^2 -> infinity."""
    b = BowenYorkBinary(M=1.0, d=2.0, chi=1.0)
    x_at_BH1 = np.array([0.0, 0.0, 1.0])  # BH 1 centre
    assert math.isinf(bowen_york_A_squared_at_point(b, x_at_BH1))


def test_A_squared_finite_in_interior():
    """At interior points (away from punctures), A^2 is finite."""
    b = BowenYorkBinary(M=1.0, d=2.0, chi=1.0)
    for x in (np.array([0.0, 0.0, 0.0]),
              np.array([2.0, 0.0, 0.0]),
              np.array([1.0, 0.5, 0.3])):
        A = bowen_york_A_squared_at_point(b, x)
        assert math.isfinite(A)
        assert A >= 0.0


def test_A_squared_decreases_with_distance():
    """Far from the binary, A^2 -> 0 as 1/r^4 (momentum) or 1/r^6 (spin)."""
    b = BowenYorkBinary()
    A_near = bowen_york_A_squared_at_point(b, np.array([2.0, 0.0, 0.0]))
    A_far = bowen_york_A_squared_at_point(b, np.array([50.0, 0.0, 0.0]))
    assert A_far < A_near
    assert A_far > 0.0


# ----- puncture conformal factor -----------------------------------------


def test_psi_diverges_approaching_BH_centre():
    b = BowenYorkBinary(M=1.0, d=2.0)
    # Move very close to BH 1 (skip the exact-zero guard) and verify
    # the 1/r divergence.
    x_near = np.array([0.0, 0.0, 1.0 - 1e-5])
    assert puncture_conformal_psi(b, x_near) > 1e4


def test_psi_at_infinity_is_one():
    b = BowenYorkBinary(M=1.0, d=2.0)
    x_far = np.array([1e6, 0.0, 0.0])
    assert puncture_conformal_psi(b, x_far) == pytest.approx(1.0, abs=1e-5)


def test_psi_at_midpoint_two():
    """psi(midpoint) = 1 + 2 * M/(2 * d/2) = 1 + 2M/d. At d=2M: psi=2."""
    b = BowenYorkBinary(M=1.0, d=2.0)
    psi = puncture_conformal_psi(b, np.array([0.0, 0.0, 0.0]))
    assert psi == pytest.approx(2.0, rel=1e-12)


# ----- Hamilton constraint residual --------------------------------------


def test_hamilton_residual_finite_in_interior():
    b = BowenYorkBinary()
    for x in (np.array([0.0, 0.0, 0.0]),
              np.array([2.0, 0.0, 0.0])):
        psi = puncture_conformal_psi(b, x)
        res = hamilton_constraint_residual(b, x, psi)
        assert math.isfinite(res)
        assert res >= 0.0


def test_hamilton_residual_decreases_with_d():
    """Wider binaries have smaller source at fixed off-axis distance."""
    r_2 = hamilton_constraint_residual(
        BowenYorkBinary(d=2.0), np.array([1.0, 0.0, 0.0]),
        puncture_conformal_psi(BowenYorkBinary(d=2.0),
                                np.array([1.0, 0.0, 0.0])),
    )
    r_20 = hamilton_constraint_residual(
        BowenYorkBinary(d=20.0), np.array([10.0, 0.0, 0.0]),
        puncture_conformal_psi(BowenYorkBinary(d=20.0),
                                np.array([10.0, 0.0, 0.0])),
    )
    assert r_20 < r_2


# ----- diagnostic report -------------------------------------------------


def test_report_dataclass():
    b = BowenYorkBinary()
    r = initial_data_diagnostic(b)
    assert isinstance(r, NRInitialDataReport)


def test_report_always_well_conditioned_for_sane_inputs():
    """Bowen-York puncture data is REGULAR by Brandt-Brügmann 1997 --
    initial data exists for ANY (d, chi) configuration. The diagnostic
    should report well_conditioned = True for all sane inputs."""
    for d in (2.0, 3.0, 5.0, 10.0, 50.0):
        for chi in (0.0, 0.5, 1.0):
            b = BowenYorkBinary(M=1.0, d=d, chi=chi)
            r = initial_data_diagnostic(b)
            assert r.constraint_well_conditioned is True, (
                f"d={d}, chi={chi} reported not well-conditioned"
            )


def test_report_psi_midpoint_matches_analytic():
    b = BowenYorkBinary(M=1.0, d=2.0)
    r = initial_data_diagnostic(b)
    # psi(midpoint) = 1 + 2M/d = 2.0
    assert r.psi_midpoint == pytest.approx(2.0, rel=1e-12)


def test_scan_separation_returns_list():
    out = scan_separation(M=1.0, chi=1.0, d_values=(2.0, 5.0, 10.0))
    assert len(out) == 3
    for r in out:
        assert isinstance(r, NRInitialDataReport)


# ----- Newton-Kantorovich Hamilton-constraint solver --------------------


def test_hamilton_solver_returns_dataclass():
    from systrophe.knopp.knopp_toroidal_nr_initial_data import (
        HamiltonSolverResult, solve_hamilton_constraint_radial,
    )
    b = BowenYorkBinary(M=1.0, d=2.0, chi=1.0)
    res = solve_hamilton_constraint_radial(
        b, r_min=0.5, r_max=20.0, n_grid=80, max_iter=20,
    )
    assert isinstance(res, HamiltonSolverResult)


def test_hamilton_solver_psi_finite_everywhere():
    from systrophe.knopp.knopp_toroidal_nr_initial_data import (
        solve_hamilton_constraint_radial,
    )
    b = BowenYorkBinary(M=1.0, d=2.0, chi=1.0)
    res = solve_hamilton_constraint_radial(
        b, r_min=0.5, r_max=20.0, n_grid=80, max_iter=20,
    )
    assert np.all(np.isfinite(res.psi_total_grid))
    assert np.all(res.psi_total_grid > 0.0)


def test_hamilton_solver_adm_mass_near_sum_of_punctures():
    from systrophe.knopp.knopp_toroidal_nr_initial_data import (
        solve_hamilton_constraint_radial, adm_mass_consistency,
    )
    b = BowenYorkBinary(M=1.0, d=2.0, chi=1.0)
    res = solve_hamilton_constraint_radial(
        b, r_min=0.5, r_max=20.0, n_grid=80, max_iter=20,
    )
    # ADM mass should be ~ 2 M (within 5% for the toy 1D solver)
    assert abs(res.adm_mass - 2.0) < 0.1


def test_adm_consistency_check_returns_dict():
    from systrophe.knopp.knopp_toroidal_nr_initial_data import (
        solve_hamilton_constraint_radial, adm_mass_consistency,
    )
    b = BowenYorkBinary(M=1.0, d=2.0, chi=1.0)
    res = solve_hamilton_constraint_radial(
        b, r_min=0.5, r_max=20.0, n_grid=80, max_iter=20,
    )
    con = adm_mass_consistency(b, res)
    for k in ("M_expected_no_binding", "M_expected_with_binding",
              "M_measured", "consistent", "rel_error"):
        assert k in con


def test_summary_mentions_evolution_bottleneck():
    """The summary should be honest about the framework's failure mode
    being the evolution, not the initial slice."""
    b = BowenYorkBinary()
    s = summarise_initial_data(initial_data_diagnostic(b))
    assert "EVOLUTION" in s or "below ISCO" in s or "plunge" in s
