"""Tests for the thick-wall VdB variational closure."""

import math

import pytest

from systrophe.vdb_thick_wall import (
    ThickWallReport,
    optimal_energy_geometrized,
    optimal_energy_numeric,
    summarise_thick_wall,
    thick_wall_floor,
)


def test_el_optimal_matches_closed_form():
    # E_min = 2 R_shell (sqrt(B_max)-1)^2 (exact for B=(1+c/r)^2 on [R,inf))
    R, Bmax = 1.0, 1e4
    assert optimal_energy_geometrized(R, Bmax) == pytest.approx(
        2.0 * R * (math.sqrt(Bmax) - 1.0) ** 2)


def test_numeric_confirms_analytic_order():
    # numeric integral is within a modest factor (linear-grid undersampling of
    # the 1/r^2-peaked integrand) but the same magnitude
    a = optimal_energy_geometrized(1.0, 1e4)
    n = optimal_energy_numeric(1.0, 1e4)
    assert 0.7 < n / a < 1.6


def test_thicker_wall_is_optimal():
    """The variational result: the energy DECREASES as the wall thickens, so
    the global minimum is the infinitely-thick (R2->inf) wall."""
    full = optimal_energy_geometrized(1.0, 1e4)            # R2 -> inf
    thin = optimal_energy_geometrized(1.0, 1e4, R2_over_Rshell=2.0)
    assert full < thin


def test_floor_is_two_rho_use_and_shell_independent():
    """The floor is E ~ 2 rho_use, independent of how small the exterior shell
    is -- shrinking the shell does NOT help."""
    r1 = thick_wall_floor(rho_use_m=2.0, R_shell_m=1e-15)
    r2 = thick_wall_floor(rho_use_m=2.0, R_shell_m=1e-30)
    assert r1.energy_per_proper_metre == pytest.approx(2.0, abs=0.05)
    assert r1.E_min_J == pytest.approx(r2.E_min_J, rel=1e-6)  # shell-independent
    assert r1.shrinking_shell_helps is False


def test_interior_usable_but_escape_closed():
    """The honest dual result: the interior IS usable (flat-B core, no tides),
    but the thick-wall escape is CLOSED -- the floor stays Jupiter-scale."""
    r = thick_wall_floor(rho_use_m=2.0)
    assert r.interior_is_flat_usable is True
    assert r.escape_closed is True
    assert 0.1 < r.E_min_jupiter_masses < 100.0
    assert r.residual_oom > 30.0


def test_summary():
    s = summarise_thick_wall(thick_wall_floor())
    assert "ESCAPE_CLOSED=True" in s and "interior_usable=True" in s
