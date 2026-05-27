"""Tests for the warp-geometry floor optimizer (Van den Broeck shell scaling)."""

import math

import pytest

from systrophe.geometry.warp_geometry import (
    GeometryFloorReport,
    optimize_geometry,
    principal_joules,
    qi_wall_floor_m,
    summarise_geometry,
    PLANCK_LENGTH_M,
)


def test_principal_scales_as_R_squared_over_sigma():
    base = principal_joules(1.0, 1.0)
    assert principal_joules(2.0, 1.0) == pytest.approx(4.0 * base, rel=1e-9)
    assert principal_joules(1.0, 2.0) == pytest.approx(0.5 * base, rel=1e-9)


def test_qi_wall_floor_is_planck_scaled():
    assert qi_wall_floor_m(100.0) == pytest.approx(100.0 * PLANCK_LENGTH_M)


def test_geometry_reduces_principal_by_many_orders():
    r = optimize_geometry(planck_multiple=100.0)
    assert isinstance(r, GeometryFloorReport)
    assert r.optimized_principal_J < r.baseline_principal_J
    assert r.reduction_orders_of_magnitude > 20.0  # large, scaling-grounded


def test_band_coverage_reduces_paid_principal_further():
    r = optimize_geometry(out_of_band_fraction=0.139)
    assert r.combined_paid_principal_J < r.optimized_principal_J


def test_residual_gap_remains_positive_no_feasibility_claim():
    """Honesty guard: even after geometry + coverage the floor stays well above
    a lab squeezed-vacuum source. The wall is NOT broken."""
    r = optimize_geometry()
    assert r.residual_orders_of_magnitude > 10.0


def test_honest_flags_are_set():
    r = optimize_geometry(planck_multiple=100.0)
    assert r.wall_is_planck_scale is True       # theory breaks here
    assert r.blowup_energy_modeled is False      # Pfenning-Ford term omitted


def test_summary_and_catcher():
    r = optimize_geometry()
    s = summarise_geometry(r)
    assert "OOM" in s and "PLANCK-WALL" in s
    assert r.novelty_verdict in {"novel_structure", "smooth", "uniform"}


def test_velocity_sweep_is_v_squared():
    from systrophe.geometry.warp_geometry import velocity_sweep
    sw = velocity_sweep((1.0, 0.1, 0.01))
    # principal ~ v^2: 0.1c -> 2 OOM below luminal, 0.01c -> 4 OOM
    assert sw[0.1]["oom_below_luminal"] == pytest.approx(2.0, abs=0.05)
    assert sw[0.01]["oom_below_luminal"] == pytest.approx(4.0, abs=0.05)
