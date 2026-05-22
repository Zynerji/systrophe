"""Tests for the Van den Broeck neck blow-up research scaffold (uncalibrated)."""

import math

import pytest

from systrophe.vdb_neck import (
    VdbFloorReport,
    blowup_energy_estimate,
    summarise_vdb,
    vdb_total_floor,
)


def test_blowup_zero_without_expansion():
    assert blowup_energy_estimate(1.0, 1.0, 1.0) == 0.0
    assert blowup_energy_estimate(2.0, 1.0, 1.0) == 0.0  # pocket < shell


def test_blowup_scales_with_ln_ratio_squared():
    e1 = blowup_energy_estimate(1e-30, 1e-3, 1e-30)
    e2 = blowup_energy_estimate(1e-30, 1e-3, 1e-30, prefactor_C=2.0)
    assert e2 == pytest.approx(2.0 * e1, rel=1e-9)
    assert e1 > 0.0


def test_blowup_dominates_and_reinflates():
    """The honest Pfenning-Ford direction: the blow-up term dominates the shell
    and re-inflates the optimistic shell-only floor."""
    r = vdb_total_floor(R_pocket_m=2.0, planck_multiple=100.0, prefactor_C=1.0)
    assert isinstance(r, VdbFloorReport)
    assert r.blowup_dominates is True
    assert r.total_floor_J > r.shell_principal_J
    assert r.blowup_reinflation_oom > 0.0
    assert r.total_reduction_oom < r.shell_only_reduction_oom


def test_scaffold_is_flagged_uncalibrated():
    r = vdb_total_floor()
    assert r.calibrated is False
    assert r.wall_is_planck_scale is True
    assert "UNCALIBRATED" in summarise_vdb(r)


def test_no_feasibility_claim():
    """Even with the (optimistic) shell + blow-up, the floor stays far above a
    lab source. The wall is not broken."""
    r = vdb_total_floor()
    assert r.residual_oom > 20.0


# --- CALIBRATED via the exact Einstein-tensor integral ----------------------

from systrophe.vdb_neck import (
    calibrate_K,
    pocket_energy_geometrized,
    vdb_calibrated_floor,
    summarise_vdb_calibrated,
)


def test_calibrate_K_converges_near_half():
    K = calibrate_K(1e5)
    assert 0.5 < K < 0.6  # tanh-wall constant, deep-pocket limit


def test_pocket_energy_scales_linearly_in_Bmax():
    """The exact integral scales as B_max^1, NOT ln^2(B_max) (the scaffold's
    error). Check the exponent over two decades."""
    e1 = pocket_energy_geometrized(1e3)
    e2 = pocket_energy_geometrized(1e5)
    exponent = math.log(e2 / e1) / math.log(1e5 / 1e3)
    assert exponent == pytest.approx(1.0, abs=0.05)


def test_calibrated_blowup_far_exceeds_scaffold():
    """Calibration corrects the scaffold UPWARD by many orders: the real
    blow-up is enormously larger than the ln^2 estimate."""
    r = vdb_calibrated_floor(R_pocket_m=2.0)
    assert r.blowup_calibrated_J > 1e10 * r.blowup_scaffold_J
    assert r.scaffold_overestimated_reduction_by_oom > 20.0


def test_geometry_does_not_reduce_floor():
    """The headline calibrated result: for a thin localized wall, VdB geometry
    does NOT reduce the floor (it is ~Jupiter-scale, set by R_pocket)."""
    r = vdb_calibrated_floor(R_pocket_m=2.0)
    assert r.calibrated is True
    assert r.geometry_reduces_floor is False
    assert r.net_reduction_oom <= 1.0          # no meaningful reduction
    assert 0.1 < r.total_floor_jupiter_masses < 100.0  # back at Jupiter scale


def test_calibrated_summary():
    r = vdb_calibrated_floor()
    s = summarise_vdb_calibrated(r)
    assert "CALIBRATED" in s and "geometry_reduces_floor=False" in s
