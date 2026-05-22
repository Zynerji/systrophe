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
