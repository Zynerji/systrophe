"""Tests for floquet-engineer."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from floquet_engineer import FloquetEngineer, FloquetSweepReport
from systrophe.floquet_mobius import FloquetMobiusResult


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_with_three_branches():
    fe = FloquetEngineer(branch_energies=[1.0, 1.1, 1.2])
    np.testing.assert_allclose(fe.branch_energies, [1.0, 1.1, 1.2])
    assert fe.hopping == 0.0


def test_construction_invalid_branch_count_raises():
    with pytest.raises(ValueError):
        FloquetEngineer(branch_energies=[1.0, 1.1])  # only 2


# ---------------------------------------------------------------------------
# analyze (single-point)
# ---------------------------------------------------------------------------


def test_analyze_returns_floquet_result():
    fe = FloquetEngineer(branch_energies=[1.0, 1.0, 1.0])
    res = fe.analyze(drive_amp=0.5, omega_drive=1.0)
    assert isinstance(res, FloquetMobiusResult)


def test_analyze_quasi_energy_has_three_entries():
    fe = FloquetEngineer(branch_energies=[1.0, 1.1, 1.2])
    res = fe.analyze(drive_amp=0.2, omega_drive=0.5)
    assert len(res.quasi_energies) == 3


# ---------------------------------------------------------------------------
# static_limit + z3_symmetry sanity checks
# ---------------------------------------------------------------------------


def test_static_limit_returns_dict():
    fe = FloquetEngineer(branch_energies=[1.0, 1.0, 1.0])
    lim = fe.static_limit()
    assert isinstance(lim, dict)


def test_z3_symmetry_returns_dict():
    fe = FloquetEngineer(branch_energies=[1.0, 1.0, 1.0])
    sym = fe.z3_symmetry()
    assert isinstance(sym, dict)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def test_sweep_returns_report():
    fe = FloquetEngineer(branch_energies=[1.0, 1.1, 1.2])
    rep = fe.sweep(
        drive_amps=np.linspace(0.1, 0.5, 3),
        omega_drives=np.linspace(0.5, 1.5, 3),
    )
    assert isinstance(rep, FloquetSweepReport)
    assert rep.gap_map.shape == (3, 3)


def test_sweep_gap_map_finite():
    fe = FloquetEngineer(branch_energies=[1.0, 1.1, 1.2])
    rep = fe.sweep(
        drive_amps=np.linspace(0.1, 0.4, 2),
        omega_drives=np.linspace(0.5, 1.0, 2),
    )
    assert np.all(np.isfinite(rep.gap_map))


def test_sweep_max_gap_matches():
    fe = FloquetEngineer(branch_energies=[1.0, 1.1, 1.2])
    rep = fe.sweep(
        drive_amps=np.linspace(0.1, 0.5, 3),
        omega_drives=np.linspace(0.5, 1.5, 3),
    )
    assert rep.max_gap == pytest.approx(float(np.max(rep.gap_map)))


def test_sweep_resonances_returned_as_list():
    fe = FloquetEngineer(branch_energies=[1.0, 1.1, 1.2])
    rep = fe.sweep(
        drive_amps=np.linspace(0.05, 0.5, 4),
        omega_drives=np.linspace(0.5, 1.5, 4),
    )
    assert isinstance(rep.resonances, list)


# ---------------------------------------------------------------------------
# Z_3 symmetric branches (equal energies) — special case
# ---------------------------------------------------------------------------


def test_z3_symmetric_no_static_gap():
    """When branch energies are equal, the static H is Z_3-symmetric;
    static spectrum collapses to one triple-degenerate level."""
    fe = FloquetEngineer(branch_energies=[1.0, 1.0, 1.0])
    res = fe.analyze(drive_amp=0.0, omega_drive=1.0)
    # All 3 quasi-energies should be equal at drive_amp = 0
    np.testing.assert_allclose(
        res.quasi_energies, res.quasi_energies[0] * np.ones(3),
        atol=1e-3,
    )
