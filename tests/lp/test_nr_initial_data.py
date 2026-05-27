"""Tests for NR initial data module."""

import json
from pathlib import Path

import numpy as np
import pytest

from systrophe.lp.nr_initial_data import (
    adm_profile_1d,
    bssn_decomposition,
    cauchy_horizon_warning_for_foliation,
    export_initial_data_json,
    extrude_to_3d_cartesian,
    hamiltonian_constraint_residual,
    momentum_constraint_residual,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


@pytest.fixture
def r_grid():
    return np.linspace(1.05, 1.7, 50)


def test_adm_profile_returns_dict(vs, r_grid):
    prof = adm_profile_1d(vs, r_grid)
    assert "lapse" in prof
    assert "shift_phi" in prof
    assert "K_rphi" in prof


def test_adm_profile_lapse_real_between_CHs(vs, r_grid):
    """In a region without F=0 crossings, lapse should be real (no NaNs)."""
    prof = adm_profile_1d(vs, r_grid)
    assert not np.any(np.isnan(prof["lapse"]))


def test_hamiltonian_constraint_returns_dict(vs, r_grid):
    res = hamiltonian_constraint_residual(vs, r_grid)
    assert "mean_abs_residual" in res
    assert res["mean_abs_residual"] >= 0


def test_momentum_constraint_returns_dict(vs, r_grid):
    res = momentum_constraint_residual(vs, r_grid)
    assert "mean_abs_residual" in res


def test_bssn_decomposition_chi_positive(vs, r_grid):
    bssn = bssn_decomposition(vs, r_grid)
    assert np.all(bssn["chi"] > 0)


def test_bssn_decomposition_returns_all_components(vs, r_grid):
    bssn = bssn_decomposition(vs, r_grid)
    for key in ("chi", "tilde_gamma_rr", "tilde_gamma_pp", "tilde_A_rphi"):
        assert key in bssn


def test_extrude_to_3d_returns_grids(vs):
    res = extrude_to_3d_cartesian(vs, box_size=3.0, n_grid=8)
    assert res["F"].shape == (8, 8, 8)
    assert res["K"].shape == (8, 8, 8)
    assert res["L"].shape == (8, 8, 8)


def test_extrude_interior_F_unity(vs):
    """Interior points (r < R) should have F = 1."""
    res = extrude_to_3d_cartesian(vs, box_size=3.0, n_grid=10)
    interior_F = res["F"][res["interior_mask"]]
    assert np.all(interior_F == 1.0)


def test_export_initial_data_writes_file(vs, tmp_path):
    out = tmp_path / "initial_data.json"
    res = export_initial_data_json(vs, out, n_samples=20)
    assert Path(res["output_path"]).exists()


def test_export_initial_data_loadable(vs, tmp_path):
    out = tmp_path / "initial_data.json"
    export_initial_data_json(vs, out, n_samples=20)
    payload = json.loads(out.read_text())
    assert payload["omega_dust"] == 1.0
    assert payload["R"] == 1.0


def test_export_initial_data_constraints_present(vs, tmp_path):
    out = tmp_path / "initial_data.json"
    export_initial_data_json(vs, out, n_samples=20)
    payload = json.loads(out.read_text())
    assert "ham_mean_abs" in payload["constraints"]


def test_cauchy_horizon_warning_returns_min_alpha_sq(vs):
    """For supercritical, alpha_sq_min should be a finite number."""
    r_grid_wide = np.linspace(1.05, 20.0, 500)
    res = cauchy_horizon_warning_for_foliation(vs, r_grid_wide)
    import math
    assert math.isfinite(res["alpha_sq_min"])
    assert "r_first_breakdown" in res


def test_cauchy_horizon_no_breakdown_in_safe_region(vs):
    """In r in (R, first CH), no breakdown."""
    r_grid_safe = np.linspace(1.05, 1.7, 50)
    res = cauchy_horizon_warning_for_foliation(vs, r_grid_safe)
    # Might still have breakdown if first CH < 1.7; for omega=R=1, first CH ~1.83
    # so should be safe
    assert res["r_first_breakdown"] is None or res["r_first_breakdown"] > 1.5
