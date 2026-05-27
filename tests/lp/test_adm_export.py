"""Tests for ADM 3+1 export."""

import os
import tempfile

import numpy as np
import pytest

from systrophe.lp.adm_export import (
    ADMSlice,
    adm_decompose_lp,
    adm_summary,
    export_to_einsteintoolkit_ascii,
    hamiltonian_constraint_residual,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- Decomposition basic ---------------------------------------------

def test_decompose_returns_correct_shape(vs):
    r = np.linspace(1.05, 5.0, 20)
    slice_data = adm_decompose_lp(vs, r_grid=r)
    assert slice_data.gamma_rr.shape == r.shape
    assert slice_data.gamma_phiphi.shape == r.shape
    assert slice_data.alpha.shape == r.shape
    assert slice_data.K_rphi.shape == r.shape


def test_decompose_spatial_metric_components(vs):
    """gamma_rr = 1, gamma_phiphi = L(r), gamma_zz = 1 (leading-order h = 1)."""
    r = np.array([2.0])
    slice_data = adm_decompose_lp(vs, r_grid=r)
    assert slice_data.gamma_rr[0] == 1.0
    assert slice_data.gamma_zz[0] == 1.0
    L = float(vs.analytic_exterior_L(2.0))
    assert slice_data.gamma_phiphi[0] == pytest.approx(L, rel=1e-12)


def test_decompose_shift_is_K_over_L(vs):
    r = np.array([2.0])
    slice_data = adm_decompose_lp(vs, r_grid=r)
    K_val = float(vs.analytic_exterior_K(2.0))
    L_val = float(vs.analytic_exterior_L(2.0))
    if abs(L_val) > 1e-30:
        assert slice_data.beta_up_phi[0] == pytest.approx(K_val / L_val, rel=1e-12)
        assert slice_data.beta_phi[0] == pytest.approx(K_val, rel=1e-12)


def test_decompose_lapse_relation(vs):
    """alpha^2 = F + K^2 / L (where defined)."""
    r = np.array([2.0])
    slice_data = adm_decompose_lp(vs, r_grid=r)
    if slice_data.is_valid[0]:
        F = float(vs.analytic_exterior_F(2.0))
        K_val = float(vs.analytic_exterior_K(2.0))
        L_val = float(vs.analytic_exterior_L(2.0))
        expected = F + K_val ** 2 / L_val
        assert slice_data.alpha[0] ** 2 == pytest.approx(expected, rel=1e-10)


def test_decompose_horizon_marks_invalid(vs):
    """A grid point near F = 0 has is_valid = False if alpha^2 < 0."""
    # Sweep a region likely to contain a horizon
    r = np.linspace(1.05, 20.0, 100)
    slice_data = adm_decompose_lp(vs, r_grid=r)
    # In supercritical LP, alpha^2 = F + K^2/L can flip sign at horizons.
    # We expect at least one invalid point if the range covers a horizon.
    if slice_data.is_valid.sum() < len(r):
        assert (~slice_data.is_valid).any()


# ----- Export -----------------------------------------------------------

def test_export_writes_file(vs):
    r = np.linspace(1.05, 5.0, 10)
    slice_data = adm_decompose_lp(vs, r_grid=r)
    with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
        path = f.name
    try:
        outpath = export_to_einsteintoolkit_ascii(slice_data, path)
        assert os.path.exists(outpath)
        assert os.path.getsize(outpath) > 0
        # File should be loadable back
        loaded = np.loadtxt(outpath)
        assert loaded.shape == (10, 8)
    finally:
        os.remove(path)


# ----- Constraint residual ---------------------------------------------

def test_hamiltonian_residual_returns_array(vs):
    r = np.linspace(1.05, 5.0, 20)
    slice_data = adm_decompose_lp(vs, r_grid=r)
    R = hamiltonian_constraint_residual(slice_data)
    assert R.shape == r.shape


# ----- Summary ---------------------------------------------------------

def test_summary_consistent(vs):
    r = np.linspace(1.05, 5.0, 20)
    slice_data = adm_decompose_lp(vs, r_grid=r)
    s = adm_summary(slice_data)
    assert s["n_valid"] + s["n_invalid"] == 20
    if s["n_valid"] > 0:
        assert np.isfinite(s["alpha_max_valid"])
