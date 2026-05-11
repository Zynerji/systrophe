"""Tests for higher-spin fields on LP background."""

import numpy as np
import pytest

from systrophe.higher_spin_fields import (
    TensorMode,
    VectorMode,
    compare_spin_sectors,
    gravitational_wave_amplitude_at_horizon,
    mode_energy,
    solve_tensor_mode,
    solve_vector_mode,
    vector_mode_spectrum,
)
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- Vector field ------------------------------------------------------

def test_vector_mode_solve_returns_VectorMode(vs):
    mode = solve_vector_mode(vs, omega=1.0)
    assert isinstance(mode, VectorMode)
    assert mode.r_grid.shape[0] > 0
    assert mode.A_t.shape == mode.r_grid.shape


def test_vector_mode_omega_recorded(vs):
    om = 2.5
    mode = solve_vector_mode(vs, omega=om)
    assert mode.omega == om


def test_vector_mode_finite(vs):
    mode = solve_vector_mode(vs, omega=1.0, r_max=5.0)
    assert np.all(np.isfinite(mode.A_t))
    assert np.all(np.isfinite(mode.A_phi))


def test_vector_mode_energy_positive(vs):
    mode = solve_vector_mode(vs, omega=1.0)
    e = mode_energy(mode, vs)
    assert e >= 0


def test_vector_mode_spectrum_returns_list(vs):
    spectrum = vector_mode_spectrum(vs, omega_range=(0.1, 2.0), n_omega=4)
    assert len(spectrum) == 4
    assert all(isinstance(m, VectorMode) for m in spectrum)


# ----- Tensor field ------------------------------------------------------

def test_tensor_mode_solve_returns_TensorMode(vs):
    mode = solve_tensor_mode(vs, omega=1.0, m=2)
    assert isinstance(mode, TensorMode)
    assert mode.h_plus.shape == mode.r_grid.shape


def test_tensor_mode_finite(vs):
    mode = solve_tensor_mode(vs, omega=1.0, m=2)
    assert np.all(np.isfinite(mode.h_plus))
    assert np.all(np.isfinite(mode.h_cross))


def test_tensor_mode_m_recorded(vs):
    mode = solve_tensor_mode(vs, omega=1.0, m=2)
    assert mode.m == 2


# ----- Spectrum sweep ----------------------------------------------------

def test_gravitational_wave_amplitude_finite(vs):
    """GW amplitude at horizon is a finite number."""
    amp = gravitational_wave_amplitude_at_horizon(vs, omega=1.0, r_horizon=5.0)
    assert np.isfinite(amp)
    assert amp >= 0


def test_compare_spin_sectors_returns_dict(vs):
    cmp = compare_spin_sectors(vs, omega=1.0)
    assert "scalar_proxy" in cmp
    assert "spin_1_amplitude" in cmp
    assert "spin_2_amplitude" in cmp
    assert "ratio_2_over_1" in cmp


def test_compare_spin_sectors_amplitudes_nonnegative(vs):
    cmp = compare_spin_sectors(vs, omega=1.0)
    assert cmp["spin_1_amplitude"] >= 0
    assert cmp["spin_2_amplitude"] >= 0
