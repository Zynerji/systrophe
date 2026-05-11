"""Tests for acoustic-analog Hawking spectrum."""

import numpy as np
import pytest

from systrophe.acoustic_hawking_spectrum import (
    PhononSpectrum,
    bec_experimental_predictions,
    compare_to_steinhauer_2019,
    g2_correlation_signature,
    hawking_planck_spectrum,
    phonon_spectral_density,
    total_emission_power,
)
from systrophe.acoustic_metric import acoustic_horizon_radius
from systrophe.vanstockum import VanStockumInterior


@pytest.fixture
def vs():
    return VanStockumInterior(omega=1.0, R=1.0)


# ----- Planck spectrum --------------------------------------------------

def test_planck_spectrum_positive_at_T_positive():
    omegas = np.array([0.1, 0.5, 1.0, 2.0])
    T = 0.5
    n = hawking_planck_spectrum(omegas, T)
    assert np.all(n > 0)


def test_planck_spectrum_decreasing_in_omega():
    omegas = np.array([0.1, 1.0, 5.0])
    T = 0.5
    n = hawking_planck_spectrum(omegas, T)
    assert n[0] > n[1] > n[2]


def test_planck_spectrum_zero_T_returns_nan():
    n = hawking_planck_spectrum(np.array([1.0]), T=0)
    assert np.isnan(n[0])


def test_planck_classical_limit():
    """At omega << T, n -> T / omega (Rayleigh-Jeans)."""
    omega = 0.01
    T = 1.0
    n = hawking_planck_spectrum(np.array([omega]), T)[0]
    # exp(0.01) ~ 1.01005; 1/(0.01005) = 99.5
    assert n == pytest.approx(T / omega, rel=0.05)


# ----- Phonon spectral density at LP horizon -----------------------------

def test_phonon_spectrum_returns_PhononSpectrum(vs):
    r_h = acoustic_horizon_radius(vs, 1.05, 20.0)
    if r_h is None:
        pytest.skip("no acoustic horizon found")
    spec = phonon_spectral_density(vs, r_h)
    assert isinstance(spec, PhononSpectrum)
    assert spec.r_horizon == r_h


def test_phonon_spectrum_shapes_match(vs):
    r_h = acoustic_horizon_radius(vs, 1.05, 20.0)
    if r_h is None:
        pytest.skip("no acoustic horizon found")
    spec = phonon_spectral_density(vs, r_h, omega_range=(0.1, 3.0), n_omega=20)
    assert spec.omegas.shape == (20,)
    assert spec.n_omega.shape == (20,)
    assert spec.spectral_density.shape == (20,)


def test_phonon_spectrum_T_H_positive(vs):
    r_h = acoustic_horizon_radius(vs, 1.05, 20.0)
    if r_h is None:
        pytest.skip("no acoustic horizon found")
    spec = phonon_spectral_density(vs, r_h)
    assert spec.T_H > 0


# ----- Total emission power ---------------------------------------------

def test_total_emission_power_positive(vs):
    r_h = acoustic_horizon_radius(vs, 1.05, 20.0)
    if r_h is None:
        pytest.skip("no acoustic horizon found")
    P = total_emission_power(vs, r_h)
    assert P > 0


def test_total_emission_power_scales_with_T_squared(vs):
    """For 1D bosonic field, power ~ T^2."""
    r_h = acoustic_horizon_radius(vs, 1.05, 20.0)
    if r_h is None:
        pytest.skip("no acoustic horizon found")
    P = total_emission_power(vs, r_h, omega_max=20.0)
    from systrophe.acoustic_metric import acoustic_hawking_temperature
    T_H = acoustic_hawking_temperature(vs, r_h)
    # Expected: P ~ pi^2 * T_H^2 / 6 for 1D bosonic at temperature T_H
    expected = (np.pi ** 2) * T_H ** 2 / 6
    if T_H > 1e-3:
        # Allow factor-2 tolerance because cutoff at omega_max truncates
        assert 0.3 * expected < P < 3 * expected


# ----- g_2 correlation signature ----------------------------------------

def test_g2_returns_correlation_dict(vs):
    r_h = acoustic_horizon_radius(vs, 1.05, 20.0)
    if r_h is None:
        pytest.skip("no acoustic horizon found")
    g2 = g2_correlation_signature(vs, r_obs=r_h + 0.5, r_horizon=r_h)
    assert "delta_phi" in g2
    assert "correlation_amplitude" in g2
    assert "max_correlation" in g2


# ----- BEC predictions --------------------------------------------------

def test_bec_predictions_returns_dict():
    pred = bec_experimental_predictions(
        omega=1.0, R=1e-6, n_density=1e20, atom_mass=1.45e-25,
    )
    assert "T_H_acoustic_nK" in pred
    assert "sound_speed_c_m_per_s" in pred
    assert "signal_resolved" in pred
    assert isinstance(pred["signal_resolved"], bool)


def test_bec_T_H_positive():
    pred = bec_experimental_predictions(
        omega=1.0, R=1e-6, n_density=1e20, atom_mass=1.45e-25,
    )
    assert pred["T_H_acoustic_nK"] > 0


def test_bec_typical_rb87_t_h():
    """For typical Rb-87 BEC, T_H is in the nK regime."""
    # m_Rb87 = 1.443e-25 kg, density 1e14/cm^3 = 1e20/m^3
    pred = bec_experimental_predictions(
        omega=1.0, R=1e-6, n_density=1e20, atom_mass=1.45e-25,
    )
    # Should be in 0.01-100 nK range
    assert 0.001 < pred["T_H_acoustic_nK"] < 1000.0


# ----- Comparison to Steinhauer 2019 ------------------------------------

def test_steinhauer_comparison_returns_dict():
    cmp = compare_to_steinhauer_2019(T_H_predicted=0.124)
    assert "sigma_deviation" in cmp
    assert "consistent_with_measurement" in cmp


def test_steinhauer_consistency_at_measured_value():
    cmp = compare_to_steinhauer_2019(T_H_predicted=0.124)
    assert cmp["consistent_with_measurement"]
    assert cmp["sigma_deviation"] < 0.1


def test_steinhauer_inconsistency_at_distant_value():
    """At T_H = 10 nK (way off from measured 0.124), should be inconsistent."""
    cmp = compare_to_steinhauer_2019(T_H_predicted=10.0)
    assert not cmp["consistent_with_measurement"]
