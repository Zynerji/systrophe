"""Tests for dynamical Casimir module."""

import numpy as np
import pytest

from systrophe.qftcs.dynamical_casimir import (
    DCEFluxResult,
    cavity_mode_frequency,
    dce_flux_estimate,
    dce_spectrum,
    linewidth_at_resonance,
    off_resonance_photon_number,
    on_resonance_photon_number,
    resonance_frequency_for_mode,
)


# ----- cavity modes -----------------------------------------------------

def test_cavity_mode_frequency_basic():
    """omega_1 = pi (for d_0 = 1, c = 1)."""
    omega = cavity_mode_frequency(n=1, d_0=1.0)
    assert omega == pytest.approx(np.pi, rel=1e-12)


def test_cavity_mode_frequency_n_scaling():
    omega1 = cavity_mode_frequency(n=1, d_0=1.0)
    omega3 = cavity_mode_frequency(n=3, d_0=1.0)
    assert omega3 == pytest.approx(3 * omega1, rel=1e-12)


def test_cavity_mode_validates():
    with pytest.raises(ValueError):
        cavity_mode_frequency(n=0, d_0=1.0)
    with pytest.raises(ValueError):
        cavity_mode_frequency(n=1, d_0=-1.0)


def test_resonance_frequency_2x_mode():
    """Parametric resonance: Omega_res = 2 omega_n."""
    omega = 3.14
    assert resonance_frequency_for_mode(omega) == 2 * omega


# ----- on-resonance flux ----------------------------------------------

def test_on_resonance_at_zero_time_is_zero():
    N = on_resonance_photon_number(epsilon=0.01, omega_n=1.0, t=0.0)
    assert N == 0.0


def test_on_resonance_grows_with_time():
    N1 = on_resonance_photon_number(epsilon=0.01, omega_n=1.0, t=10.0)
    N2 = on_resonance_photon_number(epsilon=0.01, omega_n=1.0, t=20.0)
    assert N2 > N1


def test_on_resonance_validates_negative():
    with pytest.raises(ValueError):
        on_resonance_photon_number(epsilon=-0.1, omega_n=1.0, t=1.0)


# ----- off-resonance flux ---------------------------------------------

def test_off_resonance_perturbative():
    """Detuning 10% of Omega_n, Q = 100, eps = 0.01:
        N = eps^2 / (Q^2 * detuning^2) = 1e-4 / (1e4 * 0.01) = 1e-6.
    """
    omega_n = 1.0
    Omega_drive = resonance_frequency_for_mode(omega_n) * 1.1  # 10% detuning
    N = off_resonance_photon_number(epsilon=0.01, Omega=Omega_drive,
                                      omega_n=omega_n, Q=100.0)
    expected = 0.01 ** 2 / (100.0 ** 2 * 0.1 ** 2)
    assert N == pytest.approx(expected, rel=1e-9)


def test_off_resonance_validates_Q():
    with pytest.raises(ValueError):
        off_resonance_photon_number(epsilon=0.01, Omega=1.0,
                                      omega_n=1.0, Q=0.0)


def test_off_resonance_at_exact_resonance_diverges():
    """Perturbative formula diverges if exactly on resonance."""
    omega_n = 1.0
    Omega_res = resonance_frequency_for_mode(omega_n)
    N = off_resonance_photon_number(epsilon=0.01, Omega=Omega_res,
                                      omega_n=omega_n, Q=100.0)
    assert np.isinf(N)


# ----- linewidth -------------------------------------------------------

def test_linewidth_finite():
    """Gamma = Omega_res / Q = 2 omega_n / Q."""
    Gamma = linewidth_at_resonance(omega_n=1.0, Q=100.0)
    assert Gamma > 0
    # Omega_res = 2 omega_n = 2, Q = 100 -> Gamma = 0.02
    assert Gamma == pytest.approx(2.0 / 100, rel=1e-12)


# ----- regime selection ------------------------------------------------

def test_dce_flux_estimate_picks_on_resonance():
    """Within linewidth: on_resonance regime."""
    omega_n = cavity_mode_frequency(n=1, d_0=1.0)
    Omega = resonance_frequency_for_mode(omega_n)  # exact resonance
    result = dce_flux_estimate(epsilon=0.01, Omega_drive=Omega, n=1, Q=100.0)
    assert result.on_resonance
    assert result.regime == "on_resonance"


def test_dce_flux_estimate_picks_off_resonance():
    """Far from linewidth: off_resonance regime."""
    omega_n = cavity_mode_frequency(n=1, d_0=1.0)
    Omega = resonance_frequency_for_mode(omega_n) * 1.5  # 50% detuning
    result = dce_flux_estimate(epsilon=0.01, Omega_drive=Omega, n=1, Q=100.0)
    assert not result.on_resonance
    assert result.regime == "off_resonance"


def test_dce_flux_result_fields():
    result = dce_flux_estimate(epsilon=0.01, Omega_drive=6.28, n=1)
    assert isinstance(result, DCEFluxResult)
    assert np.isfinite(result.photon_number)
    assert result.regime in ("on_resonance", "off_resonance")


# ----- spectrum sweep --------------------------------------------------

def test_spectrum_returns_arrays():
    spec = dce_spectrum(epsilon=0.01, Omega_min=5.0, Omega_max=10.0,
                          n_points=10, n_mode=1, Q=100.0, t=1.0)
    assert "Omega_drive" in spec
    assert "photon_number" in spec
    assert spec["Omega_drive"].shape == (10,)


def test_spectrum_peaks_at_resonance():
    """Photon number is much larger near Omega_res = 2 omega_1 than far away."""
    spec = dce_spectrum(epsilon=0.01, Omega_min=0.5 * np.pi,
                          Omega_max=3.5 * np.pi, n_points=100,
                          n_mode=1, d_0=1.0, Q=100.0, t=10.0)
    # peak should be near Omega_res = 2 pi
    idx_peak = np.argmax(spec["photon_number"])
    assert abs(spec["Omega_drive"][idx_peak] - 2 * np.pi) < 0.3
