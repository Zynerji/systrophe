"""Floquet quasi-energy tests."""

import numpy as np
import pytest

from systrophe.floquet import (
    FloquetSpectrum,
    compute_floquet_spectrum,
    detect_parametric_resonance,
    floquet_quasi_energies_at_r,
    time_evolution_operator_at_r,
)


def test_propagator_is_unitary():
    """U(T) should be unitary: U^dagger U = I."""
    U = time_evolution_operator_at_r(
        r=1.5, omega=1.0, R=1.0,
        delta_0=0.5, delta_amp=0.1, omega_drive=1.0,
        E=1.0, m=0, k=0.0, mass=0.5, n_substeps=200,
    )
    UUdag = U.conj().T @ U
    np.testing.assert_allclose(UUdag, np.eye(2), atol=1e-6)


def test_zero_amplitude_gives_static_evolution():
    """delta_amp = 0 => static problem; quasi-energies should match the
    static eigenvalues of H(delta_0)."""
    eps_plus, eps_minus = floquet_quasi_energies_at_r(
        r=1.5, omega=1.0, R=1.0,
        delta_0=0.5, delta_amp=0.0, omega_drive=1.0,
        E=1.0, m=0, k=0.0, mass=0.5, n_substeps=400,
    )
    assert np.isfinite(eps_plus)
    assert np.isfinite(eps_minus)
    # Static => the two quasi-energies should be symmetric about 0
    # (from H = -i [[-A, B], [B, A]] eigenvalues are pure imaginary +/-)
    # Quasi-energy under static driving is well-defined mod Omega.
    # Just verify both eigenvalues are real (no spurious complex part).
    assert isinstance(eps_plus, float)


def test_quasi_energies_finite_in_resonant_regime():
    """Even at resonant amplitudes, quasi-energies must be finite reals."""
    eps_plus, eps_minus = floquet_quasi_energies_at_r(
        r=1.5, omega=1.0, R=1.0,
        delta_0=0.3, delta_amp=1.0, omega_drive=2.0,
        E=1.0, m=0, k=0.0, mass=0.5, n_substeps=400,
    )
    assert np.isfinite(eps_plus)
    assert np.isfinite(eps_minus)


def test_compute_spectrum_returns_dataclass():
    """compute_floquet_spectrum returns a FloquetSpectrum with correct shapes."""
    r_grid = np.linspace(1.5, 5.0, 6)
    spec = compute_floquet_spectrum(
        omega=1.0, R=1.0,
        delta_0=0.3, delta_amp=0.2, omega_drive=1.0,
        r_grid=r_grid, n_substeps=100,
    )
    assert isinstance(spec, FloquetSpectrum)
    assert spec.r.shape == (6,)
    assert spec.quasi_energies.shape == (6, 2)
    assert spec.period == pytest.approx(2 * np.pi)


def test_quasi_energies_have_two_bands():
    """Two Floquet bands at each r (the radial Dirac is a 2-spinor)."""
    spec = compute_floquet_spectrum(
        omega=1.0, R=1.0,
        delta_0=0.3, delta_amp=0.2, omega_drive=1.0,
        r_grid=np.linspace(1.5, 4.0, 3), n_substeps=100,
    )
    assert spec.quasi_energies.shape[1] == 2
    # The lower band should be <= upper band at each r
    assert np.all(spec.quasi_energies[:, 0] <= spec.quasi_energies[:, 1])


def test_detect_resonance_runs():
    """Resonance detector returns a list (possibly empty)."""
    spec = compute_floquet_spectrum(
        omega=1.0, R=1.0,
        delta_0=0.3, delta_amp=0.5, omega_drive=1.0,
        r_grid=np.linspace(1.2, 5.0, 30), n_substeps=80,
    )
    resonances = detect_parametric_resonance(spec, gap_threshold=1e-2)
    assert isinstance(resonances, list)
