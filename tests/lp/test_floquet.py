"""Adiabatic Floquet tests (research-grade replacement of the v0.10.0 toy)."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.lp.floquet import (
    AdiabaticFloquetSpectrum,
    adiabatic_floquet_spectrum,
    adiabatic_floquet_validity,
    static_pair_bound_states,
)


def test_static_pair_bound_states_returns_array():
    """static_pair_bound_states returns an ndarray (possibly empty)."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    energies = static_pair_bound_states(cyl, delta=0.0, m=0, k=0, mass=0.5,
                                        r_min=1.1, r_max=3.0, n_E=40)
    assert isinstance(energies, np.ndarray)


def test_adiabatic_floquet_returns_dataclass():
    """adiabatic_floquet_spectrum returns AdiabaticFloquetSpectrum."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    spec = adiabatic_floquet_spectrum(
        cyl, delta_0=0.3, delta_amp=0.0, omega_drive=0.05,
        m=0, k=0.0, mass=0.5, n_t=4,
    )
    assert isinstance(spec, AdiabaticFloquetSpectrum)
    assert spec.t_samples.shape == (4,)
    assert spec.delta_samples.shape == (4,)
    assert spec.period == pytest.approx(2 * np.pi / 0.05)


def test_static_limit_matches_instantaneous_spectrum():
    """delta_amp = 0 -> Floquet quasi-energies equal static eigenvalues
    (modulo the Brillouin-zone wrap)."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    delta_0 = 0.3
    static_E = static_pair_bound_states(cyl, delta=delta_0)
    spec = adiabatic_floquet_spectrum(
        cyl, delta_0=delta_0, delta_amp=0.0, omega_drive=10.0,
        n_t=4,
    )
    # quasi_energies in Brillouin zone [-omega/2, omega/2); static_E is
    # the absolute spectrum. The fundamental Brillouin zone of
    # omega = 10 is [-5, 5], so static eigenvalues in this range
    # should match (up to ordering).
    if spec.quasi_energies.size > 0 and len(static_E) > 0:
        n = min(len(spec.quasi_energies), len(static_E))
        # Wrap static_E to BZ
        static_wrapped = np.sort(((static_E + 5.0) % 10.0) - 5.0)[:n]
        quasi_sorted = np.sort(spec.quasi_energies)[:n]
        np.testing.assert_allclose(quasi_sorted, static_wrapped, rtol=1e-3, atol=1e-3)


def test_quasi_energies_bounded_by_brillouin_zone():
    """Floquet quasi-energies must lie in [-omega/2, omega/2)."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    omega_drive = 0.5
    spec = adiabatic_floquet_spectrum(
        cyl, delta_0=0.3, delta_amp=0.1, omega_drive=omega_drive,
        n_t=6,
    )
    eps = spec.quasi_energies
    if eps.size > 0:
        assert np.all(eps >= -omega_drive / 2.0 - 1e-12)
        assert np.all(eps < omega_drive / 2.0 + 1e-12)


def test_validity_diagnostic_runs():
    """adiabatic_floquet_validity returns a dict with the expected keys."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    out = adiabatic_floquet_validity(
        cyl, delta_0=0.3, delta_amp=0.1, omega_drive=0.05,
    )
    assert "static_eigenvalues" in out
    assert "min_gap" in out
    assert "omega_drive" in out
    assert "omega_over_gap" in out
    assert "adiabatic_valid" in out


def test_real_metric_real_eigenvalues():
    """The radial Dirac on a real metric has real eigenvalues; verify."""
    cyl = VanStockumInterior(omega=1.0, R=1.0)
    E_n = static_pair_bound_states(cyl, delta=0.3, n_E=30)
    if len(E_n) > 0:
        assert np.all(np.isreal(E_n))
        assert np.all(np.isfinite(E_n))
