"""Particle-creation rate tests."""

import numpy as np
import pytest

from systrophe import VanStockumInterior
from systrophe.particle_creation import (
    bose_einstein,
    fermi_dirac,
    particle_creation_spectrum_at_horizon,
    total_emission_power_proxy,
)


def test_fermi_dirac_at_T_zero_rejected():
    with pytest.raises(ValueError):
        fermi_dirac(1.0, T=0.0)
    with pytest.raises(ValueError):
        fermi_dirac(1.0, T=-1.0)


def test_fermi_dirac_below_unity():
    """Fermi-Dirac is bounded by 1."""
    omegas = np.array([0.1, 1.0, 10.0])
    occ = fermi_dirac(omegas, T=1.0)
    assert np.all(occ < 1.0)
    assert np.all(occ > 0.0)


def test_bose_einstein_diverges_at_zero_freq():
    """Bose-Einstein -> infinity as omega -> 0+."""
    occ = float(bose_einstein(1e-9, T=1.0))
    assert occ > 1e6


def test_high_freq_thermal_suppressed():
    """exp(-omega/T) suppression at high omega."""
    occ = float(fermi_dirac(100.0, T=1.0))
    assert occ < 1e-30


def test_horizon_spectrum_for_supercritical_vanstockum():
    """particle_creation_spectrum_at_horizon runs at a Cauchy horizon."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    from systrophe.quantum_diagnostics import cauchy_horizon_estimate
    horizons = cauchy_horizon_estimate(vs)
    assert horizons.size >= 1
    omega_array = np.linspace(0.1, 10.0, 50)
    out = particle_creation_spectrum_at_horizon(
        vs, float(horizons[0]), omega_array, statistics="fermion"
    )
    assert out["T_H"] > 0
    assert out["kappa"] > 0
    assert out["occupation"].shape == (50,)
    assert np.all(out["occupation"] > 0)
    assert np.all(out["occupation"] < 1)


def test_horizon_spectrum_unknown_statistics():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    from systrophe.quantum_diagnostics import cauchy_horizon_estimate
    horizons = cauchy_horizon_estimate(vs)
    omega_array = np.linspace(0.1, 1.0, 5)
    with pytest.raises(ValueError):
        particle_creation_spectrum_at_horizon(
            vs, float(horizons[0]), omega_array, statistics="anyon"
        )


def test_total_emission_power_finite():
    """Total emission power proxy is finite for finite omega_max."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    from systrophe.quantum_diagnostics import cauchy_horizon_estimate
    horizons = cauchy_horizon_estimate(vs)
    P = total_emission_power_proxy(vs, float(horizons[0]), statistics="fermion", omega_max=20.0)
    assert np.isfinite(P)
    assert P > 0
