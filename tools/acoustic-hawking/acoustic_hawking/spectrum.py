"""Phonon spectrum + total emission power wrapper."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.analogs.acoustic_hawking_spectrum import (
    PhononSpectrum,
    phonon_spectral_density,
    total_emission_power,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class SpectrumReport:
    """Phonon spectrum + integrated emission power at an acoustic horizon."""
    horizon_r: float
    T_hawking: float
    omegas: np.ndarray
    mean_phonon_number: np.ndarray
    spectral_density: np.ndarray
    total_emission_power: float


def compute_phonon_spectrum(
    omega: float, R: float, r_horizon: float,
    omega_range: tuple[float, float] = (0.01, 5.0),
    n_omega: int = 100,
    omega_max_power: float = 20.0,
    n_omega_power: int = 500,
) -> SpectrumReport:
    """One-shot phonon spectrum + total emission power.

    Parameters
    ----------
    omega, R : van Stockum cylinder parameters.
    r_horizon : acoustic horizon radius (from
        AnalogHorizonAnalyser.horizon()).
    omega_range : (omega_min, omega_max) for the spectral density.
    n_omega : number of points in the spectral grid.
    omega_max_power : integration cutoff for total emission power.
    n_omega_power : number of points for the power integration.
    """
    vs = VanStockumInterior(omega=float(omega), R=float(R))
    spec: PhononSpectrum = phonon_spectral_density(
        vs, r_horizon=float(r_horizon),
        omega_range=omega_range, n_omega=int(n_omega),
    )
    P = total_emission_power(
        vs, r_horizon=float(r_horizon),
        omega_max=float(omega_max_power), n_omega=int(n_omega_power),
    )
    return SpectrumReport(
        horizon_r=float(r_horizon),
        T_hawking=float(spec.T_H),
        omegas=np.asarray(spec.omegas),
        mean_phonon_number=np.asarray(spec.n_omega),
        spectral_density=np.asarray(spec.spectral_density),
        total_emission_power=float(P),
    )
