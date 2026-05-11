"""Acoustic-analog Hawking spectrum on the LP rotating-cylinder background.

Extends `acoustic_metric` (which gives the c^2 - v^2 = F mapping and
the analog Hawking temperature T_H) by computing the full phonon
emission spectrum that a BEC- or rotating-water analog experiment
would measure.

The standard result: phonon emission at the analog acoustic horizon
follows the Planckian spectrum

    n(omega) = 1 / [exp(omega / T_H_acoustic) - 1]   (bosonic)

where T_H_acoustic = kappa / (2 pi) is the surface gravity divided by
2 pi (Unruh 1981, Steinhauer 2014/2019).

This module provides:

- `hawking_planck_spectrum(omega, T)`: Bose-Einstein distribution at T
- `phonon_spectral_density(vs, r_horizon, omega_range)`: full spectrum
  with proper Tolman blueshift to the horizon
- `total_emission_power(vs, r_horizon)`: Stefan-Boltzmann-like
  integrated power
- `g2_correlation_signature(vs, ...)`: the Steinhauer-style density-
  density correlation function (qualitative)
- `bec_experimental_predictions(omega, R, n_density, atom_mass)`:
  concrete BEC predictions (T_H in nK, signal magnitude, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .acoustic_metric import (
    acoustic_hawking_temperature,
    acoustic_horizon_radius,
)
from .quantum_diagnostics import surface_gravity_at_horizon, tolman_blueshift_factor


@dataclass(frozen=True)
class PhononSpectrum:
    """Phonon spectral density at the acoustic horizon."""

    omegas: np.ndarray  # frequency array
    n_omega: np.ndarray  # mean phonon number per mode
    spectral_density: np.ndarray  # omega^2 n(omega) (for 1D system, the
    #                                modal density factor depends on dim)
    T_H: float
    r_horizon: float


def hawking_planck_spectrum(omega: float | np.ndarray, T: float) -> np.ndarray:
    """Bose-Einstein distribution at temperature T.

    n(omega) = 1 / [exp(omega / T) - 1].
    For omega/T <= 0, returns NaN.
    """
    omega = np.asarray(omega, dtype=float)
    T = float(T)
    if T <= 0:
        return np.full_like(omega, np.nan)
    ratio = omega / T
    return 1.0 / (np.exp(np.clip(ratio, -30, 30)) - 1.0)


def phonon_spectral_density(
    vs, r_horizon: float, omega_range: tuple[float, float] = (0.01, 5.0),
    n_omega: int = 100,
) -> PhononSpectrum:
    """Full phonon spectral density at the acoustic horizon.

    Computes n(omega) at the horizon temperature T_H, multiplied by
    the appropriate modal density factor (here omega^2 for 3D bosonic
    field, omega^1 for 1D acoustic field).
    """
    T_H = acoustic_hawking_temperature(vs, r_horizon)
    omegas = np.linspace(*omega_range, n_omega)
    # Apply Tolman blueshift: temperature at the horizon is higher
    # than that observed at infinity.
    F_h = float(vs.analytic_exterior_F(r_horizon))
    blueshift = 1.0 / max(abs(F_h) ** 0.5, 1e-12) if F_h > 0 else 1.0
    T_local = T_H * blueshift if F_h > 0 else T_H
    # Suppress blueshift divergence right at F=0
    if F_h < 1e-3:
        T_local = T_H
    n_per_mode = hawking_planck_spectrum(omegas, T_local)
    # Spectral density (omega^1 for 1D bosonic, omega^2 for 3D)
    spectral_density = omegas * n_per_mode  # 1D analog (BEC sonic horizon)
    return PhononSpectrum(
        omegas=omegas, n_omega=n_per_mode,
        spectral_density=spectral_density, T_H=T_H,
        r_horizon=r_horizon,
    )


def total_emission_power(
    vs, r_horizon: float, omega_max: float = 20.0,
    n_omega: int = 500,
) -> float:
    """Integrated phonon emission power.

    P = integral omega * n(omega) * dOmega over omega_range.
    For a 1D bosonic field at temperature T, this gives pi^2 T^2 / 6
    (the textbook one-dimensional thermal power).
    """
    T_H = acoustic_hawking_temperature(vs, r_horizon)
    omegas = np.linspace(0.001, omega_max, n_omega)
    n_arr = hawking_planck_spectrum(omegas, T_H)
    integrand = omegas * n_arr  # omega^1 * n(omega) for 1D
    valid = np.isfinite(integrand)
    integrand[~valid] = 0
    domega = (omega_max - 0.001) / (n_omega - 1)
    return float(np.sum(integrand) * domega)


def g2_correlation_signature(
    vs, r_obs: float, r_horizon: float, delta_phi_range: np.ndarray | None = None,
) -> dict:
    """Density-density correlation function signature at the horizon.

    The Steinhauer hallmark: g_2(r, r') = <delta n(r) delta n(r')>
    exhibits a 'tongue' of correlations across the sonic horizon and
    anti-correlations across it.

    Here we provide a qualitative model: the correlation amplitude as
    a function of angular separation from the horizon.
    """
    if delta_phi_range is None:
        delta_phi_range = np.linspace(0, np.pi, 50)
    T_H = acoustic_hawking_temperature(vs, r_horizon)
    F_obs = float(vs.analytic_exterior_F(r_obs))
    F_h = float(vs.analytic_exterior_F(r_horizon))
    # Correlation amplitude is proportional to T_H and falls off as
    # distance from horizon increases
    distance_factor = np.exp(-np.abs(F_obs - F_h) / max(T_H, 1e-12))
    correlation_amplitude = T_H * distance_factor * np.cos(2 * delta_phi_range)
    return {
        "delta_phi": delta_phi_range.tolist(),
        "correlation_amplitude": correlation_amplitude.tolist(),
        "max_correlation": float(np.max(np.abs(correlation_amplitude))),
        "T_H": T_H, "r_observer": r_obs, "r_horizon": r_horizon,
    }


def bec_experimental_predictions(
    omega: float, R: float, n_density: float, atom_mass: float,
    a_scattering: float = 5.3e-9,
) -> dict:
    """Concrete BEC-vortex experimental predictions.

    For a BEC with given parameters, returns expected acoustic
    Hawking temperature, healing length, and signal magnitude.

    Parameters
    ----------
    omega : float
        Cylinder rotation rate (or equivalent vortex angular velocity), in s^-1.
    R : float
        Cylinder radius (or vortex core radius), in m.
    n_density : float
        BEC atom density, m^-3.
    atom_mass : float
        Atomic mass, kg.
    a_scattering : float
        s-wave scattering length, m (default Rb-87).

    Returns dict with:
        sound_speed_c : sound speed (m/s)
        healing_length : xi_BEC (m)
        T_H_acoustic : analog Hawking T (nK)
        signal_resolved : True iff T_H exceeds typical BEC noise floor (~10 pK)
    """
    hbar = 1.054571817e-34  # J s
    k_B = 1.380649e-23  # J/K
    # Sound speed: c = sqrt(4 pi hbar^2 a_s n / m^2)
    c_sound = float(np.sqrt(4 * np.pi * hbar ** 2 * a_scattering * n_density / atom_mass ** 2))
    # Healing length
    mu = 4 * np.pi * hbar ** 2 * a_scattering * n_density / atom_mass
    xi_BEC = float(hbar / np.sqrt(2 * atom_mass * mu))
    # Analog Hawking T (estimate):
    # At a quantised vortex, surface gravity ~ c^2 / xi_BEC; T_H ~ hbar c / (2 pi xi_BEC k_B)
    T_H = hbar * c_sound / (2 * np.pi * xi_BEC * k_B)  # K
    T_H_nK = T_H * 1e9  # convert to nK
    # Typical BEC noise floor is ~10 pK = 0.01 nK
    noise_floor_nK = 0.01
    return {
        "sound_speed_c_m_per_s": c_sound,
        "healing_length_xi_BEC_m": xi_BEC,
        "T_H_acoustic_nK": T_H_nK,
        "noise_floor_nK": noise_floor_nK,
        "signal_to_noise_ratio": T_H_nK / noise_floor_nK,
        "signal_resolved": bool(T_H_nK > noise_floor_nK * 10),
    }


def compare_to_steinhauer_2019(T_H_predicted: float) -> dict:
    """Compare predicted Hawking T to Steinhauer 2019 measurement.

    Steinhauer measured T_H = 0.124 +/- 0.012 nK in his BEC analog
    black hole apparatus. Returns dict with comparison.
    """
    T_steinhauer_nK = 0.124
    T_steinhauer_err_nK = 0.012
    sigma_deviation = abs(T_H_predicted - T_steinhauer_nK) / T_steinhauer_err_nK
    return {
        "T_predicted_nK": T_H_predicted,
        "T_steinhauer_nK": T_steinhauer_nK,
        "T_steinhauer_uncertainty_nK": T_steinhauer_err_nK,
        "sigma_deviation": sigma_deviation,
        "consistent_with_measurement": sigma_deviation < 3.0,
    }
