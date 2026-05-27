"""Particle-creation rates at Tipler chronology horizons.

Every Cauchy-horizon candidate F(r_h) = 0 has a surface gravity
kappa = (1/2) |F'(r_h)| (see `quantum_diagnostics.surface_gravity_at_horizon`).
For a stationary horizon, the analog of Hawking's calculation gives
a thermal spectrum at temperature

    T_H = kappa / (2 pi).

For a free scalar in 4D the occupation number of the outgoing modes is

    n_omega = 1 / (exp(omega / T_H) - 1)         [bosons],

while for a Dirac field

    n_omega = 1 / (exp(omega / T_H) + 1)         [fermions].

This module exposes:

- `bose_einstein(omega, T)`: Bose-Einstein occupation.
- `fermi_dirac(omega, T)`: Fermi-Dirac occupation.
- `particle_creation_spectrum_at_horizon(vs, r_horizon, omega_array, statistics)`:
  Hawking-like thermal occupation at each Cauchy-horizon candidate.
- `total_emission_power_proxy(vs, r_horizon, statistics, omega_max)`:
  proxy for the integrated power emitted; diverges if the horizon
  is reached (since Tipler is non-asymptotically-flat, the "emitted
  to infinity" interpretation is heuristic).

Caveats
-------
- The Hawking calculation assumes an asymptotically flat outer region;
  the Tipler exterior is *not* asymptotically flat. The thermal
  spectrum should be read as the locally-stationary-frame occupation
  number of the Cauchy horizon's analog of the future event horizon.
- Quantum back-reaction at higher order modifies the spectrum and
  is not modelled here.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


def bose_einstein(omega: float | np.ndarray, T: float) -> np.ndarray:
    """Bose-Einstein occupation 1 / (exp(omega/T) - 1)."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    omega_arr = np.asarray(omega, dtype=float)
    return 1.0 / (np.exp(omega_arr / T) - 1.0)


def fermi_dirac(omega: float | np.ndarray, T: float) -> np.ndarray:
    """Fermi-Dirac occupation 1 / (exp(omega/T) + 1)."""
    if T <= 0.0:
        raise ValueError("T must be positive")
    omega_arr = np.asarray(omega, dtype=float)
    return 1.0 / (np.exp(omega_arr / T) + 1.0)


def particle_creation_spectrum_at_horizon(
    vs,
    r_horizon: float,
    omega_array: np.ndarray,
    statistics: str = "fermion",
) -> dict:
    """Thermal occupation spectrum at a Cauchy horizon.

    Parameters
    ----------
    vs : VanStockumInterior
        The cylinder defining the Tipler exterior.
    r_horizon : float
        Cauchy-horizon candidate radius (one of the F = 0 roots).
    omega_array : np.ndarray
        Frequencies at which to sample the occupation spectrum.
    statistics : str
        'fermion' for Fermi-Dirac, 'boson' for Bose-Einstein.

    Returns
    -------
    dict with 'omega', 'occupation', 'T_H', 'kappa'.
    """
    from systrophe.qftcs.quantum_diagnostics import surface_gravity_at_horizon

    kappa = surface_gravity_at_horizon(vs, r_horizon)
    T_H = kappa / (2.0 * np.pi)
    if statistics == "fermion":
        occ = fermi_dirac(omega_array, T_H)
    elif statistics == "boson":
        occ = bose_einstein(omega_array, T_H)
    else:
        raise ValueError(f"unknown statistics '{statistics}'; use 'fermion' or 'boson'")
    return {
        "omega": np.asarray(omega_array, dtype=float),
        "occupation": occ,
        "T_H": T_H,
        "kappa": kappa,
    }


def total_emission_power_proxy(
    vs,
    r_horizon: float,
    statistics: str = "fermion",
    omega_max: float = 100.0,
) -> float:
    """Integrated power proxy: integral of omega * n(omega) d omega from 0 to omega_max.

    For a single bosonic mode this gives Stefan-Boltzmann-like scaling
    proportional to T^4 (with appropriate factors from spectral
    weight). Truncated by omega_max to avoid the formal divergence.
    """
    from systrophe.qftcs.quantum_diagnostics import surface_gravity_at_horizon

    kappa = surface_gravity_at_horizon(vs, r_horizon)
    T_H = kappa / (2.0 * np.pi)

    if statistics == "fermion":
        f = lambda omega: omega / (np.exp(omega / T_H) + 1.0)
    elif statistics == "boson":
        f = lambda omega: omega / (np.exp(omega / T_H) - 1.0) if omega > 0 else 0.0
    else:
        raise ValueError(f"unknown statistics '{statistics}'")

    eps = 1e-6 * T_H
    result, _ = quad(f, eps, omega_max, limit=100)
    return float(result)
