"""Dirac-sea structure near the chronology horizon.

The Dirac sea is the (formally infinite) collection of all negative-
energy states; the physical vacuum is its filled ground state. On a
stationary curved background the local energy of a static observer is
Tolman-shifted by 1/sqrt(F):

    E_local(r) = E_infty / sqrt(F(r)).

Near a Cauchy horizon (F -> 0), the local energy diverges, packing
arbitrarily many states per unit local frequency into a vanishing
proper volume. This is the Dirac-sea-level signature of the
chronology-protection mechanism: any sane filling of negative-energy
states would have divergent stress-energy at the chronology horizon.

This module provides:

- `local_energy(F, E_infty)`: Tolman-shifted local energy.
- `density_of_states_radial(F, m, k, mass, E)`: leading-order radial
  level density of the Dirac sea, scaling as 1/sqrt(F) near the
  chronology horizon.
- `dirac_sea_pressure_proxy(vs, r)`: dimensional-analysis proxy for the
  divergent vacuum pressure as a function of radius. Diverges as F^(-2).

Honest scope
------------
These are *qualitative* indicators. A genuine Dirac-sea calculation
requires (a) choosing a vacuum state (Boulware, Hartle-Hawking, Unruh
analog), (b) point-splitting renormalisation of the two-point
function, (c) summing over modes. We provide the leading-order classical
proxies that the full calculation would refine.
"""

from __future__ import annotations

import numpy as np


def local_energy(F: float | np.ndarray, E_infty: float) -> np.ndarray:
    """Tolman-shifted local energy E_local = E_infty / sqrt(F).

    Returns inf where F <= 0 (Cauchy horizon and beyond).
    """
    F_arr = np.asarray(F, dtype=float)
    return np.where(
        F_arr > 0,
        E_infty / np.sqrt(np.maximum(F_arr, 1e-300)),
        np.inf,
    )


def density_of_states_radial(
    F: float | np.ndarray,
    m: int,
    k: float,
    mass: float,
    E_infty: float,
) -> np.ndarray:
    """Leading-order radial level density of the Dirac sea.

    For a free Dirac field on a flat radial slice, the level density
    per unit local frequency is approximately rho_0(omega_local) =
    (1 / pi) sqrt(omega_local^2 - mass^2 - (m^2 + k^2)). On the Tipler
    background, replace omega_local = E_infty / sqrt(F):

        rho(F) = (1 / pi) sqrt( (E_infty / sqrt(F))^2 - mass^2 - (m^2 + k^2) ).

    Diverges as 1/sqrt(F) for F -> 0.
    """
    F_arr = np.asarray(F, dtype=float)
    safe = np.where(F_arr > 1e-30, F_arr, np.inf)
    omega_local_sq = (E_infty * E_infty) / safe
    inner = np.maximum(omega_local_sq - mass * mass - (m * m + k * k), 0.0)
    return np.sqrt(inner) / np.pi


def dirac_sea_pressure_proxy(vs, r: float | np.ndarray) -> np.ndarray:
    """Heuristic Dirac-sea pressure proxy: scales as 1/F^2.

    Motivated by Tolman temperature T_local = T_infty / sqrt(F) and
    the Stefan-Boltzmann fourth-power scaling. The vacuum pressure of
    a thermal Dirac sea at local temperature T_local is proportional
    to T_local^4 = T_infty^4 / F^2; we use the F^(-2) form as the
    chronology-protection proxy.

    Returns 1 / F(r)^2 for the analytic Tipler exterior of `vs`.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    F_arr = np.asarray(vs.analytic_exterior_F(r), dtype=float)
    safe = np.where(np.abs(F_arr) > 1e-30, F_arr, np.inf)
    return 1.0 / (safe * safe)


def chronology_horizon_pressure_divergence_rate(vs, r_horizon: float, eps: float = 1e-6) -> float:
    """Power-law divergence rate of the Dirac-sea pressure proxy near a Cauchy horizon.

    For F(r) ~ F'(r_h) (r - r_h) near the horizon, P_proxy ~ 1/F^2 ~
    1/(r - r_h)^2. The "divergence rate" is the power 2 of (r - r_h)^(-1)
    in this scaling. We return the numerical exponent for a finite-difference
    sample around the horizon: log(P(r_h + eps)/P(r_h + 2*eps)) / log(2).
    """
    p1 = dirac_sea_pressure_proxy(vs, r_horizon + eps)
    p2 = dirac_sea_pressure_proxy(vs, r_horizon + 2.0 * eps)
    return float(np.log(float(p1) / float(p2)) / np.log(2.0))
