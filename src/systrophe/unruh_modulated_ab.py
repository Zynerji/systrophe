"""Unruh-modulated Aharonov-Bohm fringes.

Combines three existing modules to produce *observer-dependent* fringe
visibility on the van Stockum exterior:

- ``aharonov_bohm_ctc`` for the path-difference AB phase
  phi_AB(r_loop) = 2 pi K(r_loop).
- ``which_way_ancilla`` for the intentional which-way coupling
  kappa in [0, 1] -> V_pure(kappa) = cos(pi kappa / 2).
- ``unruh_effect`` for the Unruh temperature seen by an observer
  in a circular orbit at radius r_obs and angular velocity Omega_obs.

The thermal bath at T_U then *dephases* the path-qubit at a standard
Lindblad-style thermal rate
    Gamma_th(T_U, omega) = gamma_0 * (2 nbar(omega, T_U) + 1)
giving an exponential coherence factor over an interaction time tau,
    eta_thermal(T_U) = exp(-gamma_tau * 2 nbar(omega, T_U))
(the spontaneous-emission piece "+ 1" is folded into the bare coupling
and reabsorbed; the thermal piece is what carries the Unruh signature).

The total visibility is the product
    V_total = V_pure(kappa) * eta_thermal(T_U).

Limits
------
- inertial observer (T_U = 0): V_total = V_pure(kappa) -- recovers the
  ordinary Englert duality of ``which_way_ancilla``.
- no which-way coupling (kappa = 0): V_total = eta_thermal(T_U) -- pure
  Unruh decoherence of the interferometer.
- both off: V_total = 1; both on: V_total -> 0.

This is a phenomenological combination, not a derivation of either
Englert or the Davies-Unruh decoherence rate from first principles; it
makes the *combined* observer-dependent visibility concrete enough to
plug into a fringe pattern and compare across observers.
"""

from __future__ import annotations

import math

import numpy as np

from .unruh_effect import (
    centripetal_acceleration_at_orbit,
    unruh_temperature,
)
from .vanstockum import VanStockumInterior
from .which_way_ancilla import path_coherence


def bose_einstein(omega: float, T: float) -> float:
    """nbar(omega, T) = 1 / (exp(omega/T) - 1). nbar(omega, 0) = 0."""
    if omega <= 0:
        raise ValueError(f"omega must be positive, got {omega}")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if T == 0.0:
        return 0.0
    x = omega / T
    if x > 700.0:           # protect against overflow
        return 0.0
    return float(1.0 / (math.exp(x) - 1.0))


def thermal_decoherence(
    T_U: float, omega: float = 1.0, gamma_tau: float = 1.0,
) -> float:
    """eta_thermal = exp(-gamma_tau * 2 nbar(omega, T_U)).

    Parameters
    ----------
    T_U : float
        Unruh temperature (>= 0).
    omega : float
        Effective transition frequency of the interferometer mode.
    gamma_tau : float
        Dimensionless coupling x interaction-time product (>= 0).
    """
    if gamma_tau < 0:
        raise ValueError(f"gamma_tau must be non-negative, got {gamma_tau}")
    if not math.isfinite(T_U):
        return 0.0
    nbar = bose_einstein(omega, T_U)
    return float(math.exp(-gamma_tau * 2.0 * nbar))


def observer_unruh_temperature(
    vs: VanStockumInterior, r_observer: float, angular_velocity: float = 1.0,
) -> float:
    """Wrap unruh_effect for a circular orbit at (r_observer, Omega)."""
    a_acc = centripetal_acceleration_at_orbit(
        vs, r_observer, angular_velocity=angular_velocity,
    )
    if not math.isfinite(a_acc):
        return float("inf")
    return unruh_temperature(a_acc)


def total_visibility(
    coupling: float, T_U: float, omega: float = 1.0, gamma_tau: float = 1.0,
) -> float:
    """V_total = V_pure(coupling) * eta_thermal(T_U).

    Returns
    -------
    float in [0, 1].
    """
    V_pure = path_coherence(coupling)
    eta = thermal_decoherence(T_U, omega=omega, gamma_tau=gamma_tau)
    return float(V_pure * eta)


def unruh_ab_intensity(
    y: np.ndarray, k: float, d: float, D_arm: float,
    phi_ab: float, coupling: float, T_U: float,
    omega: float = 1.0, gamma_tau: float = 1.0,
) -> np.ndarray:
    """Detector intensity I(y) = 2 + 2 V_total cos(k dL(y) + phi_AB)."""
    V = total_visibility(
        coupling, T_U, omega=omega, gamma_tau=gamma_tau,
    )
    L_U = np.sqrt(D_arm * D_arm + d * d) + np.sqrt(D_arm * D_arm + (y - d) ** 2)
    L_L = np.sqrt(D_arm * D_arm + d * d) + np.sqrt(D_arm * D_arm + (y + d) ** 2)
    return 2.0 + 2.0 * V * np.cos(k * (L_U - L_L) + phi_ab)


def sweep_observer_radii(
    vs: VanStockumInterior, r_grid: np.ndarray,
    coupling: float = 0.0, angular_velocity: float = 1.0,
    omega: float = 1.0, gamma_tau: float = 1.0,
) -> dict:
    """For each r_observer, return T_U(r) and V_total(kappa, T_U(r))."""
    Ts = np.array([
        observer_unruh_temperature(vs, float(r), angular_velocity)
        for r in r_grid
    ])
    Vs = np.array([
        total_visibility(coupling, float(T), omega=omega, gamma_tau=gamma_tau)
        for T in Ts
    ])
    return {"r_observer": r_grid, "T_U": Ts, "V_total": Vs}
