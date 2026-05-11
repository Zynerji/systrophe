"""Floquet analysis of a periodically-driven Systrophe pair.

For a co-rotating SystrophePair whose relative phase offset varies
periodically in time,

    delta(t) = delta_0 + delta_amp * sin(Omega_drive * t),

the joint L envelope L_pair(r, t) becomes a time-periodic function with
period T = 2 pi / Omega_drive. The radial Dirac (or scalar) equation
inherits this periodicity and admits Floquet solutions

    psi(t, r) = e^{-i eps t} u(t, r),     u(t + T, r) = u(t, r),

with quasi-energy eps defined modulo Omega_drive.

This module implements a one-period time-evolution-operator approach:

  1. Build the radial Dirac system at fixed (E, m, k, mass) on a grid r.
  2. The "Hamiltonian" matrix at each r depends on (F, K, L) and hence
     on delta(t). At fixed r the effective 2-spinor matrix is time-
     periodic with period T.
  3. Compute U_T(r) = T-ordered exp(-i integral_0^T H(t, r) dt) via
     small time-step product expansion.
  4. Diagonalise U_T(r) -> eigenvalues lambda_+- = exp(-i eps_+- T)
     -> Floquet quasi-energies eps_+-(r).

The Floquet quasi-energies as a function of r are the analog of the
band structure of a time-crystal modulation of the Tipler exterior.
Avoided crossings between Floquet bands signal parametric resonance,
the regime in which a driven CTC band could in principle pump energy
into the radial spinor sea.

References
----------
- J. H. Shirley, Phys. Rev. 138 (1965) B979 (original Floquet QM).
- N. Goldman and J. Dalibard, Phys. Rev. X 4 (2014) 031027 (modern
  Floquet engineering review).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class FloquetSpectrum:
    """Floquet quasi-energy spectrum on a radial grid.

    Attributes
    ----------
    r : np.ndarray
        Radial sample points.
    quasi_energies : np.ndarray
        Shape (len(r), 2): two Floquet quasi-energies per radius (the
        radial Dirac system is 2 x 2).
    period : float
        T = 2 pi / Omega_drive.
    omega_drive : float
    delta_0 : float
    delta_amp : float
    """

    r: np.ndarray
    quasi_energies: np.ndarray
    period: float
    omega_drive: float
    delta_0: float
    delta_amp: float


def _effective_hamiltonian(
    F: float, K: float, L: float, h: float,
    E: float, m: int, k: float, mass: float,
) -> np.ndarray:
    """Local Hermitian 2 x 2 effective Hamiltonian H_eff(r, t) for Floquet evolution.

    We use a toy two-level Hamiltonian capturing the essential structure
    of the radial Dirac problem on a time-periodic LP background:

        H_eff = epsilon(r) sigma_z + V(r, t) sigma_x,

    with epsilon = sqrt(max(F, 0)) * sqrt(mass^2 + m^2/(max(L, eps)) + k^2)
    (mass-shell energy of a localised mode at radius r), and V the
    "Tipler potential" coupling V = (E sqrt(F) - m K / sqrt(F)).

    H_eff is real-symmetric (hence Hermitian), so the time-evolution
    operator exp(-i H dt) is unitary. The Floquet quasi-energies
    extracted from U(T) are real and bounded mod 2 pi / T.
    """
    sqrt_F = np.sqrt(max(F, 1e-300)) if F > 0 else np.sqrt(max(-F, 1e-300))
    sqrt_h = np.sqrt(max(h, 1e-300))
    L_safe = max(abs(L), 1e-9)
    epsilon = sqrt_F * np.sqrt(mass * mass + (m * m) / L_safe + k * k)
    V = E * sqrt_F - m * K / max(sqrt_F, 1e-9)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    return epsilon * sigma_z + V * sigma_x


def _pair_L_at(r: float, omega: float, R: float, delta: float) -> float:
    """L_pair(r, t) via the standard log-shift approximation L_1(r) + L_2(r e^{-delta/alpha})."""
    from .vanstockum import VanStockumInterior

    vs = VanStockumInterior(omega=omega, R=R)
    alpha = vs.alpha
    L1 = float(vs.analytic_exterior_L(r))
    L2 = float(vs.analytic_exterior_L(r * np.exp(-delta / alpha)))
    return L1 + L2


def _pair_F_at(r: float, omega: float, R: float, delta: float) -> float:
    """F_pair(r, delta) approximated as the F-sum of the two cylinders (linear)."""
    from .vanstockum import VanStockumInterior

    vs = VanStockumInterior(omega=omega, R=R)
    alpha = vs.alpha
    F1 = float(vs.analytic_exterior_F(r))
    F2 = float(vs.analytic_exterior_F(r * np.exp(-delta / alpha)))
    return F1 + F2


def _pair_K_at(r: float, omega: float, R: float, delta: float) -> float:
    """K_pair(r, delta) approximated linearly."""
    from .vanstockum import VanStockumInterior

    vs = VanStockumInterior(omega=omega, R=R)
    alpha = vs.alpha
    K1 = float(vs.analytic_exterior_K(r))
    K2 = float(vs.analytic_exterior_K(r * np.exp(-delta / alpha)))
    return K1 + K2


def time_evolution_operator_at_r(
    r: float,
    omega: float,
    R: float,
    delta_0: float,
    delta_amp: float,
    omega_drive: float,
    E: float,
    m: int,
    k: float,
    mass: float,
    n_substeps: int = 200,
) -> np.ndarray:
    """U(T) at fixed r: one-period Floquet propagator of the local 2 x 2 system.

    Uses a product-of-Suzuki-Trotter-step approach: divide T into
    n_substeps intervals, build H(t_i) at the midpoint of each, and
    multiply U = exp(-i H_n dt) ... exp(-i H_1 dt).
    """
    T = 2.0 * np.pi / omega_drive
    dt = T / n_substeps
    U = np.eye(2, dtype=complex)
    for i in range(n_substeps):
        t_mid = (i + 0.5) * dt
        delta_t = delta_0 + delta_amp * np.sin(omega_drive * t_mid)
        F = _pair_F_at(r, omega, R, delta_t)
        K = _pair_K_at(r, omega, R, delta_t)
        L = _pair_L_at(r, omega, R, delta_t)
        h = 1.0
        H = _effective_hamiltonian(F, K, L, h, E, m, k, mass)
        U = expm(-1j * H * dt) @ U
    return U


def floquet_quasi_energies_at_r(
    r: float,
    omega: float,
    R: float,
    delta_0: float,
    delta_amp: float,
    omega_drive: float,
    E: float,
    m: int,
    k: float,
    mass: float,
    n_substeps: int = 200,
) -> tuple[float, float]:
    """Return the two Floquet quasi-energies (epsilon_+, epsilon_-) at radius r.

    From U(T) eigenvalues lambda = exp(-i epsilon T):
        epsilon = i log(lambda) / T  (taking the principal branch).
    Quasi-energies are defined modulo Omega_drive.
    """
    U = time_evolution_operator_at_r(
        r, omega, R, delta_0, delta_amp, omega_drive,
        E, m, k, mass, n_substeps,
    )
    eigvals = np.linalg.eigvals(U)
    T = 2.0 * np.pi / omega_drive
    # epsilon = -arg(lambda) / T
    epsilons = -np.angle(eigvals) / T
    epsilons = np.sort(epsilons.real)
    return float(epsilons[0]), float(epsilons[1])


def compute_floquet_spectrum(
    omega: float,
    R: float,
    delta_0: float,
    delta_amp: float,
    omega_drive: float,
    r_grid: np.ndarray,
    E: float = 1.0,
    m: int = 0,
    k: float = 0.0,
    mass: float = 0.5,
    n_substeps: int = 200,
) -> FloquetSpectrum:
    """Compute the Floquet quasi-energy band structure across r.

    For each r in r_grid, build the one-period propagator U(T) and
    diagonalise. Return both Floquet bands as a function of r.

    Note: convergence requires n_substeps >> Omega_drive * (variation
    scale of H per step); for default omega_drive = 1, n_substeps = 200
    is adequate at moderate amplitudes.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    n = len(r_grid)
    eps_array = np.zeros((n, 2), dtype=float)
    for i, r in enumerate(r_grid):
        eps_array[i] = floquet_quasi_energies_at_r(
            float(r), omega, R, delta_0, delta_amp, omega_drive,
            E, m, k, mass, n_substeps,
        )
    return FloquetSpectrum(
        r=r_grid,
        quasi_energies=eps_array,
        period=2.0 * np.pi / omega_drive,
        omega_drive=omega_drive,
        delta_0=delta_0,
        delta_amp=delta_amp,
    )


def detect_parametric_resonance(spectrum: FloquetSpectrum, gap_threshold: float = 1e-3) -> list[tuple[int, float, float]]:
    """Detect avoided crossings between the two Floquet bands.

    Returns a list of (index, r, gap) tuples where the band gap
    abs(epsilon_+ - epsilon_-) dips below `gap_threshold`. These are
    candidate parametric-resonance points.
    """
    gaps = np.abs(spectrum.quasi_energies[:, 1] - spectrum.quasi_energies[:, 0])
    resonances = []
    for i in range(1, len(gaps) - 1):
        if gaps[i] < gap_threshold and gaps[i] <= gaps[i - 1] and gaps[i] <= gaps[i + 1]:
            resonances.append((int(i), float(spectrum.r[i]), float(gaps[i])))
    return resonances
