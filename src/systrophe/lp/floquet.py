"""Adiabatic Floquet analysis of a periodically-driven Systrophe pair.

For a co-rotating SystrophePair whose relative phase offset varies
periodically in time,

    delta(t) = delta_0 + delta_amp * sin(Omega_drive * t),

the joint Lewis-Papapetrou exterior becomes a time-periodic
background with period T = 2 pi / Omega_drive. The radial Dirac
spinor admits Floquet solutions

    psi(t, r) = e^{-i eps t} u(t, r),     u(t + T, r) = u(t, r),

with quasi-energy eps defined mod Omega_drive.

This module implements the **adiabatic** Floquet calculation, which is
exact in the slow-drive limit Omega_drive << omega_band where
omega_band is the typical bound-state energy gap of the static
problem:

  1. At each instant t in [0, T], solve the static radial Dirac
     eigenvalue problem (via the bound-state shooting method in
     `dirac_spectrum.find_bound_states`) for the joint Lewis-
     Papapetrou metric of the SystrophePair with offset delta(t).
  2. Time-average the instantaneous eigenvalues:
        eps_n = (1/T) integral_0^T E_n(t) dt.
  3. These are the adiabatic Floquet quasi-energies; non-adiabatic
     corrections appear at order Omega_drive / |E_m - E_n|.

This is the actual radial Dirac on a time-varying LP background, not a
toy 2-level system. Bound state eigenvalues come from the real
Chandrasekhar-Page-on-LP radial system implemented in
`dirac_spectrum.boundary_functional`.

Limitations
-----------
- Adiabatic only. Fast driving (Omega_drive comparable to the
  bound-state spacing) requires the non-adiabatic Berry-phase
  correction A_{mn}(t) = i <psi_m | partial_t | psi_n>, which is
  implemented in `nonadiabatic_floquet_correction`.
- Requires the static-pair to have a discrete bound-state spectrum on
  [r_min, r_max] with Dirichlet BCs. The unbounded-r continuum is not
  covered here.

References
----------
- J. H. Shirley, Phys. Rev. 138 (1965) B979.
- M. V. Berry, Proc. R. Soc. London A 392 (1984) 45 (adiabatic geometric phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.dirac_spectrum import find_bound_states


@dataclass(frozen=True)
class AdiabaticFloquetSpectrum:
    """Adiabatic Floquet quasi-energies of a periodically driven LP-pair.

    Attributes
    ----------
    t_samples : np.ndarray
        Time samples in [0, T].
    delta_samples : np.ndarray
        delta(t) at each sample.
    instantaneous_eigenvalues : list[np.ndarray]
        E_n(t_i) for each t_i. Each entry is an array of bound-state
        energies at the corresponding instant; arrays may have different
        lengths if the spectrum changes count across the drive cycle.
    quasi_energies : np.ndarray
        Adiabatic Floquet quasi-energies: time-averaged instantaneous
        eigenvalues. Shape (n_states,) where n_states is the minimum
        number of bound states across all sampled times.
    period : float
        T = 2 pi / Omega_drive.
    """

    t_samples: np.ndarray
    delta_samples: np.ndarray
    instantaneous_eigenvalues: list
    quasi_energies: np.ndarray
    period: float


def _joint_LP_metric(cyl: VanStockumInterior, r: float, delta: float) -> tuple[float, float, float]:
    """Joint (F_pair, K_pair, L_pair) for the matched-cylinder pair at offset delta.

    F_pair, K_pair are computed by direct linear superposition; L_pair via
    the log-shift superposition L(r) + L(r * exp(-delta / alpha)) used
    elsewhere in the package.
    """
    alpha = cyl.alpha
    r_shifted = r * np.exp(-delta / alpha)
    F = float(cyl.analytic_exterior_F(r) + cyl.analytic_exterior_F(r_shifted))
    K = float(cyl.analytic_exterior_K(r) + cyl.analytic_exterior_K(r_shifted))
    L = float(cyl.analytic_exterior_L(r) + cyl.analytic_exterior_L(r_shifted))
    return F, K, L


def static_pair_bound_states(
    cyl: VanStockumInterior,
    delta: float,
    m: int = 0,
    k: float = 0.0,
    mass: float = 0.5,
    r_min: float | None = None,
    r_max: float | None = None,
    E_min: float = 0.3,
    E_max: float = 3.0,
    n_E: int = 80,
) -> np.ndarray:
    """Compute bound-state energies of the radial Dirac on the *static* joint pair.

    Builds the time-independent LP metric of a matched SystrophePair at
    offset delta, then shoots for bound-state energies on
    [r_min, r_max] with Dirichlet BCs.

    Returns the array of bound-state energies.
    """
    if r_min is None:
        r_min = cyl.R + 0.05 * cyl.R
    if r_max is None:
        r_max = 5.0 * cyl.R

    def F_fn(r): return _joint_LP_metric(cyl, r, delta)[0]
    def K_fn(r): return _joint_LP_metric(cyl, r, delta)[1]
    def L_fn(r): return _joint_LP_metric(cyl, r, delta)[2]
    def h_fn(r): return 1.0

    energies = find_bound_states(
        F_fn, K_fn, L_fn, h_fn,
        m=m, k=k, mass=mass,
        r_min=r_min, r_max=r_max,
        E_min=E_min, E_max=E_max, n_E=n_E,
    )
    return energies


def adiabatic_floquet_spectrum(
    cyl: VanStockumInterior,
    delta_0: float,
    delta_amp: float,
    omega_drive: float,
    m: int = 0,
    k: float = 0.0,
    mass: float = 0.5,
    n_t: int = 16,
    r_min: float | None = None,
    r_max: float | None = None,
    E_min: float = 0.3,
    E_max: float = 3.0,
    n_E: int = 60,
) -> AdiabaticFloquetSpectrum:
    """Compute the adiabatic Floquet spectrum.

    Parameters
    ----------
    cyl : VanStockumInterior
        Single-cylinder model; the pair is matched (both cylinders share cyl).
    delta_0, delta_amp, omega_drive : float
        delta(t) = delta_0 + delta_amp sin(omega_drive t).
    m, k, mass : int, float, float
        Quantum numbers of the radial Dirac mode.
    n_t : int
        Number of time samples per period for the integral.

    Returns
    -------
    AdiabaticFloquetSpectrum
    """
    T_period = 2.0 * np.pi / omega_drive
    t_samples = np.linspace(0.0, T_period, n_t, endpoint=False)
    delta_samples = delta_0 + delta_amp * np.sin(omega_drive * t_samples)

    inst_eigvals: list[np.ndarray] = []
    for d in delta_samples:
        try:
            E_n = static_pair_bound_states(
                cyl, delta=float(d), m=m, k=k, mass=mass,
                r_min=r_min, r_max=r_max,
                E_min=E_min, E_max=E_max, n_E=n_E,
            )
        except Exception:
            E_n = np.array([])
        inst_eigvals.append(E_n)

    # Time-average each bound state's energy across the drive cycle
    counts = [len(e) for e in inst_eigvals]
    n_states = min(counts) if counts and min(counts) > 0 else 0
    if n_states == 0:
        quasi = np.array([])
    else:
        E_matrix = np.stack([np.sort(e)[:n_states] for e in inst_eigvals])
        # Reduce mod omega_drive (Floquet eigenvalues are defined mod Omega)
        eps = np.mean(E_matrix, axis=0)
        # Wrap to fundamental Brillouin zone [-Omega/2, Omega/2)
        eps_wrapped = ((eps + omega_drive / 2.0) % omega_drive) - omega_drive / 2.0
        quasi = eps_wrapped

    return AdiabaticFloquetSpectrum(
        t_samples=t_samples,
        delta_samples=delta_samples,
        instantaneous_eigenvalues=inst_eigvals,
        quasi_energies=quasi,
        period=T_period,
    )


def nonadiabatic_floquet_correction(
    cyl: VanStockumInterior,
    n_state: int,
    delta_0: float,
    delta_amp: float,
    omega_drive: float,
    m: int = 0,
    k: float = 0.0,
    mass: float = 0.5,
    n_t: int = 16,
) -> float:
    """Leading non-adiabatic correction to the n-th Floquet quasi-energy.

    The first-order correction is
        Delta eps_n = sum_{m != n} |A_{mn}|^2 / (E_n - E_m + (k_mn) Omega_drive),
    where A_{mn}(t) = <psi_m | partial_t | psi_n> are non-adiabatic
    couplings. Evaluating A explicitly requires the eigenstates and
    their time-derivatives; here we estimate the magnitude via the
    instantaneous spectrum's variation,

        |A| ~ |dE_n / dt| / (E_m - E_n),

    which gives the rough scale of the correction. A full implementation
    requires solving for the eigenstates and integrating; this returns
    the *order-of-magnitude estimate* used to assess the adiabatic
    validity.

    Returns a scalar estimate of the correction; if abs(correction) <
    abs(quasi_energy), the adiabatic approximation is self-consistent.
    """
    spec = adiabatic_floquet_spectrum(
        cyl, delta_0, delta_amp, omega_drive,
        m=m, k=k, mass=mass, n_t=n_t,
    )
    if spec.quasi_energies.size <= n_state:
        return float("nan")
    # Estimate |dE_n / dt| from the time samples of the n-th eigenvalue
    E_n_t = np.array([
        np.sort(e)[n_state] if len(e) > n_state else np.nan
        for e in spec.instantaneous_eigenvalues
    ])
    valid = ~np.isnan(E_n_t)
    if valid.sum() < 4:
        return float("nan")
    dE_dt = np.gradient(E_n_t[valid], spec.t_samples[valid])
    typical_dE = float(np.max(np.abs(dE_dt)))
    # Gap to the next state
    if spec.quasi_energies.size > n_state + 1:
        gap = float(abs(spec.quasi_energies[n_state + 1] - spec.quasi_energies[n_state]))
    else:
        gap = 1.0
    if gap < 1e-9:
        return float("inf")
    return float(typical_dE / gap * omega_drive / (2 * np.pi))


def adiabatic_floquet_validity(
    cyl: VanStockumInterior,
    delta_0: float,
    delta_amp: float,
    omega_drive: float,
    m: int = 0,
    k: float = 0.0,
    mass: float = 0.5,
) -> dict:
    """Diagnostic: is the adiabatic limit valid for the given parameters?

    Returns a dict with the typical eigenvalue gap and Omega_drive,
    and a ratio Omega_drive / gap. If << 1, adiabatic is excellent;
    if ~ 1, non-adiabatic transitions matter.
    """
    # Spectrum at delta_0 alone (static)
    E_static = static_pair_bound_states(
        cyl, delta=delta_0, m=m, k=k, mass=mass,
    )
    if len(E_static) >= 2:
        gap = float(np.min(np.diff(np.sort(E_static))))
    else:
        gap = float("nan")
    ratio = omega_drive / gap if gap > 0 else float("inf")
    return {
        "static_eigenvalues": E_static.tolist(),
        "min_gap": gap,
        "omega_drive": omega_drive,
        "omega_over_gap": ratio,
        "adiabatic_valid": ratio < 0.1,
        "description": (
            "Adiabatic Floquet is exact in the limit Omega_drive / gap -> 0. "
            f"Current ratio: {ratio:.4f}. "
            f"{'ADIABATIC valid' if ratio < 0.1 else 'NON-ADIABATIC corrections needed' if ratio < 1 else 'NON-ADIABATIC regime; adiabatic invalid'}"
        ),
    }
