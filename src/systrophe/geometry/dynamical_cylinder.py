"""Dynamical Tipler cylinder: time-dependent rotation rate omega(t).

The static Tipler-cylinder spacetime has a fixed rotation rate. In a
realistic physical scenario, omega might evolve (spin-up via accretion,
spin-down via radiation, or quasi-static deformation). This module
treats the *quasi-static approximation*: for slow time evolution, the
exterior is approximately the static LP exterior at the instantaneous
omega, with adiabatic corrections.

Key questions:

- As omega crosses the critical value (omega R = 1/2), do CTC bands
  appear or disappear continuously, or does a topological transition
  occur?
- What is the time-derivative of total CTC log-measure across a slow
  spin-up?
- Does back-reaction (e.g., Hawking radiation feedback) drive omega
  away from the supercritical regime?

This module provides:

- `dynamical_omega(t, profile)`: pre-built profiles (linear ramp,
  step, exponential, sinusoidal)
- `instantaneous_ctc_measure(omega, R, ...)`: integrated CTC measure
  at omega(t)
- `spin_up_history(t_array, profile, ...)`: track CTC content during
  spin-up
- `formation_transition(profile, R, ...)`: find time when CTC bands
  first appear
- `adiabatic_back_reaction_estimate(omega, ...)`: leading-order
  back-reaction on omega from Hawking-like emission
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


# ----- Time-dependent omega profiles --------------------------------------

def linear_ramp_profile(omega_initial: float, omega_final: float,
                          t_total: float) -> Callable[[float], float]:
    """Linear ramp omega(t) from omega_initial to omega_final."""
    def omega_t(t):
        if t < 0:
            return omega_initial
        if t > t_total:
            return omega_final
        return omega_initial + (omega_final - omega_initial) * (t / t_total)
    return omega_t


def step_profile(omega_initial: float, omega_final: float,
                  t_step: float) -> Callable[[float], float]:
    """Step profile: omega = omega_initial for t < t_step, then omega_final."""
    def omega_t(t):
        return omega_final if t >= t_step else omega_initial
    return omega_t


def exponential_profile(omega_inf: float, tau: float,
                         omega_0: float = 0.0) -> Callable[[float], float]:
    """Exponential approach: omega(t) = omega_inf * (1 - exp(-t/tau)) + omega_0."""
    def omega_t(t):
        if t < 0:
            return omega_0
        return omega_inf * (1 - np.exp(-t / tau)) + omega_0
    return omega_t


def sinusoidal_profile(omega_mean: float, omega_amp: float,
                        period: float) -> Callable[[float], float]:
    """Sinusoidal variation about omega_mean."""
    def omega_t(t):
        return omega_mean + omega_amp * np.sin(2 * np.pi * t / period)
    return omega_t


# ----- Instantaneous CTC measure ------------------------------------------

def instantaneous_ctc_measure(
    omega: float, R: float = 1.0,
    r_min: float = 1.05, r_max: float = 20.0, n_grid: int = 4001,
) -> dict:
    """CTC content at a single instant given omega.

    Returns dict with:
      - omega
      - is_supercritical (omega R > 1/2)
      - n_ctc_bands
      - total_log_measure (sum of ln(r_outer/r_inner))
      - n_chronology_horizons (F = 0 zeros)
    """
    try:
        vs = VanStockumInterior(omega=omega, R=R)
    except ValueError:
        return {"omega": omega, "is_supercritical": False,
                "n_ctc_bands": 0, "total_log_measure": 0.0,
                "n_chronology_horizons": 0}

    is_super = vs.is_supercritical()
    rs = np.linspace(r_min, r_max, n_grid)
    Ls = np.array([float(vs.analytic_exterior_L(r)) for r in rs])
    Fs = np.array([float(vs.analytic_exterior_F(r)) for r in rs])

    # CTC bands: L < 0 intervals
    in_ctc = Ls < 0
    if not in_ctc.any():
        n_bands = 0
        total_log = 0.0
    else:
        diff = np.diff(in_ctc.astype(int))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if in_ctc[0]:
            starts = np.insert(starts, 0, 0)
        if in_ctc[-1]:
            ends = np.append(ends, len(in_ctc) - 1)
        n_bands = len(starts)
        total_log = sum(float(np.log(rs[e] / rs[s])) for s, e in zip(starts, ends))

    # Chronology horizons: F = 0
    sign_F = np.sign(Fs)
    n_horizons = int(np.sum(np.diff(sign_F) != 0))

    return {
        "omega": omega, "is_supercritical": is_super,
        "n_ctc_bands": n_bands, "total_log_measure": total_log,
        "n_chronology_horizons": n_horizons,
    }


# ----- Spin-up history -----------------------------------------------------

@dataclass(frozen=True)
class SpinUpHistory:
    """Dynamical evolution of CTC content."""

    times: np.ndarray
    omegas: np.ndarray
    n_bands: list[int]
    log_measures: list[float]
    n_horizons: list[int]
    formation_time: float | None  # t at which first CTC band appears


def spin_up_history(
    t_array: np.ndarray, omega_profile: Callable[[float], float],
    R: float = 1.0,
) -> SpinUpHistory:
    """Track CTC content over time t_array given an omega profile."""
    omegas = np.array([omega_profile(float(t)) for t in t_array])
    n_bands_list = []
    measures = []
    horizons = []
    formation_time = None
    for i, om in enumerate(omegas):
        m = instantaneous_ctc_measure(float(om), R=R)
        n_bands_list.append(m["n_ctc_bands"])
        measures.append(m["total_log_measure"])
        horizons.append(m["n_chronology_horizons"])
        if formation_time is None and m["n_ctc_bands"] > 0:
            formation_time = float(t_array[i])
    return SpinUpHistory(
        times=t_array, omegas=omegas,
        n_bands=n_bands_list, log_measures=measures,
        n_horizons=horizons,
        formation_time=formation_time,
    )


def formation_transition(
    omega_profile: Callable[[float], float],
    R: float = 1.0, t_max: float = 100.0, n_t: int = 200,
) -> dict:
    """Find the formation time when CTC bands first appear.

    Returns dict with:
      - formation_time: t at which CTC content first becomes nonzero
      - critical_omega: omega(formation_time)
      - matches_critical_a: True iff critical_omega * R is close to 0.5
    """
    t_array = np.linspace(0, t_max, n_t)
    hist = spin_up_history(t_array, omega_profile, R=R)
    if hist.formation_time is None:
        return {"formation_time": None, "critical_omega": None,
                "matches_critical_a": False}
    omega_form = omega_profile(hist.formation_time)
    matches = abs(omega_form * R - 0.5) < 0.05  # within 10% of critical
    return {
        "formation_time": hist.formation_time,
        "critical_omega": omega_form,
        "matches_critical_a": matches,
        "critical_a": omega_form * R,
    }


# ----- Adiabatic back-reaction -------------------------------------------

def adiabatic_back_reaction_estimate(
    omega: float, R: float = 1.0,
    t_evolution_scale: float = 1.0,
) -> dict:
    """Leading-order back-reaction estimate on omega from QFT effects.

    For a static rotating cylinder, the analog Hawking emission from the
    F = 0 horizon carries away energy and angular momentum. In the
    adiabatic approximation, omega decreases monotonically by

        d omega/dt ~ -T_H^2 / (J_cyl * t_evolution_scale)

    where J_cyl is the cylinder's angular momentum and T_H is the
    analog Hawking temperature.

    Returns dict with:
      - omega
      - T_H_acoustic (analog Hawking T at the F=0 horizon)
      - J_cylinder (van Stockum angular momentum per length)
      - d_omega_dt: leading back-reaction rate
      - omega_relaxation_time: omega / |d_omega/dt|
    """
    try:
        vs = VanStockumInterior(omega=omega, R=R)
    except ValueError:
        return {"omega": omega, "is_supercritical": False,
                "d_omega_dt": 0.0, "omega_relaxation_time": float("inf")}

    if not vs.is_supercritical():
        return {"omega": omega, "is_supercritical": False,
                "T_H_acoustic": 0.0,
                "J_cylinder": vs.angular_momentum_per_unit_length,
                "d_omega_dt": 0.0, "omega_relaxation_time": float("inf")}

    # Find first chronology horizon
    from systrophe.analogs.acoustic_metric import (
        acoustic_horizon_radius, acoustic_hawking_temperature,
    )
    r_h = acoustic_horizon_radius(vs, r_min=1.05, r_max=50.0)
    if r_h is None:
        return {"omega": omega, "T_H_acoustic": 0.0,
                "d_omega_dt": 0.0, "omega_relaxation_time": float("inf")}

    T_H = acoustic_hawking_temperature(vs, r_h)
    J = vs.angular_momentum_per_unit_length

    # Back-reaction: d omega/dt ~ -T_H^2 / J / t_scale  (heuristic)
    d_omega_dt = -(T_H ** 2) / max(abs(J), 1e-30) / t_evolution_scale
    relax_time = abs(omega / d_omega_dt) if abs(d_omega_dt) > 1e-30 else float("inf")

    return {
        "omega": omega, "is_supercritical": True,
        "T_H_acoustic": T_H, "J_cylinder": J,
        "d_omega_dt": d_omega_dt,
        "omega_relaxation_time": relax_time,
    }
