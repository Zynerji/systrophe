"""Feedback-amplified warp-bubble shell: parametric Q-cavity model.

A bare warp-bubble shell needs the integrated negative energy density
to satisfy the Pfenning-Ford quantum inequality::

    integral over tau of |T_{tt}|  >=  3 / (32 pi^2 sigma^2)

The bound is on the *product* of magnitude and duration, not either
factor alone. This module models a high-Q feedback resonator
embedded in the shell that converts the requirement for an
instantaneously-large negative energy into a continuously-pumped
standing-wave with long lifetime tau >> sigma^{-1}.

The control-engineering picture
-------------------------------
The shell carries a standing wave of stored "exotic" field energy at
its natural frequency f_0 = c / (2 pi sigma) (sigma = wall thickness).
A coherent pump at 2 f_0 (parametric drive) pumps positive-energy
photons into the wall; degenerate parametric amplification mixes
them into squeezed-vacuum states whose vacuum expectation value of
T_{tt} can be negative.

The energy balance is::

    E_shell(t)  =  E_shell(0) * exp(g * t)
    P_drive     =  (E_shell / tau_cavity)   (sustained at saturation)

with parametric gain g and cavity lifetime tau_cavity = Q / f_0.

Pfenning-Ford compatibility
---------------------------
With cavity factor Q, the local instantaneous |T_{tt}| is reduced
relative to the impulsive Alcubierre requirement by a factor 1/Q,
while the necessary duration grows by Q. The product is invariant
(P-F is saturated), so the bound is RESPECTED, but instantaneous
power demand falls by Q.

This module
-----------
- ``shell_natural_frequency`` -- f_0 = c / (2 pi sigma)
- ``parametric_gain_to_steady`` -- g such that |T_{tt}| reaches the
  target negative density at saturation.
- ``required_drive_power`` -- P_drive given (Q, sigma, v_s, R).
- ``q_factor_sweep`` -- sweep Q at fixed bubble parameters; the
  catcher should be smooth (continuous tradeoff) UNLESS the cavity
  hits a resonance with the wall standing-wave mode, in which case
  a sharp Hamming transition flags the resonance.
- ``novelty_scan`` -- catcher across (Q, sigma) for the standing-
  wave saturation amplitude.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.geometry.alcubierre import (
    alcubierre_total_negative_energy,
    pfenning_ford_quantum_bound,
)
from systrophe.catchers.novelty_catcher import scan_novelty


def shell_natural_frequency(sigma: float, c: float = 1.0) -> float:
    """f_0 = c / (2 pi sigma). Units of inverse time."""
    return c / (2.0 * math.pi * sigma)


def cavity_lifetime(Q: float, sigma: float, c: float = 1.0) -> float:
    """tau = Q / f_0."""
    return Q / shell_natural_frequency(sigma, c)


def saturation_field_energy(
    v_s: float, R: float, sigma: float, Q: float,
) -> float:
    """|E_shell| at saturation, given a target Alcubierre bubble.

    The shell stores enough stress-energy to sustain the Alcubierre
    NEC profile for tau = Q / f_0. The required magnitude is
        |E_shell| = |E_neg_bare| / Q  (per cavity cycle)
    """
    E_neg = abs(alcubierre_total_negative_energy(v_s=v_s, R=R, sigma=sigma))
    return E_neg / max(Q, 1.0)


def required_drive_power(
    v_s: float, R: float, sigma: float, Q: float,
) -> float:
    """Sustained pump power at saturation.

    P_drive = |E_shell| / tau_cavity = |E_neg_bare| / Q / tau
             = |E_neg_bare| * f_0 / Q^2
    """
    E_shell = saturation_field_energy(v_s, R, sigma, Q)
    tau = cavity_lifetime(Q, sigma)
    return float(E_shell / max(tau, 1e-15))


def pfenning_ford_check(
    v_s: float, R: float, sigma: float, Q: float,
) -> dict:
    """Verify that the feedback-amplified shell respects Pfenning-Ford.

    Returns a dict with the integrated |T_{tt}| * tau and the P-F
    lower bound. Compatible iff `product >= lower_bound`.
    """
    E_shell = saturation_field_energy(v_s, R, sigma, Q)
    tau = cavity_lifetime(Q, sigma)
    product = E_shell * tau
    bound = pfenning_ford_quantum_bound(v_s=v_s, R=R, sigma=sigma)
    return {
        "stored_energy_times_tau": float(product),
        "pfenning_ford_lower_bound": float(bound),
        "compatible": product >= bound - 1e-12,
        "Q_factor": float(Q),
        "tau_cavity": float(tau),
    }


def parametric_gain_to_steady(
    v_s: float, R: float, sigma: float, Q: float,
    g_max: float = 10.0,
) -> float:
    """Parametric gain required to maintain the shell at saturation.

    For an idealised parametric amplifier g satisfies
        exp(g * tau) = Q ,  so  g = log(Q) / tau
    """
    if Q <= 1.0:
        return 0.0
    tau = cavity_lifetime(Q, sigma)
    g = math.log(Q) / max(tau, 1e-15)
    return float(min(g, g_max))


@dataclass(frozen=True)
class FeedbackSweep:
    Q_grid: np.ndarray
    sigma_grid: np.ndarray
    drive_power: np.ndarray  # shape (n_Q, n_sigma)
    novelty_verdict: str


def novelty_scan(
    Q_range: tuple[float, float] = (1.0, 100.0), n_Q: int = 30,
    sigma_range: tuple[float, float] = (1.0, 20.0), n_sigma: int = 30,
    v_s: float = 1.0, R: float = 1.0,
) -> dict:
    """Sweep (Q, sigma) and run the catcher on the drive-power curve.

    Predicted resonance: when Q * f_0 hits an integer multiple of the
    pump frequency, the cavity rings up; catcher should flag a sharp
    Hamming transition there.
    """
    Q_grid = np.linspace(*Q_range, n_Q)
    sigma_grid = np.linspace(*sigma_range, n_sigma)
    drive_power = np.zeros((n_Q, n_sigma))
    for i, Q in enumerate(Q_grid):
        for j, sg in enumerate(sigma_grid):
            drive_power[i, j] = required_drive_power(v_s, R, float(sg), float(Q))
    def fn(qv):
        idx = int(np.argmin(np.abs(Q_grid - qv)))
        return drive_power[idx]
    result = scan_novelty(Q_grid, fn, n_bits=32)
    return {
        "Q_grid": Q_grid.tolist(),
        "sigma_grid": sigma_grid.tolist(),
        "drive_power_grid": drive_power.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
