"""Quantum tunneling out of a CTC band into a chronology-safe region.

A particle confined to a CTC band r in (r_n, r_{n+1}) classically
cannot escape (the chronology horizons F=0 are turning points of the
radial potential for timelike trajectories). Quantum mechanically,
the particle can tunnel out via a WKB barrier crossing.

We compute:
- effective_radial_potential: V(r) for a confined wave
- tunneling_action: int sqrt(2m(V-E)) dr through the barrier
- tunneling_rate: Gamma ~ omega_attempt * exp(-2 S / hbar)
- escape_to_chronology_safe_region: probability per unit time
- multi_band_tunneling_diagram: rates between adjacent bands

This is closely related to Phase 22 (KG scattering) but focuses on
*bound state* tunneling out of confining CTC regions, with
explicit Boltzmann-style escape rates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class CTCTunnelingRate:
    r_inner: float
    r_outer: float
    barrier_action: float
    attempt_frequency: float
    escape_rate: float
    half_life: float


def effective_radial_potential(
    vs: VanStockumInterior, r: float,
    energy: float = 1.0, angular_momentum: int = 0,
) -> float:
    """V_eff(r) for a quantum particle on the LP background.

    V_eff = (1/2) F(r)^{-1} + (angular_momentum^2) / (2 |L(r)|) - energy
    Acts as turning-point potential for radial motion.
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    if abs(F) < 1e-12:
        return float("inf")  # barrier wall at chronology horizon
    V = 0.5 / abs(F) + (angular_momentum ** 2) / max(2 * abs(L), 1e-12) - energy
    return float(V)


def tunneling_action(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    energy: float = 1.0, mass: float = 1.0,
    n_samples: int = 200,
) -> float:
    """Action S = int_{r_inner}^{r_outer} sqrt(2m(V-E)) dr in barrier."""
    r_grid = np.linspace(r_inner, r_outer, n_samples)
    dr = (r_outer - r_inner) / (n_samples - 1)
    total = 0.0
    for r in r_grid:
        V = effective_radial_potential(vs, float(r), energy=0.0)
        delta = V - energy
        if delta > 0:
            total += math.sqrt(2 * mass * delta) * dr
    return float(total)


def attempt_frequency(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
) -> float:
    """Heuristic attempt freq = c / (width of well) in natural units."""
    return float(1.0 / max(r_outer - r_inner, 1e-12))


def tunneling_rate(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    energy: float = 1.0, mass: float = 1.0,
    hbar: float = 1.0,
) -> CTCTunnelingRate:
    """WKB tunneling rate: Gamma = nu * exp(-2 S / hbar)."""
    S = tunneling_action(vs, r_inner, r_outer, energy=energy, mass=mass)
    nu = attempt_frequency(vs, r_inner, r_outer)
    exponent = -2 * S / max(hbar, 1e-30)
    # Bound to avoid underflow
    if exponent < -700:
        rate = 0.0
        half_life = float("inf")
    else:
        rate = nu * math.exp(exponent)
        half_life = math.log(2) / max(rate, 1e-30)
    return CTCTunnelingRate(
        r_inner=float(r_inner), r_outer=float(r_outer),
        barrier_action=float(S),
        attempt_frequency=float(nu),
        escape_rate=float(rate),
        half_life=float(half_life),
    )


def escape_to_chronology_safe_region(
    vs: VanStockumInterior, r_inner_CH: float, r_safe: float,
    energy: float = 1.0, mass: float = 1.0,
) -> dict:
    """Probability of escape from r_inner_CH inward to chronology-safe r_safe.

    Note: r_safe must be in the region where F > 0 (chronology-safe).
    For LP supercritical exterior, r_safe < r_1 (first chronology horizon).
    """
    if r_safe >= r_inner_CH:
        raise ValueError("r_safe must be < r_inner_CH (inside-out tunneling)")
    F_safe = float(vs.analytic_exterior_F(np.array([r_safe]))[0])
    if F_safe <= 0:
        raise ValueError(f"r_safe = {r_safe} is not chronology-safe (F<=0)")
    rate = tunneling_rate(vs, r_safe, r_inner_CH, energy=energy, mass=mass)
    return {
        "r_safe": r_safe,
        "r_inner_CH": r_inner_CH,
        "F_safe": F_safe,
        "escape_rate": rate.escape_rate,
        "half_life": rate.half_life,
        "barrier_action": rate.barrier_action,
    }


def multi_band_tunneling_diagram(
    vs: VanStockumInterior, n_bands: int = 3,
    energy: float = 1.0, mass: float = 1.0,
) -> list[dict]:
    """Tunneling rates between adjacent CTC bands of the supercritical exterior."""
    if not vs.is_supercritical():
        return []
    R = vs.R
    alpha = vs.alpha
    gamma_c = math.pi - math.atan(alpha)
    zeros = []
    for n in range(1, 200):
        u_n = (n * math.pi - gamma_c) / alpha
        if u_n <= 0:
            continue
        r_n = R * math.exp(u_n)
        zeros.append(r_n)
        if len(zeros) >= n_bands + 1:
            break
    out = []
    for i in range(min(n_bands, len(zeros) - 1)):
        rate = tunneling_rate(vs, zeros[i], zeros[i + 1], energy=energy, mass=mass)
        out.append({
            "band_index": i,
            "r_inner": zeros[i],
            "r_outer": zeros[i + 1],
            "barrier_action": rate.barrier_action,
            "escape_rate": rate.escape_rate,
            "half_life": rate.half_life,
        })
    return out


def tunneling_resonance_locus(
    vs: VanStockumInterior, energy_range: tuple[float, float] = (0.1, 10.0),
    n_energies: int = 50, r_inner: float = None, r_outer: float = None,
) -> dict:
    """Search for resonant tunneling energies (Gamma >> typical)."""
    if r_inner is None:
        r_inner = 1.83
    if r_outer is None:
        r_outer = 11.23
    energies = np.linspace(energy_range[0], energy_range[1], n_energies)
    rates = []
    for E in energies:
        rate = tunneling_rate(vs, r_inner, r_outer, energy=float(E))
        rates.append(rate.escape_rate)
    rates_arr = np.asarray(rates)
    # Find resonance peaks (rate > median * 10)
    peaks = []
    median = np.median(rates_arr[rates_arr > 0]) if any(rates_arr > 0) else 0
    for i in range(1, len(rates_arr) - 1):
        if rates_arr[i] > rates_arr[i - 1] and rates_arr[i] > rates_arr[i + 1]:
            if median > 0 and rates_arr[i] > 10 * median:
                peaks.append({"energy": float(energies[i]),
                              "rate": float(rates_arr[i])})
    return {
        "energies": energies,
        "rates": rates_arr,
        "resonance_peaks": peaks,
        "median_rate": float(median),
    }
