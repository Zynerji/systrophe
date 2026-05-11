"""Page curve for the CTC-enclosed region of a Systrophe cylinder.

Treat the chronology-bounded region r in (r_1, r_2) (between two
consecutive F=0 surfaces) as an "evaporating" subsystem. The
enclosed quantum fields decay outward through Hawking-like emission
at the inner CH and absorb inward through the outer CH. The
entanglement entropy between enclosed-region degrees of freedom and
exterior degrees of freedom traces a Page-like curve:
  - early: S grows linearly (thermal radiation regime)
  - Page time: S reaches max ~ log(d_enclosed)
  - late: S decreases linearly back to 0 (information return)

For our toy model:
  S_grav(t) = min(s_emit * t, S_BH_initial)
  S_QFT(t) = depends on entanglement wedge

Page curve = min(S_grav(t), S_QFT(t)).

This is *highly* speculative for Systrophe -- there's no real
evaporation in a stationary LP geometry. The module gives a
"as-if" Page curve for the entanglement entropy of the enclosed
quantum field, as it would evolve under a slow spin-down of the
cylinder.

Functions
---------
- enclosed_region_volume: between two consecutive CHs
- bekenstein_bound_estimate: max entropy of enclosed fields
- page_time_estimate: time at which S_grav crosses S_QFT
- entanglement_entropy_proxy: heuristic S(t) curve
- information_paradox_resolved_at_page_time: bool
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class PageCurveData:
    s_at_times: np.ndarray
    t_grid: np.ndarray
    page_time: float
    max_entropy: float


def enclosed_region_volume(vs: VanStockumInterior,
                            r_inner: float, r_outer: float,
                            cylinder_length: float = 1.0) -> float:
    """Approximate proper volume between two radii (constant z slice)."""
    if r_outer <= r_inner:
        raise ValueError("r_outer must exceed r_inner")
    r_samples = np.linspace(r_inner, r_outer, 50)
    F_vals = vs.analytic_exterior_F(r_samples)
    L_vals = vs.analytic_exterior_L(r_samples)
    # Proper volume element: sqrt(|g_rr * g_pp|) dr d phi dz
    integrand = np.sqrt(np.abs(L_vals))  # g_rr ~ 1 by our convention
    dr = (r_outer - r_inner) / (len(r_samples) - 1)
    vol = float(np.trapezoid(integrand, dx=dr) * 2 * math.pi * cylinder_length)
    return vol


def bekenstein_bound_estimate(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    energy_density: float = 1.0,
) -> float:
    """S_max ~ 2 pi R * E (Bekenstein). Use enclosed volume as proxy."""
    V = enclosed_region_volume(vs, r_inner, r_outer)
    E = energy_density * V
    R_enc = (r_outer - r_inner) / 2.0
    return float(2 * math.pi * R_enc * E)


def page_time_estimate(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    emission_rate: float = 0.01,
    energy_density: float = 1.0,
) -> float:
    """Time at which S_grav crosses S_QFT (heuristic).

    Page time t_P = T_evap / 2 = S_max / s_emit. Peak of the curve.
    """
    S_max = bekenstein_bound_estimate(vs, r_inner, r_outer, energy_density)
    if emission_rate <= 0:
        return float("inf")
    return float(S_max / emission_rate)


def entanglement_entropy_proxy(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    t: float, emission_rate: float = 0.01, energy_density: float = 1.0,
) -> float:
    """S(t) = min(s_emit * t, S_max - s_emit * t) for t in [0, T_evap]."""
    S_max = bekenstein_bound_estimate(vs, r_inner, r_outer, energy_density)
    if emission_rate <= 0:
        return 0.0
    T_evap = 2 * S_max / emission_rate
    if t <= 0:
        return 0.0
    if t >= T_evap:
        return 0.0
    return float(min(emission_rate * t, emission_rate * (T_evap - t)))


def page_curve(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    emission_rate: float = 0.01, energy_density: float = 1.0,
    n_t_samples: int = 200,
) -> PageCurveData:
    """Full Page curve S(t) over t in [0, T_evap]."""
    S_max = bekenstein_bound_estimate(vs, r_inner, r_outer, energy_density)
    if emission_rate <= 0:
        T_evap = 1.0
    else:
        T_evap = 2 * S_max / emission_rate
    t_grid = np.linspace(0.0, T_evap, n_t_samples)
    s_vals = np.array([
        entanglement_entropy_proxy(vs, r_inner, r_outer, float(t),
                                     emission_rate, energy_density)
        for t in t_grid
    ])
    pt = page_time_estimate(vs, r_inner, r_outer, emission_rate,
                              energy_density)
    return PageCurveData(
        s_at_times=s_vals,
        t_grid=t_grid,
        page_time=float(pt),
        max_entropy=float(S_max),
    )


def information_paradox_resolved_at_page_time(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    emission_rate: float = 0.01,
) -> dict:
    """Diagnostic: at the Page time, the entanglement should drop.

    Returns: page_time, entropy at page_time, entropy at 2*page_time.
    Resolution: entropy(2*pt) < entropy(pt) confirms decay -> information
    is returning.
    """
    pt = page_time_estimate(vs, r_inner, r_outer, emission_rate)
    if not math.isfinite(pt):
        return {"page_time": float("inf"), "resolved": False}
    S_at_pt = entanglement_entropy_proxy(vs, r_inner, r_outer, pt, emission_rate)
    S_at_2pt = entanglement_entropy_proxy(vs, r_inner, r_outer, 2 * pt, emission_rate)
    return {
        "page_time": float(pt),
        "entropy_at_page_time": S_at_pt,
        "entropy_at_2x_page_time": S_at_2pt,
        "resolved": S_at_2pt < S_at_pt,
    }


def all_ctc_band_page_curves(
    vs: VanStockumInterior, n_bands: int = 3,
    emission_rate: float = 0.01,
) -> list[PageCurveData]:
    """One Page curve per CTC band."""
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
    curves = []
    for i in range(min(n_bands, len(zeros) - 1)):
        curve = page_curve(vs, zeros[i], zeros[i + 1], emission_rate)
        curves.append(curve)
    return curves
