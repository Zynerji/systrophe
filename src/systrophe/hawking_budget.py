"""Hawking radiation budget per chronology horizon.

Each F=0 surface in the LP supercritical exterior is locally a Killing
horizon. The local Hawking temperature is

    T_H(r_n) = kappa(r_n) / (2 pi)

with surface gravity kappa = (1/2) |F'(r_n)|. Combined with the
horizon area A_n and Bekenstein-Hawking entropy S_n = A_n / 4, we get
a per-horizon "energy budget":

    Total radiation energy: E_rad ~ T_H^4 * A * (time)  (Stefan-Boltzmann)
    Total entropy radiated: S_rad ~ T_H^3 * A * (time)

Module functions:
- surface_gravity_at_CH
- hawking_temperature_at_CH
- horizon_area_at_CH
- bekenstein_hawking_entropy_at_CH
- evaporation_time_estimate
- per_CH_radiation_budget (full report per horizon)
- total_radiation_lifetime
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class CHRadiationBudget:
    horizon_index: int
    r_CH: float
    surface_gravity: float
    hawking_temperature: float
    horizon_area: float
    bekenstein_hawking_entropy: float
    evaporation_time: float


def surface_gravity_at_CH(vs: VanStockumInterior, r_CH: float) -> float:
    """kappa = (1/2) |dF/dr| at the horizon."""
    eps = 1e-4 * r_CH
    F_plus = float(vs.analytic_exterior_F(np.array([r_CH + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r_CH - eps]))[0])
    Fp = (F_plus - F_minus) / (2 * eps)
    return float(0.5 * abs(Fp))


def hawking_temperature_at_CH(vs: VanStockumInterior, r_CH: float) -> float:
    """T_H = kappa / (2 pi) in natural units."""
    kappa = surface_gravity_at_CH(vs, r_CH)
    return float(kappa / (2 * math.pi))


def horizon_area_at_CH(vs: VanStockumInterior, r_CH: float,
                        cylinder_length: float = 1.0) -> float:
    """A = 2 pi sqrt(|L(r_CH)|) * cylinder_length."""
    L = float(vs.analytic_exterior_L(np.array([r_CH]))[0])
    return float(2 * math.pi * math.sqrt(abs(L)) * cylinder_length)


def bekenstein_hawking_entropy_at_CH(
    vs: VanStockumInterior, r_CH: float, cylinder_length: float = 1.0,
) -> float:
    """S_BH = A / (4 l_P^2). Natural units: l_P = 1."""
    A = horizon_area_at_CH(vs, r_CH, cylinder_length)
    return float(A / 4.0)


def evaporation_time_estimate(
    vs: VanStockumInterior, r_CH: float, cylinder_length: float = 1.0,
) -> float:
    """t_evap ~ S_BH / T_H (heuristic).

    Stefan-Boltzmann gives dE/dt ~ T^4 A. Total E ~ T_H * S_BH. So
    t_evap ~ S_BH / T_H^3 if we keep A constant; more accurately
    t_evap = M / (dE/dt) with M = T_H S_BH for Schwarzschild-like.
    Use heuristic: t_evap ~ S_BH / T_H.
    """
    T = hawking_temperature_at_CH(vs, r_CH)
    S = bekenstein_hawking_entropy_at_CH(vs, r_CH, cylinder_length)
    if T <= 0:
        return float("inf")
    return float(S / T)


def per_CH_radiation_budget(
    vs: VanStockumInterior, n_horizons: int = 5,
    cylinder_length: float = 1.0,
) -> list[CHRadiationBudget]:
    """One budget record per chronology horizon."""
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
        if len(zeros) >= n_horizons:
            break
    out = []
    for i, r_CH in enumerate(zeros):
        out.append(CHRadiationBudget(
            horizon_index=i,
            r_CH=r_CH,
            surface_gravity=surface_gravity_at_CH(vs, r_CH),
            hawking_temperature=hawking_temperature_at_CH(vs, r_CH),
            horizon_area=horizon_area_at_CH(vs, r_CH, cylinder_length),
            bekenstein_hawking_entropy=bekenstein_hawking_entropy_at_CH(
                vs, r_CH, cylinder_length),
            evaporation_time=evaporation_time_estimate(vs, r_CH, cylinder_length),
        ))
    return out


def total_radiation_lifetime(
    vs: VanStockumInterior, n_horizons: int = 5,
    cylinder_length: float = 1.0,
) -> float:
    """Sum of t_evap across all horizons (rough total lifetime)."""
    budgets = per_CH_radiation_budget(vs, n_horizons, cylinder_length)
    finite = [b.evaporation_time for b in budgets if math.isfinite(b.evaporation_time)]
    if not finite:
        return float("inf")
    return float(sum(finite))


def log_periodic_temperature_pattern(
    vs: VanStockumInterior, n_horizons: int = 5,
) -> dict:
    """Check if Hawking temperatures at successive CHs follow a
    geometric (log-periodic) pattern.

    From Tipler: r_{n+1} / r_n = exp(pi / alpha). If kappa scales as
    1/r at horizon (heuristic), then T_n ratios should equal r_n/r_{n+1}
    = exp(-pi/alpha). Test this prediction.
    """
    budgets = per_CH_radiation_budget(vs, n_horizons)
    if len(budgets) < 2:
        return {"ratios": [], "predicted_ratio": float("nan")}
    if vs.is_supercritical():
        predicted = math.exp(-math.pi / vs.alpha)
    else:
        predicted = float("nan")
    ratios = []
    for i in range(len(budgets) - 1):
        T_i = budgets[i].hawking_temperature
        T_ip1 = budgets[i + 1].hawking_temperature
        if T_i > 0:
            ratios.append(T_ip1 / T_i)
    return {
        "ratios": ratios,
        "predicted_ratio": float(predicted),
        "mean_observed_ratio": float(np.mean(ratios)) if ratios else float("nan"),
    }
