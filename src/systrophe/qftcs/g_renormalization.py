"""Scale-dependent renormalization of Newton's constant on the LP background.

Asymptotic Safety / Renormalization Group Improvement (Reuter 1998,
Weinberg 2009) treats Newton's G as a scale-dependent coupling:

    G(k) = G_0 / (1 + omega_R G_0 k^2)

with omega_R ~ 0.78 the dimensionless gravitational coupling at the
UV fixed point. At high k (UV), G(k) -> 0 (asymptotic freedom);
at low k (IR), G(k) -> G_0 (Newtonian limit).

On the LP background, the relevant scale k can be set by:
- Curvature: k ~ sqrt(|R^{(4)}|)
- Inverse local length: k ~ 1/r
- Hawking temperature: k ~ T_H

This module computes G_eff(r) along the LP exterior, identifies
crossover scales, and tests whether the supercritical regime
modifies the standard RG flow.

Functions
---------
- g_running: G(k) flow
- curvature_scale_at_r: R^{(4)}(r) for the LP exterior (heuristic)
- g_eff_at_r: G(k=k(r))
- ultraviolet_fixed_point_distance: where G(k) = 0.5 G_0
- ctc_band_g_modification: G shift within a CTC band
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior

# AS / RG coupling
OMEGA_R = 0.78  # dimensionless gravitational coupling at UV fixed point
G_0 = 1.0  # natural units


@dataclass(frozen=True)
class GRunningData:
    k: float
    G_at_k: float
    relative_change_from_G0: float


def g_running(k: float, G0: float = G_0,
               omega_R: float = OMEGA_R) -> float:
    """G(k) = G0 / (1 + omega_R * G0 * k^2)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    denom = 1.0 + omega_R * G0 * k * k
    return float(G0 / denom)


def curvature_scale_at_r(vs: VanStockumInterior, r: float) -> float:
    """Heuristic k(r) = sqrt(|F''| / |F|) at radius r.

    Captures the local curvature scale. Diverges at F=0 (chronology
    horizons).
    """
    eps = 1e-4 * max(r, 1.0)
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    Fpp = (F_plus + F_minus - 2 * F) / (eps * eps)
    if abs(F) < 1e-12:
        return float("inf")
    return float(math.sqrt(abs(Fpp) / abs(F)))


def g_eff_at_r(vs: VanStockumInterior, r: float,
                G0: float = G_0,
                omega_R: float = OMEGA_R) -> GRunningData:
    """G_eff(r) via curvature-scale-induced running."""
    k = curvature_scale_at_r(vs, r)
    if not math.isfinite(k):
        return GRunningData(k=float("inf"), G_at_k=0.0,
                             relative_change_from_G0=-1.0)
    G_k = g_running(k, G0, omega_R)
    rel = (G_k - G0) / G0
    return GRunningData(k=k, G_at_k=G_k, relative_change_from_G0=rel)


def ultraviolet_fixed_point_distance(
    vs: VanStockumInterior,
    threshold_factor: float = 0.5,
    r_min: float = None, r_max: float = None,
    n_samples: int = 1000,
) -> dict:
    """Find r where G(k(r)) drops to threshold_factor * G_0."""
    if r_min is None:
        r_min = vs.R * 1.001
    if r_max is None:
        r_max = vs.R * 50.0
    r_grid = np.geomspace(r_min, r_max, n_samples)
    threshold = threshold_factor * G_0
    crossings = []
    prev_G = G_0
    for r in r_grid:
        data = g_eff_at_r(vs, float(r))
        if (prev_G > threshold) != (data.G_at_k > threshold):
            crossings.append(float(r))
        prev_G = data.G_at_k
    return {
        "threshold_factor": threshold_factor,
        "crossing_radii": crossings,
        "n_crossings": len(crossings),
    }


def ctc_band_g_modification(
    vs: VanStockumInterior, n_bands: int = 3,
) -> list[dict]:
    """G_eff at midpoint of each CTC band."""
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
        r_mid = math.sqrt(zeros[i] * zeros[i + 1])
        data = g_eff_at_r(vs, r_mid)
        out.append({
            "band_index": i,
            "r_mid": r_mid,
            "k_curvature": data.k,
            "G_at_band": data.G_at_k,
            "relative_change": data.relative_change_from_G0,
        })
    return out


def asymptotic_safety_consistent(
    vs: VanStockumInterior, r_test: float = 1.5,
) -> dict:
    """Check whether G_eff(r) stays bounded in (0, G_0] at r_test."""
    data = g_eff_at_r(vs, r_test)
    consistent = (0 < data.G_at_k <= G_0 + 1e-9)
    return {
        "r": r_test,
        "k": data.k,
        "G_eff": data.G_at_k,
        "is_consistent": bool(consistent),
    }


def newtonian_limit_recovery(
    vs: VanStockumInterior, r_large: float = 100.0,
) -> dict:
    """At large r, G_eff should approach G_0 (Newtonian limit)."""
    data = g_eff_at_r(vs, r_large)
    rel = abs(data.G_at_k - G_0) / G_0
    return {
        "r": r_large,
        "G_eff_at_r": data.G_at_k,
        "G0": G_0,
        "relative_error": float(rel),
        "newtonian_limit_holds": bool(rel < 0.1),
    }
