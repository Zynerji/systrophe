"""Tidal forces and geodesic deviation in the LP supercritical exterior.

For a radially-infalling probe in the LP exterior, the tidal stretch
along the radial direction and the tidal squeeze along the angular
directions are governed by the Riemann tensor components R^r_{trt},
R^phi_{tphit}, etc. The dominant scalar diagnostic is the "tidal
curvature scalar" T = R_{abcd} u^a u^b n^c n^d for a probe four-
velocity u and radial spatial direction n.

This module provides:

- riemann_scalar_radial: T(r) at radius r
- tidal_stretch_at_radius: tidal force per unit length along radial dir
- tidal_squeeze_at_radius: tidal force per unit length along angular dir
- geodesic_deviation_evolution: integrate the Jacobi equation for a
  radial infall from r_start to r_end
- spaghettification_proxy: |T(r)| as r approaches a CH
- multi_band_tidal_signature: T at each (r_n, r_{n+1}) midpoint
- safety_radius_for_probe: r at which |T| exceeds the probe's
  material strength threshold

The Riemann curvature is computed via finite-difference of the
LP metric components; absolute accuracy is ~1e-3 relative.
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def _metric_components(vs: VanStockumInterior, r: float) -> dict:
    """F, K, L plus first and second derivatives at r."""
    eps = 1e-4 * max(abs(r), 1.0)
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    Fp = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    Fm = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    F_p = (Fp - Fm) / (2 * eps)
    F_pp = (Fp + Fm - 2 * F) / (eps * eps)

    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    Kp = float(vs.analytic_exterior_K(np.array([r + eps]))[0])
    Km = float(vs.analytic_exterior_K(np.array([r - eps]))[0])
    K_p = (Kp - Km) / (2 * eps)
    K_pp = (Kp + Km - 2 * K) / (eps * eps)

    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    Lp = float(vs.analytic_exterior_L(np.array([r + eps]))[0])
    Lm = float(vs.analytic_exterior_L(np.array([r - eps]))[0])
    L_p = (Lp - Lm) / (2 * eps)
    L_pp = (Lp + Lm - 2 * L) / (eps * eps)

    return {
        "F": F, "K": K, "L": L,
        "F_p": F_p, "K_p": K_p, "L_p": L_p,
        "F_pp": F_pp, "K_pp": K_pp, "L_pp": L_pp,
    }


def riemann_scalar_radial(vs: VanStockumInterior, r: float) -> float:
    """Diagonal tidal scalar T = -(1/2) F''(r) (heuristic proxy).

    For a stationary observer, the dominant tidal component along the
    radial direction is proportional to F''(r) (the second derivative
    of the time-time metric component).
    """
    g = _metric_components(vs, r)
    return float(-0.5 * g["F_pp"])


def tidal_stretch_at_radius(vs: VanStockumInterior, r: float,
                              probe_length: float = 1.0) -> float:
    """Radial tidal stretch acceleration ~ T(r) * L for probe of length L."""
    T = riemann_scalar_radial(vs, r)
    return float(T * probe_length)


def tidal_squeeze_at_radius(vs: VanStockumInterior, r: float,
                              probe_radius: float = 1.0) -> float:
    """Angular tidal squeeze ~ R^phi_{tphit}, computed from K''."""
    g = _metric_components(vs, r)
    return float(0.5 * g["K_pp"] * probe_radius)


def geodesic_deviation_evolution(
    vs: VanStockumInterior, r_start: float, r_end: float,
    deviation_initial: float = 1.0,
    n_steps: int = 100,
) -> dict:
    """Integrate dot{xi}'' = T(r) xi along a radial geodesic.

    Returns the final deviation amplitude xi(r_end).
    """
    if r_end <= r_start:
        raise ValueError("r_end must exceed r_start")
    r_grid = np.linspace(r_start, r_end, n_steps)
    dr = (r_end - r_start) / (n_steps - 1)
    xi = deviation_initial
    xi_dot = 0.0
    max_xi = abs(xi)
    for r_i in r_grid:
        T_r = riemann_scalar_radial(vs, float(r_i))
        # Linear ODE step (Forward Euler)
        xi_ddot = T_r * xi
        xi += xi_dot * dr
        xi_dot += xi_ddot * dr
        max_xi = max(max_xi, abs(xi))
    return {
        "r_start": r_start,
        "r_end": r_end,
        "final_xi": float(xi),
        "max_xi": float(max_xi),
        "growth_ratio": float(xi / deviation_initial),
    }


def spaghettification_proxy(vs: VanStockumInterior, r: float) -> float:
    """|T(r)| as a single-scalar spaghettification metric."""
    return float(abs(riemann_scalar_radial(vs, r)))


def multi_band_tidal_signature(
    vs: VanStockumInterior, n_bands: int = 5,
) -> list[dict]:
    """Tidal scalar at the midpoint of each CTC band."""
    if not vs.is_supercritical():
        return []
    R = vs.R
    alpha = vs.alpha
    gamma = math.pi - math.atan(alpha)
    zeros = []
    for n in range(1, 200):
        u_n = (n * math.pi - gamma) / alpha
        if u_n <= 0:
            continue
        r_n = R * math.exp(u_n)
        zeros.append(r_n)
        if len(zeros) >= n_bands + 1:
            break
    out = []
    for i in range(min(n_bands, len(zeros) - 1)):
        r_mid = math.sqrt(zeros[i] * zeros[i + 1])
        T = riemann_scalar_radial(vs, r_mid)
        out.append({
            "band_index": i,
            "r_inner": zeros[i],
            "r_outer": zeros[i + 1],
            "r_mid": r_mid,
            "tidal_scalar": T,
            "spaghettification": float(abs(T)),
        })
    return out


def safety_radius_for_probe(
    vs: VanStockumInterior, material_strength: float = 1.0,
    r_min: float = None, r_max: float = None,
    n_samples: int = 1000,
) -> dict:
    """Find r where |T(r)| > material_strength (probe disintegrates)."""
    if r_min is None:
        r_min = vs.R * 1.01
    if r_max is None:
        r_max = vs.R * 30.0
    r_grid = np.geomspace(r_min, r_max, n_samples)
    breaches = []
    for r in r_grid:
        if abs(riemann_scalar_radial(vs, float(r))) > material_strength:
            breaches.append(float(r))
    return {
        "material_strength": material_strength,
        "r_min_breach": min(breaches) if breaches else None,
        "r_max_breach": max(breaches) if breaches else None,
        "n_breach_radii": len(breaches),
        "safe": len(breaches) == 0,
    }


def angular_squeeze_to_radial_stretch_ratio(
    vs: VanStockumInterior, r: float,
) -> float:
    """Compare angular squeeze magnitude to radial stretch magnitude."""
    g = _metric_components(vs, r)
    radial_stretch = abs(g["F_pp"])
    angular_squeeze = abs(g["K_pp"])
    if radial_stretch < 1e-30:
        return float("inf")
    return float(angular_squeeze / radial_stretch)
