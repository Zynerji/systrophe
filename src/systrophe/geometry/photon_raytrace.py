"""Photon ray tracing through the Tipler exterior.

Light bending and lensing in stationary axisymmetric spacetimes follow
from the null radial equation
    h(r) (r')^2 = -V_eff(r) = -(F l^2 - 2 K E l - L E^2) / r^2
and the conserved-quantity relations
    dphi/dr = (F l - E K) / [r^2 sqrt(-V_eff(r) / h(r))],
where the impact parameter b = l/E parameterises the family of null
geodesics.

The deflection angle of a photon coming in from r_max with impact
parameter b, reaching its perihelion r_min (where r' = 0), and going
out again is
    Delta phi = 2 * integral_{r_min}^{r_max} dphi/dr dr  -  pi.

For a Tipler / van Stockum supercritical exterior, F, K, L oscillate
log-periodically. The deflection angle inherits this oscillation:
photons with different impact parameters can experience qualitatively
different bending depending on which CTC band their perihelion falls in.

This module provides:
- `photon_perihelion`: numerically find the smallest r where V_eff = 0 for a given b.
- `photon_deflection_angle`: integrate the deflection along a null geodesic.
- `lensing_pattern`: sweep impact parameter and report the deflection profile.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _V_eff(F: float, K: float, L: float, r: float, E: float, ell: float) -> float:
    """Effective radial potential for a null geodesic, V_eff = (F l^2 - 2 K E l - L E^2)/r^2."""
    return (F * ell * ell - 2.0 * K * E * ell - L * E * E) / (r * r)


def photon_perihelion(
    F_fn: Callable[[float], float],
    K_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    E: float,
    ell: float,
    r_lower: float,
    r_upper: float,
    tol: float = 1e-9,
) -> float | None:
    """Find the perihelion r_min where V_eff(r) = 0 in [r_lower, r_upper].

    Returns the smallest r in the range satisfying V_eff = 0, or None if
    no zero exists. Uses a coarse grid + brentq bisection.
    """
    n = 401
    r_grid = np.linspace(r_lower, r_upper, n)
    V = np.array([_V_eff(F_fn(r), K_fn(r), L_fn(r), r, E, ell) for r in r_grid])
    sign_changes = np.where(np.diff(np.sign(V)) != 0)[0]
    for i in sign_changes:
        r_lo, r_hi = r_grid[i], r_grid[i + 1]
        try:
            r_zero = brentq(
                lambda r: _V_eff(F_fn(r), K_fn(r), L_fn(r), r, E, ell),
                r_lo, r_hi, xtol=tol, rtol=tol
            )
            return float(r_zero)
        except ValueError:
            continue
    return None


def photon_deflection_angle(
    F_fn: Callable[[float], float],
    K_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    h_fn: Callable[[float], float],
    E: float,
    ell: float,
    r_min: float,
    r_max: float,
) -> float:
    """Net deflection angle of a photon between perihelion r_min and r_max.

    Integrates dphi/dr from r_min outward to r_max along a null geodesic,
    doubles for the in- and out-going legs, and subtracts pi for an
    undeflected photon. A positive return value indicates net bending.

    The integrand is
        dphi/dr = (F l - E K) / [r^2 sqrt(-V_eff / h)],
    where V_eff < 0 in the photon's accessible region.
    """
    def dphi_dr(r):
        F = F_fn(r)
        K = K_fn(r)
        L = L_fn(r)
        h = h_fn(r)
        V = _V_eff(F, K, L, r, E, ell)
        denom = r * r * np.sqrt(max(-V / h, 1e-30))
        return (F * ell - E * K) / denom

    # Integrate from r_min + epsilon to avoid the integrable square-root singularity at perihelion
    eps = 1e-6 * max(r_min, 1.0)
    result, _ = quad(dphi_dr, r_min + eps, r_max, limit=200)
    return float(2.0 * abs(result) - np.pi)


def lensing_pattern(
    F_fn: Callable[[float], float],
    K_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    h_fn: Callable[[float], float],
    E: float,
    ell_array: np.ndarray,
    r_search_lo: float,
    r_search_hi: float,
    r_max: float,
) -> dict:
    """Sweep angular momentum (impact parameter) and report perihelion + deflection.

    For each ell in `ell_array`, find the photon perihelion in
    [r_search_lo, r_search_hi] and compute the deflection angle out to r_max.
    Returns a dict with arrays 'ell', 'b' (impact parameter), 'r_perihelion',
    and 'deflection_angle'.
    """
    n = len(ell_array)
    out = {
        "ell": np.asarray(ell_array, dtype=float),
        "b": np.asarray(ell_array, dtype=float) / E,
        "r_perihelion": np.full(n, np.nan),
        "deflection_angle": np.full(n, np.nan),
    }
    for i, ell in enumerate(ell_array):
        r_min = photon_perihelion(F_fn, K_fn, L_fn, E, float(ell), r_search_lo, r_search_hi)
        if r_min is None:
            continue
        out["r_perihelion"][i] = r_min
        out["deflection_angle"][i] = photon_deflection_angle(
            F_fn, K_fn, L_fn, h_fn, E, float(ell), r_min, r_max
        )
    return out
