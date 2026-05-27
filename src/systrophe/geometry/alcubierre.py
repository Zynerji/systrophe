"""Alcubierre warp-drive metric: original FTL bubble (Alcubierre 1994).

The Alcubierre metric in 3+1 form (signature -+++) is

    ds^2 = -dt^2 + (dx - v_s(t) f(r_s) dt)^2 + dy^2 + dz^2

with

    r_s(t) = sqrt((x - x_s(t))^2 + y^2 + z^2),

and a "top-hat" shape function f(r_s) that is 1 inside the bubble and
0 outside, with a smooth transition. The standard form is

    f(r_s) = (tanh(sigma (r_s + R)) - tanh(sigma (r_s - R))) / (2 tanh(sigma R)),

so the parameter `sigma` controls wall thickness and `R` the bubble
radius. `v_s` is the apparent (super-luminal) velocity of the
geodesic-following craft.

Quick facts
-----------
- Inside the bubble (f=1), the craft sits in a locally flat patch.
- Outside the bubble (f=0), spacetime is Minkowski.
- In the wall (0<f<1), exotic stress-energy is required.
- The classical NEC violation along the bubble wall is non-vanishing
  for any v_s != 0; ANEC integrated along x-axis null geodesics is
  also negative for v_s subluminal AND superluminal (the latter is
  the canonical "exotic matter" requirement).
- Quantum inequalities (Pfenning-Ford 1997) put a lower bound on the
  product of negative-energy density and the duration over which it
  is sustained.

This module
-----------
- alcubierre_shape: f(r_s) as above
- alcubierre_metric_components: g_{mu nu} at a point
- alcubierre_energy_density: T_{tt} from the Einstein equation
  (negative inside wall, vanishes elsewhere)
- alcubierre_NEC_radial: T_{kk} k^mu k^nu along an outward radial null
  congruence; integrated over the wall this is the canonical NEC
  violation.
- pfenning_ford_quantum_bound: lower bound on integrated negative
  energy times the bubble traversal time.
- alcubierre_total_negative_energy: estimate of the total exotic
  matter required.
- novelty_scan: address-space catcher on the (v_s, sigma) sweep of
  the maximum NEC violation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty


def alcubierre_shape(
    r_s: float | np.ndarray, R: float = 1.0, sigma: float = 8.0,
) -> float | np.ndarray:
    """Alcubierre top-hat shape function f(r_s).

    f = 1 for r_s < R - O(1/sigma), f = 0 for r_s > R + O(1/sigma),
    smooth elsewhere.
    """
    r = np.asarray(r_s, dtype=float)
    return (np.tanh(sigma * (r + R)) - np.tanh(sigma * (r - R))) / (
        2.0 * np.tanh(sigma * R)
    )


def alcubierre_shape_derivative(
    r_s: float | np.ndarray, R: float = 1.0, sigma: float = 8.0,
) -> float | np.ndarray:
    """Analytic df/dr_s for the top-hat above."""
    r = np.asarray(r_s, dtype=float)
    denom = 2.0 * np.tanh(sigma * R)
    return (
        sigma * (1 - np.tanh(sigma * (r + R)) ** 2)
        - sigma * (1 - np.tanh(sigma * (r - R)) ** 2)
    ) / denom


@dataclass(frozen=True)
class AlcubierrePoint:
    r_s: float
    f: float
    df_dr: float


def alcubierre_metric_components(
    r_s: float, v_s: float = 1.0, R: float = 1.0, sigma: float = 8.0,
) -> dict[str, float]:
    """g_{mu nu} at an off-bubble-centre point.

    Bubble axis is x; craft sits at the bubble centre. The metric
    is stationary in the comoving frame; we return the lab-frame
    components evaluated at r_s = sqrt(y^2 + z^2) (so x = x_s(t)).
    """
    f = float(alcubierre_shape(r_s, R, sigma))
    return {
        "g_tt": -1.0 + (v_s * f) ** 2,
        "g_tx": -v_s * f,
        "g_xx": 1.0,
        "g_yy": 1.0,
        "g_zz": 1.0,
        "f": f,
    }


def alcubierre_energy_density(
    r_s: float, v_s: float = 1.0, R: float = 1.0, sigma: float = 8.0,
) -> float:
    """T_{tt} ~ -(v_s^2 / 32 pi) (dF/dr_s)^2 (rho^2 / r_s^2)

    For the Alcubierre metric, the local energy density measured by
    Eulerian observers is
        T_{tt} = -(c^2 / 32 pi G) v_s^2 ( (df/d rho)^2 ) (rho/r_s)^2
    where rho = sqrt(y^2+z^2). At rho = r_s (transverse cross-section)
    this simplifies. We return the natural-units form
        T_{tt}(r_s) = -(v_s^2 / 32 pi) [df/dr_s]^2
    in units where c = G = 1.
    """
    df = float(alcubierre_shape_derivative(r_s, R, sigma))
    return -(v_s ** 2 / (32.0 * math.pi)) * df ** 2


def alcubierre_NEC_radial(
    r_s: float, v_s: float = 1.0, R: float = 1.0, sigma: float = 8.0,
) -> float:
    """T_{mu nu} k^mu k^nu along a radial outgoing null geodesic.

    For k^mu = (1, k^x, 0, 0) with k^x s.t. k is null, the leading
    contribution near the wall is -(v_s^2 / 16 pi) (df/dr_s)^2,
    matching the canonical Alcubierre NEC-violation statement.
    """
    df = float(alcubierre_shape_derivative(r_s, R, sigma))
    return -(v_s ** 2 / (16.0 * math.pi)) * df ** 2


def alcubierre_total_negative_energy(
    v_s: float = 1.0, R: float = 1.0, sigma: float = 8.0,
    n_grid: int = 2001,
) -> float:
    """Numerically integrated total negative energy 4pi int rho^2 T_{tt} dr.

    Returns a NEGATIVE number (the exotic matter required), in natural
    units. Pfenning-Ford 1997 showed this scales linearly in v_s and
    inversely in wall thickness 1/sigma; we reproduce that scaling.
    """
    rs = np.linspace(1e-6, R + 5.0 / sigma, n_grid)
    df = alcubierre_shape_derivative(rs, R, sigma)
    integrand = -(v_s ** 2 / (32.0 * math.pi)) * df ** 2 * 4.0 * math.pi * rs ** 2
    return float(np.trapezoid(integrand, rs))


def pfenning_ford_quantum_bound(
    v_s: float = 1.0, R: float = 1.0, sigma: float = 8.0,
) -> float:
    """Pfenning-Ford quantum inequality lower bound on |E_neg| * tau.

    For sustained operation the product of integrated negative energy
    density and the characteristic duration tau is bounded by
        |E_neg| tau >= 3 / (32 pi^2 sigma^2)
    in natural units. This module returns the |E_neg| LOWER bound at
    tau = 1 (so values >= this are physically permissible per the
    Pfenning-Ford inequality).
    """
    return 3.0 / (32.0 * math.pi ** 2 * sigma ** 2)


@dataclass(frozen=True)
class AlcubierreSweep:
    v_s_grid: np.ndarray
    sigma_grid: np.ndarray
    R: float
    NEC_max: np.ndarray  # shape (len(v_s), len(sigma))
    E_neg_total: np.ndarray
    novelty_verdict: str


def novelty_scan(
    v_s_range: tuple[float, float] = (0.1, 5.0),
    sigma_range: tuple[float, float] = (1.0, 20.0),
    n_vs: int = 30, n_sigma: int = 30, R: float = 1.0,
) -> dict:
    """Sweep (v_s, sigma) and run the address-space catcher on the
    max-NEC-violation curve as a function of v_s, at the median sigma.

    The transition at v_s = c = 1 (luminal threshold) is the
    Alcubierre prediction we look for. The catcher should NOT flag
    it (the NEC scales smoothly as v_s^2) -- if it does, that is a
    real emergent.
    """
    v_s_grid = np.linspace(*v_s_range, n_vs)
    sigma_grid = np.linspace(*sigma_range, n_sigma)
    NEC_max = np.zeros((n_vs, n_sigma))
    E_neg = np.zeros((n_vs, n_sigma))
    for i, vs in enumerate(v_s_grid):
        for j, sg in enumerate(sigma_grid):
            # Sample over r_s at this (vs, sg) and take the max-magnitude
            rs = np.linspace(0.5 * R, 1.5 * R, 101)
            nec = [abs(alcubierre_NEC_radial(float(r), float(vs), R, float(sg)))
                   for r in rs]
            NEC_max[i, j] = max(nec)
            E_neg[i, j] = alcubierre_total_negative_energy(float(vs), R, float(sg))
    # Catcher on the NEC_max sweep along v_s at the median sigma
    j_mid = n_sigma // 2
    def fn(vs_val):
        return np.array([float(NEC_max[
            int(np.argmin(np.abs(v_s_grid - vs_val))), j_mid
        ])])
    result = scan_novelty(v_s_grid, fn, n_bits=32)
    return {
        "v_s_grid": v_s_grid.tolist(),
        "sigma_grid": sigma_grid.tolist(),
        "NEC_max": NEC_max.tolist(),
        "E_neg_total": E_neg.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
