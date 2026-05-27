"""Holographic complexity on the LP region.

Complexity = Volume (CV duality): C ~ V_max / G_N L_AdS, where V_max
is the maximal volume in the bulk anchored on a boundary slice.

Complexity = Action (CA duality): C ~ I_WdW / (pi hbar), where I_WdW
is the gravitational action on the Wheeler-DeWitt patch.

For the LP enclosed region (between two CHs), we compute both
proxies and check for the log-periodic structure.

Module:
- volume_proxy_C: V_max in CTC band -> complexity proxy
- action_proxy_C: I on WdW patch
- complexity_growth_rate: dC/dt
- pair_extinction_complexity
- novelty_scan
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior


def volume_proxy_C(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    cylinder_length: float = 1.0, G_newton: float = 1.0,
) -> float:
    """C_V = V_max / G_N. V_max = integral dr * sqrt(|L|) * 2 pi * length."""
    r_grid = np.linspace(r_inner, r_outer, 200)
    L_vals = vs.analytic_exterior_L(r_grid)
    integrand = np.sqrt(np.abs(L_vals))
    V_max = float(np.trapezoid(integrand, r_grid)) * 2 * math.pi * cylinder_length
    return float(V_max / max(G_newton, 1e-30))


def action_proxy_C(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    cylinder_length: float = 1.0,
) -> float:
    """C_A ~ I_WdW / pi. Heuristic: I = integral (Ricci scalar proxy) dV."""
    r_grid = np.linspace(r_inner, r_outer, 200)
    eps = 1e-4
    integrand = []
    for r in r_grid:
        F = float(vs.analytic_exterior_F(np.array([r]))[0])
        F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
        F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
        Fpp = (F_plus + F_minus - 2 * F) / (eps * eps)
        L = float(vs.analytic_exterior_L(np.array([r]))[0])
        ricci_proxy = Fpp / max(abs(F), 1e-12)
        integrand.append(ricci_proxy * math.sqrt(abs(L)))
    I = float(np.trapezoid(integrand, r_grid)) * 2 * math.pi * cylinder_length
    return float(I / math.pi)


def complexity_growth_rate(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    time_step: float = 1.0,
) -> float:
    """Heuristic dC/dt for the enclosed region."""
    return float(volume_proxy_C(vs, r_inner, r_outer) / max(time_step, 1e-30))


def pair_extinction_complexity(
    vs: VanStockumInterior, r_inner: float, r_outer: float, delta: float,
) -> dict:
    """Both proxies scale by (1+cos delta)/2 in pair geometry."""
    CV = volume_proxy_C(vs, r_inner, r_outer)
    CA = action_proxy_C(vs, r_inner, r_outer)
    extinction = 0.5 * (1.0 + math.cos(delta))
    return {
        "r_inner": r_inner,
        "r_outer": r_outer,
        "delta": delta,
        "extinction_factor": extinction,
        "C_V_single": CV,
        "C_V_pair": float(CV * extinction),
        "C_A_single": CA,
        "C_A_pair": float(CA * extinction),
    }


def novelty_scan(n_band_widths: int = 30) -> dict:
    """Catcher on complexity vs band width."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    width_grid = np.linspace(0.1, 5.0, n_band_widths)
    r_base = 1.5
    def fn(width: float) -> np.ndarray:
        CV = volume_proxy_C(vs, r_base, r_base + width)
        CA = action_proxy_C(vs, r_base, r_base + width)
        return np.array([
            CV if math.isfinite(CV) else 1e6,
            CA if math.isfinite(CA) else 1e6,
        ])
    result = scan_novelty(width_grid, fn, n_bits=32, parameter_label="band_width")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
