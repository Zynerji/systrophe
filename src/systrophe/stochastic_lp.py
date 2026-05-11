"""Brownian motion / Langevin dynamics on the LP background.

Treat the LP exterior as a noisy environment in which a particle
undergoes stochastic radial motion. The Langevin equation
    dr/dt = -V_eff'(r) + sqrt(2 D) xi(t)
with V_eff(r) ~ -1/F(r) (drift toward chronology horizons) governs
the particle's diffusion across the radial direction.

Predictions:
- Mean first-passage time to a chronology horizon
- Stationary distribution rho_ss(r) ~ exp(-V_eff / D)
- Diffusion across multiple CTC bands

Module:
- diffusion_coefficient_estimate: D ~ T_Hawking
- mean_first_passage_to_CH: <T_fp> from r_0 to nearest CH
- stationary_distribution: rho(r) on a grid
- escape_probability_in_finite_time
- novelty_scan: catcher on first-passage distribution
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class StochasticResult:
    r_start: float
    r_target: float
    diffusion_coeff: float
    mean_first_passage_time: float


def diffusion_coefficient_estimate(vs: VanStockumInterior,
                                     r: float = None) -> float:
    """Heuristic D ~ Hawking T at nearest CH (sets the noise scale)."""
    if r is None:
        r = vs.R * 2.0
    eps = 1e-4 * r
    F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    Fp = (F_plus - F_minus) / (2 * eps)
    return float(abs(Fp) / (4 * math.pi))


def mean_first_passage_to_CH(
    vs: VanStockumInterior, r_start: float,
    diffusion: float = None,
) -> StochasticResult:
    """For a Brownian particle starting at r_start, mean time to reach the
    nearest chronology horizon.

    Kramers escape: <T> ~ (1/D) * integral exp(V/D) dr.
    """
    if diffusion is None:
        diffusion = diffusion_coefficient_estimate(vs, r_start)
    # Find nearest CH above r_start
    if not vs.is_supercritical():
        return StochasticResult(r_start=r_start, r_target=float("inf"),
                                diffusion_coeff=diffusion,
                                mean_first_passage_time=float("inf"))
    R = vs.R
    alpha = vs.alpha
    gamma_c = math.pi - math.atan(alpha)
    nearest_ch = None
    for n in range(1, 200):
        u_n = (n * math.pi - gamma_c) / alpha
        if u_n <= 0:
            continue
        r_n = R * math.exp(u_n)
        if r_n > r_start:
            nearest_ch = r_n
            break
    if nearest_ch is None:
        return StochasticResult(r_start=r_start, r_target=float("inf"),
                                diffusion_coeff=diffusion,
                                mean_first_passage_time=float("inf"))
    # Heuristic Kramers: T ~ exp(V_barrier / D) / D
    # V_barrier ~ 1/|F'(midpoint)|
    r_mid = (r_start + nearest_ch) / 2
    F_mid = float(vs.analytic_exterior_F(np.array([r_mid]))[0])
    if F_mid <= 0:
        T_fp = (nearest_ch - r_start) ** 2 / max(diffusion, 1e-30)
    else:
        barrier = 1.0 / max(F_mid, 1e-12)
        T_fp = math.exp(min(barrier / max(diffusion, 1e-30), 700)) / max(diffusion, 1e-30)
    return StochasticResult(
        r_start=r_start, r_target=nearest_ch,
        diffusion_coeff=diffusion,
        mean_first_passage_time=float(T_fp),
    )


def stationary_distribution(
    vs: VanStockumInterior,
    r_grid: np.ndarray = None,
    diffusion: float = 0.1,
) -> dict:
    """Stationary rho(r) ~ exp(-V_eff/D); normalised."""
    if r_grid is None:
        r_grid = np.linspace(vs.R * 1.05, vs.R * 5.0, 200)
    rho_unnormed = []
    for r in r_grid:
        F = float(vs.analytic_exterior_F(np.array([r]))[0])
        V = 1.0 / max(abs(F), 1e-12)
        exponent = -V / max(diffusion, 1e-30)
        if exponent < -700:
            rho_unnormed.append(0.0)
        else:
            rho_unnormed.append(math.exp(exponent))
    rho_arr = np.asarray(rho_unnormed)
    total = float(np.trapezoid(rho_arr, r_grid))
    if total > 0:
        rho_norm = rho_arr / total
    else:
        rho_norm = rho_arr
    return {
        "r_grid": r_grid,
        "rho_stationary": rho_norm,
        "diffusion": diffusion,
    }


def escape_probability_in_finite_time(
    vs: VanStockumInterior, r_start: float, T_window: float,
    diffusion: float = None,
) -> float:
    """1 - exp(-T / <T_fp>)."""
    res = mean_first_passage_to_CH(vs, r_start, diffusion)
    if not math.isfinite(res.mean_first_passage_time):
        return 0.0
    return float(1.0 - math.exp(-T_window / res.mean_first_passage_time))


def novelty_scan(n_r_values: int = 30) -> dict:
    """Catcher: first-passage time vs r_start should show log-periodic
    structure (peaks/valleys mirroring the CTC band spacing)."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return {"verdict": "subcritical", "sharp_features": []}
    r_grid = np.linspace(1.05 * vs.R, 5.0 * vs.R, n_r_values)
    def fn(r_start: float) -> np.ndarray:
        res = mean_first_passage_to_CH(vs, r_start)
        if math.isfinite(res.mean_first_passage_time):
            return np.array([math.log10(max(res.mean_first_passage_time, 1e-30))])
        return np.array([30.0])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r_start")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
