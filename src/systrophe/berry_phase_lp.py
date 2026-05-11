"""Berry / geometric phase for wave packets on the LP background.

A quantum state |psi(R)> depending on slowly-varying parameters R(t)
acquires a geometric phase (Berry 1984)
   gamma = -i oint <psi | d_R psi> . dR
on a closed loop in parameter space. For an LP wave packet moving
adiabatically along a closed phi-orbit at radius r, the geometric
phase encodes the orbit's holonomy.

Module:
- berry_connection: A_phi(r) = <psi | d_phi psi>
- berry_phase_per_revolution: integral A_phi dphi from 0 to 2 pi
- berry_curvature: dA/dr
- topological_chern_number: integral berry_curvature over r-phi annulus
- pair_extinction_berry: at delta=pi, A -> 0, no phase
- novelty_scan
"""

from __future__ import annotations

import math

import numpy as np

from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior


def berry_connection(vs: VanStockumInterior, r: float) -> float:
    """A_phi(r) = K(r) / (2 sqrt(L(r))). Heuristic Berry connection
    on the phi-bundle over the cylinder."""
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    if abs(L) < 1e-12:
        return float("inf")
    return float(K / (2 * math.sqrt(abs(L))))


def berry_phase_per_revolution(vs: VanStockumInterior, r: float) -> float:
    """gamma = 2 pi A_phi(r) (Berry phase per phi-revolution)."""
    A = berry_connection(vs, r)
    if not math.isfinite(A):
        return float("inf")
    return float(2 * math.pi * A)


def berry_curvature(vs: VanStockumInterior, r: float) -> float:
    """F_r_phi = d A_phi / dr."""
    eps = 1e-4 * r
    A_plus = berry_connection(vs, r + eps)
    A_minus = berry_connection(vs, r - eps)
    if not (math.isfinite(A_plus) and math.isfinite(A_minus)):
        return float("inf")
    return float((A_plus - A_minus) / (2 * eps))


def topological_chern_number(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    n_samples: int = 200,
) -> float:
    """C = (1/2 pi) integral berry_curvature dr d phi over annulus."""
    r_grid = np.linspace(r_inner, r_outer, n_samples)
    integrand = []
    for r in r_grid:
        F = berry_curvature(vs, float(r))
        integrand.append(F if math.isfinite(F) else 0.0)
    integral = float(np.trapezoid(integrand, r_grid)) * 2 * math.pi
    return integral / (2 * math.pi)


def pair_extinction_berry(
    vs: VanStockumInterior, r: float, delta: float,
) -> dict:
    """Berry phase scales by (1+cos delta)/2 in pair geometry."""
    base = berry_phase_per_revolution(vs, r)
    extinction = 0.5 * (1.0 + math.cos(delta))
    return {
        "r": r,
        "delta": delta,
        "extinction_factor": extinction,
        "gamma_single": base,
        "gamma_pair": float(base * extinction) if math.isfinite(base) else float("inf"),
        "phase_trivial_at_pi": bool(abs(delta - math.pi) < 1e-9),
    }


def novelty_scan(n_radii: int = 30) -> dict:
    """Catcher on Berry phase / curvature profile."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05 * vs.R, 10 * vs.R, n_radii)
    def fn(r: float) -> np.ndarray:
        gamma = berry_phase_per_revolution(vs, r)
        A = berry_connection(vs, r)
        F = berry_curvature(vs, r)
        return np.array([
            gamma if math.isfinite(gamma) else 1e6,
            A if math.isfinite(A) else 1e6,
            F if math.isfinite(F) else 1e6,
        ])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
