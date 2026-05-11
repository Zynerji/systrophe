"""Soliton solutions on the LP rotating-cylinder background.

A scalar phi^4 field on the LP exterior admits soliton solutions
(kinks, bounces, vortices) when the effective potential supports them.
The cylinder's frame-dragging modifies the soliton mass and stability.

Module:
- kink_profile: 1D kink phi(r) = tanh(r-r_0)
- kink_mass_on_lp: M_kink with frame-drag correction
- vortex_winding_number: integer charge for cylindrical vortices
- bound_state_spectrum_in_band
- pair_extinction_soliton_decay: at delta=pi, solitons unbound
- novelty_scan: catcher on mass spectrum vs winding
"""

from __future__ import annotations

import math

import numpy as np

from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior


def kink_profile(r: np.ndarray, r_0: float = 2.0, width: float = 0.5) -> np.ndarray:
    """phi(r) = tanh((r - r_0) / width). Classic kink."""
    return np.tanh((r - r_0) / max(width, 1e-12))


def kink_mass_on_lp(
    vs: VanStockumInterior, r_0: float = 2.0, width: float = 0.5,
    n_samples: int = 200,
) -> float:
    """Kink mass with frame-drag correction: M = M_0 / sqrt(F(r_0))."""
    F = float(vs.analytic_exterior_F(np.array([r_0]))[0])
    if F <= 0:
        return float("inf")
    M_flat = 2 * math.sqrt(2) / (3 * width)  # phi^4 kink mass
    return float(M_flat / math.sqrt(F))


def vortex_winding_number(theta_field: np.ndarray) -> int:
    """For a closed-loop field theta(s), winding = (1/2pi) integral d theta."""
    phases = np.unwrap(theta_field)
    if len(phases) < 2:
        return 0
    total = phases[-1] - phases[0]
    return int(round(total / (2 * math.pi)))


def bound_state_spectrum_in_band(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    n_modes: int = 5,
) -> list[float]:
    """Approximate bound-state energy levels in a CTC band.

    Heuristic: E_n = n * pi / (r_outer - r_inner) (particle in box).
    """
    L = r_outer - r_inner
    if L <= 0:
        return []
    return [float(n * math.pi / L) for n in range(1, n_modes + 1)]


def pair_extinction_soliton_decay(
    vs: VanStockumInterior, r_0: float, delta: float,
) -> dict:
    """Soliton mass scales by (1+cos delta)/2; at delta=pi, kink dissolves."""
    base = kink_mass_on_lp(vs, r_0)
    extinction = 0.5 * (1.0 + math.cos(delta))
    mass_pair = base * extinction
    return {
        "r_0": r_0,
        "delta": delta,
        "extinction_factor": extinction,
        "M_single": base,
        "M_pair": float(mass_pair),
        "soliton_dissolved": bool(extinction < 1e-9),
    }


def topological_charge(
    field: np.ndarray, x_grid: np.ndarray,
) -> int:
    """Q = (1/2 pi) integral dx (d phi / dx).

    For kink phi = tanh, Q = 1.
    """
    if len(field) < 2:
        return 0
    dphi = np.diff(field)
    delta_phi = float(np.sum(dphi))
    # For a kink: phi from -1 to +1, delta = 2; topological charge = delta / 2
    return int(round(delta_phi / 2.0))


def novelty_scan(n_r0_values: int = 30) -> dict:
    """Catcher on M_kink(r_0) profile."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r0_grid = np.linspace(1.1 * vs.R, 10 * vs.R, n_r0_values)
    def fn(r0: float) -> np.ndarray:
        M = kink_mass_on_lp(vs, float(r0))
        if math.isfinite(M):
            return np.array([math.log10(max(M, 1e-30))])
        return np.array([30.0])
    result = scan_novelty(r0_grid, fn, n_bits=32, parameter_label="r0")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
