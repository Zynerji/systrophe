"""Casimir-Polder force on an atom near the rotating cylinder.

A polarizable atom at distance d from a conducting cylinder
experiences a Casimir-Polder force scaling as 1/d^5 (close range) or
1/d^5 with thermal corrections (long range). For a rotating cylinder,
frame-dragging modifies the photon density-of-states near the surface,
giving a rotation-dependent correction.

Module:
- casimir_polder_force_static: standard 1/d^5 result
- casimir_polder_with_rotation: rotation-modified force
- frame_drag_correction: |1 - omega^2 r^2|^{-1/2} factor
- distance_dependence_exponent_in_band: deviates from 5 inside CTC band
- novelty_scan: catcher on force(d) profile
"""

from __future__ import annotations

import math

import numpy as np

from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior

HBAR_C_OVER_4PI = 1.0 / (4 * math.pi)  # natural units


def casimir_polder_force_static(
    distance: float, polarizability: float = 1.0,
) -> float:
    """Standard CP force: F = -3 hbar c alpha / (2 pi d^5)."""
    if distance <= 0:
        return float("inf")
    return float(-3 * polarizability * HBAR_C_OVER_4PI / distance ** 5)


def frame_drag_correction(vs: VanStockumInterior, r: float) -> float:
    """1 / sqrt(F(r)) frame-dragging factor (heuristic)."""
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    if abs(F) < 1e-12:
        return float("inf")
    return float(1.0 / math.sqrt(abs(F)))


def casimir_polder_with_rotation(
    vs: VanStockumInterior, distance: float,
    polarizability: float = 1.0,
) -> dict:
    """Rotation-modified CP force at radius r = R + distance."""
    if distance <= 0:
        return {"force": float("inf")}
    r = vs.R + distance
    F_static = casimir_polder_force_static(distance, polarizability)
    correction = frame_drag_correction(vs, r)
    F_modified = F_static * correction
    return {
        "distance": distance,
        "r_total": r,
        "F_static": float(F_static),
        "frame_drag_correction": float(correction),
        "F_modified": float(F_modified),
    }


def distance_dependence_exponent_in_band(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    n_samples: int = 50,
) -> float:
    """Fit F(d) ~ -A / d^n inside the band; return n."""
    d_grid = np.linspace(0.01, r_outer - r_inner, n_samples)
    forces = []
    for d in d_grid:
        res = casimir_polder_with_rotation(vs, float(d))
        forces.append(abs(res["F_modified"]))
    forces = np.asarray(forces)
    # Filter out infinities
    mask = np.isfinite(forces) & (forces > 0)
    if np.sum(mask) < 3:
        return float("nan")
    # Linear fit in log-log space
    log_d = np.log(d_grid[mask])
    log_F = np.log(forces[mask])
    slope, _ = np.polyfit(log_d, log_F, 1)
    return float(-slope)


def thermal_correction_at_T(
    vs: VanStockumInterior, distance: float, temperature: float,
    polarizability: float = 1.0,
) -> dict:
    """Thermal Casimir at finite T: corrections in d / lambda_T."""
    if temperature <= 0:
        return {"correction_factor": 1.0, "regime": "zero_temperature"}
    lambda_T = 1.0 / temperature  # thermal wavelength
    ratio = distance / lambda_T
    if ratio < 0.1:
        regime = "close_range"
        correction = 1.0
    elif ratio < 10:
        regime = "intermediate"
        correction = 1.0 + 0.5 * ratio  # heuristic
    else:
        regime = "thermal"
        correction = 4 * math.pi * ratio / 3
    return {
        "temperature": temperature,
        "thermal_wavelength": lambda_T,
        "d_over_lambda_T": ratio,
        "regime": regime,
        "correction_factor": float(correction),
    }


def novelty_scan(n_d_values: int = 30) -> dict:
    """Catcher on force(d) profile."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    d_grid = np.linspace(0.01, 5.0, n_d_values)
    def fn(d: float) -> np.ndarray:
        res = casimir_polder_with_rotation(vs, d)
        if math.isfinite(res["F_modified"]):
            return np.array([math.log10(max(abs(res["F_modified"]), 1e-30))])
        return np.array([30.0])
    result = scan_novelty(d_grid, fn, n_bits=32, parameter_label="distance")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
