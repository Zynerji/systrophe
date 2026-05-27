"""One-loop quantum back-reaction on the LP metric.

A massless scalar field on the LP background induces a one-loop
quantum stress tensor that modifies the Einstein equations:
    R_mu_nu - (1/2) g_mu_nu R = 8 pi G * <T_mu_nu>_quantum

The leading effect is a renormalization of F(r):
    F_corrected(r) = F_classical(r) + epsilon * delta_F_1loop(r)

with epsilon = hbar * G / R^2 (dimensionless small parameter).

This module:
- one_loop_F_correction: delta_F(r)
- corrected_F: F + epsilon * delta_F
- shifted_chronology_horizons: how F=0 surfaces move
- back_reaction_to_alpha: induced shift in log-frequency
- novelty_scan: catcher on delta_F profile
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior

EPSILON_DEFAULT = 0.01  # heuristic small parameter


def one_loop_F_correction(vs: VanStockumInterior, r: float) -> float:
    """delta_F_1loop(r) ~ (1/24 pi^2) * F''(r) / R^2 (Polyakov-like).

    Heuristic curvature-squared contribution.
    """
    eps = 1e-4 * r
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    Fpp = (F_plus + F_minus - 2 * F) / (eps * eps)
    return float(Fpp / (24 * math.pi ** 2 * vs.R ** 2))


def corrected_F(vs: VanStockumInterior, r: float,
                 epsilon: float = EPSILON_DEFAULT) -> float:
    """F + epsilon * delta_F."""
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    delta = one_loop_F_correction(vs, r)
    return float(F + epsilon * delta)


def shifted_chronology_horizons(
    vs: VanStockumInterior, epsilon: float = EPSILON_DEFAULT,
    r_min: float = None, r_max: float = None, n_samples: int = 2000,
) -> dict:
    """Find r where corrected_F = 0 (shifted chronology horizons)."""
    if r_min is None:
        r_min = vs.R * 1.01
    if r_max is None:
        r_max = vs.R * 20.0
    if not vs.is_supercritical():
        return {"shifted_horizons": [], "regime": "subcritical"}
    r_grid = np.linspace(r_min, r_max, n_samples)
    F_corr = np.array([corrected_F(vs, float(r), epsilon) for r in r_grid])
    # Find sign changes
    crossings = []
    for i in range(len(F_corr) - 1):
        if F_corr[i] * F_corr[i + 1] < 0:
            # Linear interpolation
            t = F_corr[i] / (F_corr[i] - F_corr[i + 1])
            r_cross = r_grid[i] + t * (r_grid[i + 1] - r_grid[i])
            crossings.append(float(r_cross))
    return {
        "shifted_horizons": crossings,
        "n_shifted": len(crossings),
        "epsilon": epsilon,
        "regime": "supercritical",
    }


def back_reaction_to_alpha(
    vs: VanStockumInterior, epsilon: float = EPSILON_DEFAULT,
) -> dict:
    """Estimate corrected alpha from shifted CH spacing.

    For classical LP: r_{n+1}/r_n = exp(pi/alpha).
    For corrected: compute new ratio from shifted horizons.
    """
    res = shifted_chronology_horizons(vs, epsilon=epsilon)
    if len(res["shifted_horizons"]) < 2:
        return {"alpha_classical": vs.alpha if vs.is_supercritical() else 0.0,
                "alpha_corrected": float("nan"),
                "relative_shift": float("nan")}
    r1 = res["shifted_horizons"][0]
    r2 = res["shifted_horizons"][1]
    if r2 <= r1:
        return {"alpha_classical": vs.alpha if vs.is_supercritical() else 0.0,
                "alpha_corrected": float("nan"),
                "relative_shift": float("nan")}
    alpha_corr = math.pi / math.log(r2 / r1)
    alpha_class = vs.alpha if vs.is_supercritical() else 0.0
    rel_shift = (alpha_corr - alpha_class) / max(alpha_class, 1e-30)
    return {
        "alpha_classical": float(alpha_class),
        "alpha_corrected": float(alpha_corr),
        "relative_shift": float(rel_shift),
    }


def trace_anomaly_at_r(vs: VanStockumInterior, r: float) -> float:
    """<T_mu^mu> ~ c * R^{(2)} (conformal anomaly proxy)."""
    # Use F''/F as curvature proxy
    eps = 1e-4 * r
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    Fpp = (F_plus + F_minus - 2 * F) / (eps * eps)
    c_anomaly = 1.0 / 24.0  # conformal anomaly coefficient for a massless scalar
    return float(c_anomaly * Fpp / max(abs(F), 1e-12))


def novelty_scan(n_radii: int = 30) -> dict:
    """Catcher on delta_F(r) profile."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05 * vs.R, 10 * vs.R, n_radii)
    def fn(r: float) -> np.ndarray:
        return np.array([one_loop_F_correction(vs, r),
                         corrected_F(vs, r),
                         trace_anomaly_at_r(vs, r)])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
