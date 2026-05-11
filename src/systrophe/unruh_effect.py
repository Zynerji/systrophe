"""Unruh effect for accelerated observers on the LP background.

An observer with proper acceleration `a_acc` sees a thermal bath at
the Unruh temperature T_U = a_acc * hbar / (2 pi c k_B). For an
observer rotating about the LP cylinder axis at proper-centripetal
acceleration, this gives an angular-velocity-dependent thermal
spectrum that combines with the cylinder's frame dragging.

Module:
- unruh_temperature: T_U = a / (2 pi)
- centripetal_acceleration_at_orbit: a_circ(r) for a circular orbit
- combined_unruh_hawking_T: addition formula
- frame_drag_modified_T: rotation correction
- novelty_scan: catcher on T(r) profile
"""

from __future__ import annotations

import math

import numpy as np

from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior


def unruh_temperature(acceleration: float) -> float:
    """T_U = a / (2 pi) (natural units)."""
    if acceleration < 0:
        return 0.0
    return float(acceleration / (2 * math.pi))


def centripetal_acceleration_at_orbit(
    vs: VanStockumInterior, r: float, angular_velocity: float = 1.0,
) -> float:
    """Proper centripetal acceleration: a = Omega^2 r / sqrt(F)."""
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    if F <= 0:
        return float("inf")
    return float(angular_velocity ** 2 * r / math.sqrt(F))


def combined_unruh_hawking_T(
    vs: VanStockumInterior, r: float, angular_velocity: float = 1.0,
) -> dict:
    """Combine Unruh T from observer rotation + Hawking T from CH.

    Heuristic addition: T_combined = sqrt(T_U^2 + T_H^2).
    """
    a_acc = centripetal_acceleration_at_orbit(vs, r, angular_velocity)
    T_U = unruh_temperature(a_acc) if math.isfinite(a_acc) else float("inf")
    # Hawking T at nearest CH
    if not vs.is_supercritical():
        T_H = 0.0
    else:
        R = vs.R
        alpha = vs.alpha
        gamma_c = math.pi - math.atan(alpha)
        r_ch = None
        for n in range(1, 200):
            u_n = (n * math.pi - gamma_c) / alpha
            if u_n <= 0:
                continue
            r_n = R * math.exp(u_n)
            if r_n > r:
                r_ch = r_n
                break
        if r_ch is None:
            T_H = 0.0
        else:
            eps = 1e-4 * r_ch
            F_plus = float(vs.analytic_exterior_F(np.array([r_ch + eps]))[0])
            F_minus = float(vs.analytic_exterior_F(np.array([r_ch - eps]))[0])
            Fp = (F_plus - F_minus) / (2 * eps)
            T_H = abs(Fp) / (4 * math.pi)
    T_combined = math.sqrt(T_U ** 2 + T_H ** 2) if math.isfinite(T_U) else T_U
    return {
        "r": r,
        "T_Unruh": float(T_U),
        "T_Hawking": float(T_H),
        "T_combined": float(T_combined),
    }


def frame_drag_modified_T(
    vs: VanStockumInterior, r: float,
) -> float:
    """T modulated by frame-dragging factor |K(r) / r^2|."""
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    return float(abs(K) / (2 * math.pi * r ** 2))


def comparison_to_Unruh_DeWitt(
    acceleration: float, omega_field: float = 1.0,
) -> dict:
    """Unruh-DeWitt detector transition rate at given a and omega."""
    T_U = unruh_temperature(acceleration)
    if T_U <= 0:
        return {"rate": 0.0, "T_U": T_U}
    rate = omega_field / (math.exp(omega_field / T_U) - 1)
    return {
        "acceleration": acceleration,
        "T_U": T_U,
        "omega_field": omega_field,
        "rate": float(rate),
    }


def novelty_scan(n_r_values: int = 30) -> dict:
    """Catcher: T(r) profile combining Unruh + Hawking + frame-drag."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05 * vs.R, 10 * vs.R, n_r_values)
    def fn(r: float) -> np.ndarray:
        res = combined_unruh_hawking_T(vs, r)
        return np.array([
            min(res["T_Unruh"], 1e6),
            res["T_Hawking"],
            min(res["T_combined"], 1e6),
        ])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
