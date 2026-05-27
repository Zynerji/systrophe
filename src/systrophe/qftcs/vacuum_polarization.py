"""QED vacuum polarization shifts on the LP rotating-cylinder background.

A massive charged scalar / spinor loop creates a vacuum-polarization
correction to the photon propagator: pi(q^2). On a curved background,
the spectrum and weighting of this correction shifts. The leading
result is the Heisenberg-Euler effective action, modified by frame
dragging.

Module:
- heisenberg_euler_shift: Lagrangian correction at field strength E
- effective_fine_structure_running: alpha(scale) on LP
- pair_production_threshold: critical field for breakdown
- lp_modified_E_critical
- novelty_scan: catcher on vacuum-pol profile
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior

ALPHA_FINE = 1.0 / 137.036  # fine-structure constant
E_CRITICAL_SCHWINGER = math.pi  # natural units, m^2 / e (schematic)


def heisenberg_euler_shift(electric_field: float,
                            alpha_qed: float = ALPHA_FINE) -> float:
    """delta_L = (alpha^2 / 90 pi^2 m^4) (E^2 + B^2)^2 (one-loop).

    For pure E and natural units: delta_L = (alpha^2 / 90 pi^2) E^4.
    """
    return float((alpha_qed ** 2 / (90 * math.pi ** 2)) * electric_field ** 4)


def effective_fine_structure_running(
    energy_scale: float, alpha_0: float = ALPHA_FINE,
    electron_mass_eV: float = 5.11e5,  # 511 keV
) -> float:
    """One-loop QED beta: alpha(E) = alpha_0 / (1 - (alpha_0 / 3 pi) ln(E/m))."""
    if energy_scale <= electron_mass_eV:
        return alpha_0
    log_ratio = math.log(energy_scale / electron_mass_eV)
    denom = 1.0 - (alpha_0 / (3 * math.pi)) * log_ratio
    return float(alpha_0 / max(denom, 1e-12))


def pair_production_threshold(electric_field: float,
                                 mass_eV: float = 5.11e5) -> dict:
    """Schwinger threshold: pairs produced when E >= E_crit = m^2 c^3 / (e hbar)."""
    E_crit = mass_eV ** 2  # natural units, simplified
    ratio = electric_field / E_crit
    if ratio >= 1.0:
        regime = "above_threshold"
        rate_proxy = (electric_field / E_crit) ** 2
    else:
        regime = "below_threshold"
        rate_proxy = math.exp(-math.pi / max(ratio, 1e-30))
    return {
        "E_field": electric_field,
        "E_critical": E_crit,
        "ratio": float(ratio),
        "regime": regime,
        "rate_proxy": float(rate_proxy),
    }


def lp_modified_E_critical(vs: VanStockumInterior, r: float) -> float:
    """Frame-drag-modified critical field: E_crit -> E_crit / sqrt(F(r))."""
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    if abs(F) < 1e-12:
        return float("inf")
    return float(E_CRITICAL_SCHWINGER / math.sqrt(abs(F)))


def vacuum_polarization_at_r(
    vs: VanStockumInterior, r: float, q_squared: float = 1.0,
) -> dict:
    """Pi(q^2) at radius r on LP."""
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    if abs(F) < 1e-12:
        return {"pi": float("inf"), "regime": "near_horizon"}
    # Heuristic: pi(q^2) ~ (alpha/3 pi) ln(|q^2|) / sqrt(F)
    pi = (ALPHA_FINE / (3 * math.pi)) * math.log(max(abs(q_squared), 1e-30)) / math.sqrt(abs(F))
    return {
        "r": r,
        "F": F,
        "q_squared": q_squared,
        "pi": float(pi),
        "regime": "regular",
    }


def novelty_scan(n_radii: int = 30) -> dict:
    """Catcher on vacuum-pol profile."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05 * vs.R, 10 * vs.R, n_radii)
    def fn(r: float) -> np.ndarray:
        res = vacuum_polarization_at_r(vs, r)
        e_crit = lp_modified_E_critical(vs, r)
        return np.array([
            res["pi"] if math.isfinite(res["pi"]) else 1e6,
            min(e_crit, 1e6) if math.isfinite(e_crit) else 1e6,
        ])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
