"""T-duality and S-duality on the LP rotating cylinder.

For a compactified extra dimension of radius L_xi (Phase 25), T-duality
inverts L_xi: L_xi -> alpha'/L_xi. This swaps Kaluza-Klein modes with
winding modes. For the LP cylinder, T-duality of the xi direction
relates a large-L_xi configuration (KK-mode dominated) to a small-L_xi
configuration (winding-mode dominated).

S-duality (electric-magnetic duality) on the rotating cylinder swaps
the angular-momentum/rotation roles. Conjecture: the LP a parameter
has an S-dual partner a_S = 1/(2a) or similar.

This module:
- t_dual_L_xi: alpha' / L_xi (with alpha' = string length squared)
- t_dual_kk_to_winding: spectrum exchange
- s_dual_a: candidate a-dual
- duality_invariant_observables: things that don't change under duality
- novelty_scan: catcher on the duality orbit of an observable
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior

ALPHA_PRIME = 1.0  # string length squared, natural units


def t_dual_L_xi(L_xi: float, alpha_prime: float = ALPHA_PRIME) -> float:
    """L_xi -> alpha' / L_xi."""
    if L_xi <= 0:
        raise ValueError("L_xi must be positive")
    return float(alpha_prime / L_xi)


def t_dual_kk_to_winding(n: int, L_xi: float) -> dict:
    """KK mode n (mass n/L_xi) <-> winding mode w (mass w L_xi/alpha')."""
    L_dual = t_dual_L_xi(L_xi)
    kk_mass = n / L_xi
    winding_mass = n * L_dual / ALPHA_PRIME  # equivalent in dual frame
    return {
        "kk_mode_n": int(n),
        "original_L_xi": L_xi,
        "dual_L_xi": L_dual,
        "kk_mass_original": float(kk_mass),
        "winding_mass_dual": float(winding_mass),
        "masses_match": bool(abs(kk_mass - winding_mass) < 1e-9),
    }


def s_dual_a(a: float) -> float:
    """Candidate S-dual: a -> 1/(2a). Maps a=1/2 (critical) to itself."""
    if a <= 0:
        return float("inf")
    return float(1.0 / (2 * a))


def duality_invariant_observables(vs: VanStockumInterior) -> dict:
    """Observables conjectured invariant under combined T*S duality."""
    a = vs.a
    a_S = s_dual_a(a)
    # alpha = sqrt(4a^2 - 1) for supercritical; for a_S = 1/(2a), alpha_S = sqrt(1/a^2 - 1) for a > 1
    if a > 0.5:
        alpha = math.sqrt(4 * a ** 2 - 1)
    else:
        alpha = float("nan")
    if a_S > 0.5:
        alpha_S = math.sqrt(4 * a_S ** 2 - 1)
    else:
        alpha_S = float("nan")
    return {
        "a": float(a),
        "a_S": float(a_S),
        "alpha": float(alpha) if math.isfinite(alpha) else None,
        "alpha_S": float(alpha_S) if math.isfinite(alpha_S) else None,
        "fixed_point_at_a_eq_half": bool(abs(a - 0.5) < 1e-9),
    }


def t_dual_orbit(L_xi: float, n_iter: int = 3,
                  alpha_prime: float = ALPHA_PRIME) -> list[float]:
    """T-duality is involutive: applying twice returns to L_xi."""
    orbit = [L_xi]
    cur = L_xi
    for _ in range(n_iter):
        cur = t_dual_L_xi(cur, alpha_prime)
        orbit.append(cur)
    return orbit


def novelty_scan(n_a_values: int = 30) -> dict:
    """Catcher on the (a, alpha) curve under S-duality."""
    a_grid = np.linspace(0.6, 5.0, n_a_values)
    def fn(a_float: float) -> np.ndarray:
        a = float(a_float)
        a_S = s_dual_a(a)
        if a > 0.5:
            alpha = math.sqrt(4 * a ** 2 - 1)
        else:
            alpha = 0.0
        return np.array([a, a_S, alpha])
    result = scan_novelty(a_grid, fn, n_bits=32, parameter_label="a")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
