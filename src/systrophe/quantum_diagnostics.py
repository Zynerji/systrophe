"""Classical pre-quantum diagnostics related to chronology protection.

This module provides simple invariants that *would* appear in a full
semi-classical analysis of the Tipler / Systrophe spacetime, without
attempting the full QFT-in-curved-spacetime calculation.

The chronology-protection conjecture (Hawking 1992) predicts that
quantum back-reaction destabilises spacetimes containing CTCs by
diverging the renormalised stress-energy tensor at the Cauchy
horizon. The full calculation involves point-splitting renormalisation
of two-point functions for a free scalar field on the Tipler
background and is out of scope for this package.

What we provide
---------------
- `tolman_blueshift_factor(F)`: the Tolman blueshift of a locally static
  observer at $g_{tt} = -F$ relative to one with $F = 1$. Diverges as
  $1/\\sqrt{F}$ as $F \\to 0$ (ergosurface).
- `chronology_protection_indicator(F)`: a simple $1/F^2$ proxy that
  approximates the leading divergence of the renormalised stress-
  energy near the ergosurface (and hence near the inner CTC band edges
  in Bonnor Case III). This is a *qualitative* indicator only.
- `cauchy_horizon_estimate(vs)`: returns the radii of the Tipler
  exterior where the Tolman blueshift formally diverges (i.e. the
  $F$-zeros). These are candidate Cauchy horizons.

These quantities are entirely classical. A genuine chronology-protection
test requires the renormalised semi-classical stress-energy tensor;
this module exposes the *classical* invariants that gate that
calculation. References for the full programme: Hawking 1992;
Visser 1996, *Lorentzian Wormholes*, Ch. 14.
"""

from __future__ import annotations

import numpy as np


def tolman_blueshift_factor(F: float | np.ndarray) -> np.ndarray:
    """Tolman blueshift factor 1 / sqrt(F) for a locally static observer.

    Diverges as F -> 0 (ergosurface / Cauchy horizon candidate).
    Returns inf where F <= 0; the user may want to mask such points.
    """
    F_arr = np.asarray(F, dtype=float)
    out = np.where(F_arr > 0, 1.0 / np.sqrt(np.where(F_arr > 0, F_arr, 1.0)), np.inf)
    return out


def chronology_protection_indicator(F: float | np.ndarray) -> np.ndarray:
    """Heuristic 1 / F^2 indicator of the leading classical divergence.

    A rough proxy for the leading divergence of the renormalised
    stress-energy tensor near the ergosurface, motivated by the
    Tolman / Hawking blueshift scaling. Not a substitute for a real
    QFTCS calculation; use only as a qualitative chronology-protection
    flag.
    """
    F_arr = np.asarray(F, dtype=float)
    return np.where(np.abs(F_arr) > 1e-30, 1.0 / (F_arr * F_arr), np.inf)


def ricci_scalar(vs, r: float | np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Numerically compute the Ricci scalar R(r) on the Tipler exterior.

    Uses central finite differences of the analytic Case III metric components.
    The Ricci scalar of a Lewis-Papapetrou metric depends on F, K, L, h and
    their first and second r-derivatives. We use the formula for vacuum
    cylindrical metrics (which gives R = 0 to leading order; departures
    from zero are numerical artefacts and a sanity check on the integrator).

    More precisely: for a vacuum solution R_{mu nu} = 0 implies R = 0
    everywhere. This function is exposed for diagnostic / debugging use,
    not as a "physics" Ricci scalar (which is identically zero for the
    vacuum Tipler exterior).

    Returns the numerical R(r) values at the requested radii. For an
    exact vacuum solution, this should be zero up to finite-difference error.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    r_arr = np.asarray(r, dtype=float)

    def F(rr): return vs.analytic_exterior_F(rr)
    def K(rr): return vs.analytic_exterior_K(rr)
    def L(rr): return vs.analytic_exterior_L(rr)

    # First and second derivatives
    Fp = (F(r_arr + eps) - F(r_arr - eps)) / (2 * eps)
    Fpp = (F(r_arr + eps) - 2 * F(r_arr) + F(r_arr - eps)) / (eps * eps)
    Kp = (K(r_arr + eps) - K(r_arr - eps)) / (2 * eps)
    Lp = (L(r_arr + eps) - L(r_arr - eps)) / (2 * eps)

    # For the Ernst master equation in vacuum,
    # F(F'' + F'/r) = (F')^2 - c^2 holds identically.
    # The Ricci scalar of the LP vacuum solution is zero; this routine
    # returns a residual measuring numerical violation:
    c = 2.0 * vs.omega
    residual = F(r_arr) * (Fpp + Fp / r_arr) - (Fp ** 2 - c ** 2)
    return residual


def conformal_anomaly_2d_proxy(vs, r: float | np.ndarray) -> np.ndarray:
    """Proxy for the trace anomaly of a conformal scalar in 2D effective theory.

    In 2D, a conformally coupled massless scalar has trace anomaly
        <T^mu_mu> = R / (24 pi),
    where R is the 2D Ricci scalar. For the radial slice of a stationary
    cylindrical spacetime, an effective 2D Ricci scalar can be obtained
    by dimensional reduction; here we use a heuristic proxy
        R_2d_proxy = -F''(r) / F(r) - F'(r) / (r F(r))
    which approximates the leading curvature contribution in the
    radial-temporal slice. Diverges at F = 0 (Cauchy horizon), giving
    qualitative support for chronology protection via divergent
    conformal anomaly at the chronology horizon.

    Returns R_2d_proxy / (24 pi).

    This is a CLASSICAL proxy. A genuine conformal-anomaly calculation
    requires fixing the 2D reduction, choosing a vacuum state, and
    point-splitting renormalisation -- outside the scope of this
    package.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    r_arr = np.asarray(r, dtype=float)
    eps = 1e-5

    def F(rr): return vs.analytic_exterior_F(rr)

    Fp = (F(r_arr + eps) - F(r_arr - eps)) / (2 * eps)
    Fpp = (F(r_arr + eps) - 2 * F(r_arr) + F(r_arr - eps)) / (eps * eps)
    F_safe = np.where(np.abs(F(r_arr)) > 1e-9, F(r_arr), np.sign(F(r_arr)) * 1e-9)
    R_2d = -Fpp / F_safe - Fp / (r_arr * F_safe)
    return R_2d / (24.0 * np.pi)


def surface_gravity_at_horizon(vs, r_horizon: float, eps: float = 1e-6) -> float:
    """Surface gravity kappa at a Cauchy-horizon candidate r_horizon.

    For a stationary spacetime with Killing horizon at F = 0, the surface
    gravity is
        kappa = (1/2) F'(r_horizon)
    in coordinates where g_{tt} = -F. The associated Hawking temperature
    is T_H = kappa / (2 pi). For the Tipler chronology horizons (F-zeros
    in supercritical), this quantifies the rate at which thermal back-
    reaction would diverge in a full QFTCS calculation.

    Returns the numerical kappa value.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    F_plus = float(vs.analytic_exterior_F(r_horizon + eps))
    F_minus = float(vs.analytic_exterior_F(r_horizon - eps))
    Fp = (F_plus - F_minus) / (2.0 * eps)
    return 0.5 * abs(Fp)


def hawking_temperature_at_horizon(vs, r_horizon: float, eps: float = 1e-6) -> float:
    """Hawking temperature T_H = kappa / (2 pi) at a Cauchy horizon candidate."""
    return surface_gravity_at_horizon(vs, r_horizon, eps) / (2.0 * np.pi)


def cauchy_horizon_estimate(vs) -> np.ndarray:
    """Radii of the Tipler exterior where the Tolman blueshift diverges (F = 0).

    For a supercritical van Stockum cylinder these are the closed-form
    F-zeros r_n = R * exp((n pi - gamma) / alpha) for n = 1, 2, ....
    The Cauchy horizon of the maximally-extended chronology-violating
    region is the *first* such surface (r_1).

    Returns the array of F-zero radii up to a default upper bound of 10 R.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    if not vs.is_supercritical():
        raise ValueError("Cauchy horizon estimate only defined in Case III (a > 1/2)")

    alpha = vs.alpha
    gamma = float(np.pi - np.arctan(alpha))
    zeros = []
    n = 1
    upper_u = float(np.log(10.0))  # default r_max = 10 R
    while True:
        u_n = (n * np.pi - gamma) / alpha
        if u_n > upper_u:
            break
        if u_n > 0:
            zeros.append(float(vs.R * np.exp(u_n)))
        n += 1
    return np.array(zeros, dtype=float)
