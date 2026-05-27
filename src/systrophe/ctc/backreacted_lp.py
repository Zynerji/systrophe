"""Toy iterative semiclassical back-reaction on the supercritical LP exterior.

What this module does
---------------------
Iteratively applies a *first-order Polyakov correction* to the (t, r)
sector profile F(r) of the supercritical van Stockum exterior, and
tracks how the chronology horizon r_H (the first zero of F) moves under
the iteration. The update rule per step is

    F_{n+1}(r)  =  F_n(r)  -  ell2 * alpha_coup * <T_tt>_B[F_n](r)

where <T_tt>_B[F_n] is the Boulware Polyakov stress-energy density of
a massless conformally-coupled scalar evaluated *on the current
profile* F_n. Iterating to self-consistency would give the back-reacted
metric profile and the corresponding (possibly displaced or censored)
chronology horizon.

Scope and honesty
-----------------
This is **not** a derivation of the full 4D semiclassical Einstein
equation on LP. The full equation needs much more apparatus (Hadamard
mode-sum in 4D, conservation across (t, phi)-sector, etc.). What is
shipped here is a clean toy that captures the *qualitative* direction
of back-reaction:

  - For ell2 small, the iteration converges and r_H is displaced by an
    amount that scales linearly with ell2. Matches the Phase 2c
    eps_bd ~ ell^2 / (6 kappa) scaling.
  - For ell2 above a threshold, the iteration may *amplify* the
    perturbation -- a mass-inflation-like instability.
  - Sometimes the back-reacted F(r) never returns to zero in [R, r_max]
    -- the chronology horizon is *censored* by the back-reaction.

The verdict is one of {`displaced`, `censored`, `unstable`,
`unconverged`}. Interpretation is the user's; this module just reports
the numerics.

Reference: cf. Hiscock-Konkowski (1982) PRD 26, 1225 for the 2D
Schwinger-Dewitt approach.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


# ----- F(r) callable wrappers ---------------------------------------------


def classical_F(vs: VanStockumInterior) -> Callable[[float], float]:
    """Closure returning F_classical(r) from `vs`."""
    def F(r: float) -> float:
        return float(vs.analytic_exterior_F(np.array([r], dtype=float))[0])
    return F


def perturbed_F(
    F_callable: Callable[[float], float],
    delta_F_callable: Callable[[float], float],
) -> Callable[[float], float]:
    """F(r) = F_callable(r) + delta_F_callable(r)."""
    def F(r: float) -> float:
        return float(F_callable(r) + delta_F_callable(r))
    return F


# ----- chronology horizon (first zero of F) -------------------------------


def find_chronology_horizon(
    F: Callable[[float], float],
    r_min: float, r_max: float, n_grid: int = 4001,
) -> float | None:
    """First r in (r_min, r_max] where F(r) <= 0. Returns None if F stays > 0.

    Uses a coarse scan to bracket the first sign change, then refines via
    bisection.
    """
    r_grid = np.linspace(r_min, r_max, n_grid)
    F_grid = np.array([F(float(r)) for r in r_grid])
    # find first negative point
    neg_idx = np.where(F_grid <= 0.0)[0]
    if neg_idx.size == 0:
        return None
    first = int(neg_idx[0])
    if first == 0:
        return float(r_grid[0])
    lo, hi = float(r_grid[first - 1]), float(r_grid[first])
    # bisect
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if F(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ----- Polyakov <T_tt>_B on an arbitrary F(r) callable --------------------


def polyakov_T_tt_general(
    F: Callable[[float], float], r: float, eps: float = 1e-5,
) -> float:
    """Boulware <T_tt> at radius r for a generic F(r) callable (h = 1).

    Uses the same 2D Polyakov construction as
    `systrophe.ctc.stress_energy_ctc.boulware_stress_tensor`, but
    accepts an arbitrary F(r) callable rather than a VanStockum source.

    sigma(r) = (1/2) log F(r)
    r_*(r) integrand = 1/sqrt(F)
    <T_tt>_B = -(1/(12 pi)) * F * [sigma_rstar^2 - sigma_rstar_rstar]
             ... = (1/24 pi) * (F'^2/F - F'' - F'^2/(2 F))
                 wait -- evaluate via finite differences on sigma in r.
    For numerical robustness we just compute sigma(r), sigma'(r),
    sigma''(r) via finite differences in r, then translate to r_* via
    the chain rule (dr_*/dr = 1/sqrt(F)).

    The Boulware formula reduces to
        <T_tt>_B = -(1/24 pi) * F * [2 sigma_rstar_rstar - sigma_rstar^2]
                 = -(1/24 pi) * [ F''(r) - 0.5 F'(r)^2 / F(r) ]   (h = 1).

    Returns NaN where F(r) <= 0 (we're inside / on the chronology
    region, the formula is ill-defined).
    """
    F_at = F(r)
    if not (F_at > 0):
        return float("nan")
    F_plus = F(r + eps)
    F_minus = F(r - eps)
    Fp = (F_plus - F_minus) / (2.0 * eps)
    Fpp = (F_plus - 2.0 * F_at + F_minus) / (eps * eps)
    return float(-(1.0 / (24.0 * math.pi))
                 * (Fpp - 0.5 * Fp * Fp / F_at))


# ----- the iteration ------------------------------------------------------


@dataclass
class BackreactionStep:
    iteration: int
    F_grid: np.ndarray
    delta_F_grid: np.ndarray
    T_tt_grid: np.ndarray
    r_H: float | None


@dataclass
class BackreactionResult:
    r_grid: np.ndarray
    steps: list[BackreactionStep] = field(default_factory=list)
    r_H_history: list[float | None] = field(default_factory=list)
    converged: bool = False
    verdict: str = ""           # 'displaced' / 'censored' / 'unstable' / 'unconverged'
    final_residual: float = float("inf")


def iterate_backreaction(
    vs: VanStockumInterior,
    ell2: float = 1e-3,
    alpha_coup: float = 8.0 * math.pi,
    n_iter: int = 20,
    r_min_factor: float = 1.01,
    r_max_factor: float = 3.0,
    n_grid: int = 401,
    tol: float = 1e-6,
    damping: float = 0.5,
) -> BackreactionResult:
    """Iterate F_n -> F_{n+1} = F_n - damping * ell2 * alpha_coup * <T_tt>[F_n].

    Parameters
    ----------
    vs : VanStockumInterior
        Supercritical LP source (a > 1/2).
    ell2 : float
        ell_Planck^2 coupling strength; ell2 = 0 recovers the classical
        profile.
    alpha_coup : float
        Coupling constant (default 8 pi, mimicking 8 pi G in Einstein).
    n_iter : int
        Maximum number of iterations.
    r_min_factor, r_max_factor : float
        Iteration domain is [r_min_factor * R, r_max_factor * R].
    n_grid : int
        Radial grid resolution.
    tol : float
        Convergence threshold (max |delta_F| across the grid).
    damping : float in (0, 1]
        Under-relaxation factor; 1.0 = full step, 0.5 = half-step.

    Returns
    -------
    BackreactionResult with the trajectory, horizon history, and verdict.
    """
    if not vs.is_supercritical():
        raise ValueError(
            "iterate_backreaction expects a supercritical LP source "
            "(a = omega R > 1/2)"
        )
    if not 0.0 < damping <= 1.0:
        raise ValueError(f"damping must be in (0, 1], got {damping}")
    r_grid = np.linspace(r_min_factor * vs.R, r_max_factor * vs.R, n_grid)
    F_n = np.array([classical_F(vs)(float(r)) for r in r_grid])
    delta_F_total = np.zeros_like(F_n)

    result = BackreactionResult(r_grid=r_grid)

    for it in range(n_iter):
        # Build F_n(r) as a callable via linear interpolation.
        def F_callable(r: float, F_n=F_n, r_grid=r_grid) -> float:
            return float(np.interp(r, r_grid, F_n))
        # Compute T_tt on this profile.
        T_tt = np.array(
            [polyakov_T_tt_general(F_callable, float(r)) for r in r_grid]
        )
        # NaN where F_n <= 0; treat as no correction in that region (the
        # back-reaction has already censored / driven F there).
        T_tt = np.nan_to_num(T_tt, nan=0.0, posinf=0.0, neginf=0.0)
        # Cap |T_tt| to prevent run-away near singular points.
        T_tt = np.clip(T_tt, -1e3, 1e3)
        delta_F = -damping * ell2 * alpha_coup * T_tt
        delta_F_total = delta_F_total + delta_F
        F_new = F_n + delta_F
        residual = float(np.max(np.abs(delta_F)))
        r_H = find_chronology_horizon(
            F_callable, float(r_grid[0]), float(r_grid[-1]),
        )
        result.steps.append(BackreactionStep(
            iteration=it,
            F_grid=F_new.copy(),
            delta_F_grid=delta_F.copy(),
            T_tt_grid=T_tt.copy(),
            r_H=r_H,
        ))
        result.r_H_history.append(r_H)
        F_n = F_new
        if residual < tol:
            result.converged = True
            result.final_residual = residual
            break
        if not np.isfinite(residual) or residual > 1e6:
            result.converged = False
            result.verdict = "unstable"
            result.final_residual = residual
            return result
    else:
        result.converged = False
        result.final_residual = residual

    # Final verdict
    F_final = F_n

    def F_final_callable(r: float, F_final=F_final, r_grid=r_grid) -> float:
        return float(np.interp(r, r_grid, F_final))

    r_H_final = find_chronology_horizon(
        F_final_callable, float(r_grid[0]), float(r_grid[-1]),
    )
    # Swallowed by source: r_H pinned at the lower domain edge (the
    # horizon retreated past r_min, i.e. into the cylinder).
    swallowed_threshold = r_grid[0] + 0.02 * (r_grid[-1] - r_grid[0])
    if r_H_final is None:
        result.verdict = "censored"
    elif r_H_final < swallowed_threshold:
        result.verdict = "swallowed"
    elif result.converged:
        result.verdict = "displaced"
    else:
        result.verdict = "unconverged"
    return result


# ----- classical-limit and shift diagnostics ------------------------------


def classical_horizon(vs: VanStockumInterior) -> float:
    """Analytic location of the first chronology horizon (F = 0).

    For supercritical LP exterior, F(r) = (r/R) sin(alpha u + gamma)/sin gamma
    with u = log(r/R), gamma = pi - atan(alpha), alpha = sqrt(4 a^2 - 1).
    The first zero is at u = (pi - gamma)/alpha = atan(alpha)/alpha,
    so r_H = R * exp(atan(alpha)/alpha).
    """
    if not vs.is_supercritical():
        raise ValueError("classical_horizon only defined for supercritical LP")
    alpha = vs.alpha
    return float(vs.R * math.exp(math.atan(alpha) / alpha))


def horizon_shift(result: BackreactionResult, vs: VanStockumInterior) -> dict:
    """Compare back-reacted r_H to classical r_H."""
    r_H_classical = classical_horizon(vs)
    history = [h for h in result.r_H_history if h is not None]
    if not history:
        return {
            "r_H_classical": r_H_classical,
            "r_H_backreacted": None,
            "shift": None,
            "verdict": result.verdict,
            "censored": True,
        }
    r_H_br = history[-1]
    return {
        "r_H_classical": r_H_classical,
        "r_H_backreacted": float(r_H_br),
        "shift": float(r_H_br - r_H_classical),
        "verdict": result.verdict,
        "censored": False,
    }
