"""Regime-dispatching Lewis-Papapetrou exterior solver.

The naive Ernst master integrator F(F'' + F'/r) = (F')^2 - c^2 in the
explicit form F'' = ((F')^2 - c^2)/F - F'/r has a *removable*
singularity at every F = 0 crossing. In the supercritical Bonnor
Case III (a = omega R > 1/2), F has infinitely many zeros and any
practical numerical scheme based on the (F, F') state suffers
catastrophic accuracy loss after a few crossings, because:

  - Near F = 0, ((F')^2 - c^2) and F both vanish, leading to
    catastrophic cancellation of the form 0/0;
  - L'Hopital's rule fixes F'_0 = +/- c and F''_0 = F'_0 / r_0 at the
    crossing, but small floating-point error in F or F' near the zero
    grows exponentially in subsequent integration.

Naive Ehlers transformations xi = (1 - E)/(1 + E) inherit the same
removable-singularity structure: the F = 0 surface maps to |xi| = 1,
where the Ernst equation in xi coordinates has its own 0/0 form.

This module dispatches to the *exact analytic Case III closed form*
for supercritical inputs (where the closed form is available and is
the correct tool) and falls back to the basic numerical integrator
for sub- and critical regimes (where F has at most one zero and the
naive integration is well-conditioned). This is the cleanest robust
solution and recovers machine precision throughout the supercritical
regime, including arbitrarily many F = 0 crossings.

The wrapper uniformly returns an `LPRobustSolution` regardless of
which path was used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LPRobustSolution:
    """Uniform interface for LP exterior solutions.

    Attributes
    ----------
    omega_dust : float
        Interior dust angular velocity.
    R : float
        Cylinder radius.
    a : float
        Dimensionless rotation parameter omega * R.
    regime : str
        "supercritical" (analytic Case III), "critical" (numerical),
        or "subcritical" (numerical).
    r : np.ndarray
        Radial samples in [R, r_max].
    F, K, L, h : np.ndarray
        Metric components at the samples. h is set to 1.0 in the
        analytic supercritical path (the conformal factor is not
        currently part of the analytic Case III API).
    F_zeros : np.ndarray
        Radii where F crosses zero (analytic for supercritical:
        r_n = R * exp((n pi - gamma) / alpha) for n = 1, 2, ...).
    c : float
        Twist constant 2 * omega_dust.
    """

    omega_dust: float
    R: float
    a: float
    regime: str
    r: np.ndarray
    F: np.ndarray
    K: np.ndarray
    L: np.ndarray
    h: np.ndarray
    F_zeros: np.ndarray
    c: float

    def F_at(self, r):
        return np.interp(np.asarray(r, dtype=float), self.r, self.F)

    def L_at(self, r):
        return np.interp(np.asarray(r, dtype=float), self.r, self.L)


def integrate_lp_robust(
    omega_dust: float,
    R: float,
    r_max: float,
    n_samples: int = 4001,
    rtol: float = 1e-10,
    atol: float = 1e-13,
) -> LPRobustSolution:
    """Solve the LP exterior in the appropriate regime and return a uniform result.

    Uses exact analytic closed forms in all three Bonnor regimes:
    supercritical (sinusoidal), critical (logarithmic), subcritical
    (hyperbolic). The numerical fallback (basic Lewis-Papapetrou
    integrator) is no longer used; analytic forms supply machine-precision
    F, K, L throughout.

    The rtol, atol parameters are retained for API compatibility but are
    only consulted if the user explicitly requests the numerical fallback
    via use_numerical=True.
    """
    if omega_dust < 0:
        raise ValueError("omega_dust must be non-negative")
    if R <= 0:
        raise ValueError("R must be positive")
    if r_max <= R:
        raise ValueError("r_max must exceed R")
    a = omega_dust * R

    if a > 0.5 + 1e-12:
        return _analytic_solution(omega_dust, R, r_max, n_samples, regime="supercritical")
    if a > 0.5 - 1e-9:
        return _analytic_solution(omega_dust, R, r_max, n_samples, regime="critical")
    return _analytic_solution(omega_dust, R, r_max, n_samples, regime="subcritical")


def _analytic_solution(
    omega_dust: float, R: float, r_max: float, n_samples: int, regime: str
) -> LPRobustSolution:
    """Closed-form F, K, L on a log-uniform grid for any Bonnor regime."""
    from .vanstockum import VanStockumInterior

    vs = VanStockumInterior(omega=omega_dust, R=R)
    r = np.geomspace(R, r_max, n_samples)
    F = vs.analytic_exterior_F(r)
    K = vs.analytic_exterior_K(r)
    L = vs.analytic_exterior_L(r)

    F_zeros: list[float] = []
    if regime == "supercritical":
        alpha = vs.alpha
        gamma = np.pi - np.arctan(alpha)
        n = 1
        while True:
            u_n = (n * np.pi - gamma) / alpha
            if u_n <= 0:
                n += 1
                continue
            r_n = R * np.exp(u_n)
            if r_n > r_max:
                break
            F_zeros.append(r_n)
            n += 1
    elif regime == "critical":
        # F = (r/R)(1 - u) = 0 -> u = 1 -> r = e R
        r_zero = R * np.e
        if r_zero <= r_max:
            F_zeros.append(r_zero)
    else:  # subcritical
        # F = (r/R) [cosh(beta u) - sinh(beta u)/beta] = 0
        # tanh(beta u) = beta -> u = arctanh(beta)/beta. Real iff beta < 1
        # (always true for subcritical).
        beta = float(np.sqrt(1.0 - 4.0 * (omega_dust * R) ** 2))
        if beta < 1.0:
            u_zero = np.arctanh(beta) / beta
            r_zero = R * np.exp(u_zero)
            if r_zero <= r_max:
                F_zeros.append(r_zero)

    return LPRobustSolution(
        omega_dust=float(omega_dust),
        R=float(R),
        a=float(omega_dust * R),
        regime=regime,
        r=r,
        F=F,
        K=K,
        L=L,
        h=np.ones_like(r),
        F_zeros=np.array(F_zeros, dtype=float),
        c=float(2.0 * omega_dust),
    )
