"""Lewis-Papapetrou vacuum exterior integrator for cylindrical WLP metric.

Master ODE (Ernst formulation, cylindrical, z-independent vacuum)
----------------------------------------------------------------
For metric

    ds^2 = -F dt^2 + 2 K dt dphi + L dphi^2 + h (dr^2 + dz^2)

with FL + K^2 = r^2 (canonical Weyl coordinate) and all functions of r
only, the vacuum Einstein equations reduce, via the Ernst potential
E = F + i psi (psi the twist, related to omega = K/F by
omega' = c r / F^2 with c = const), to a single second-order ODE for F:

    F * (F'' + F'/r) = (F')^2 - c^2.

Auxiliary quadratures:

    omega(r)  = omega(R) + integral_{R}^{r} (c rho / F(rho)^2) drho
    K(r)      = F(r) * omega(r)
    L(r)      = (r^2 - K(r)^2) / F(r)
    h(r)      = h(R) * exp[ integral_{R}^{r} (rho/2) ((F'/F)^2 + (psi'/F)^2) drho ]

Matching to van Stockum dust interior at r = R (interior: F=1, K=omega_dust*r^2)
gives:
    F(R)         = 1
    F'(R)        = 0
    c            = 2 * omega_dust              (twist conserved across r=R)
    omega(R)     = omega_dust * R^2            (metric angular velocity)
    h(R)         = exp(-omega_dust^2 * R^2)    (interior conformal factor)

Reference
---------
Lewis 1932 (cylindrical vacuum); Bonnor 1980 J. Phys. A 13, 2121 (case
classification); Tipler 1974 (CTC threshold a = omega R > 1/2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class LPSolution:
    """Numerical solution of the Lewis-Papapetrou master ODE in r > R.

    Stored as dense interpolants over [R, r_max].

    Attributes
    ----------
    omega_dust : float
        Interior dust angular velocity.
    R : float
        Cylinder radius.
    r : np.ndarray
        Radial samples (post-integration grid).
    F : np.ndarray
    K : np.ndarray
    L : np.ndarray
    h : np.ndarray
        Metric components at the samples.
    omega_metric : np.ndarray
        K/F at each sample.
    c : float
        Twist constant 2 * omega_dust.
    F_zeros : np.ndarray
        Radii r > R where F crosses zero (ergo-surfaces; *removable*
        singularities of the ODE for Case III). Empty if none in [R, r_max].
    """

    omega_dust: float
    R: float
    r: np.ndarray
    F: np.ndarray
    K: np.ndarray
    L: np.ndarray
    h: np.ndarray
    omega_metric: np.ndarray
    c: float
    F_zeros: np.ndarray

    def gphiphi(self, r: float | np.ndarray) -> np.ndarray:
        """Interpolate L = g_{phi phi} at given r."""
        return np.interp(np.asarray(r, dtype=float), self.r, self.L)

    def gtt(self, r: float | np.ndarray) -> np.ndarray:
        """Interpolate g_{tt} = -F at given r."""
        return -np.interp(np.asarray(r, dtype=float), self.r, self.F)

    def gtphi(self, r: float | np.ndarray) -> np.ndarray:
        """Interpolate g_{t phi} = K at given r."""
        return np.interp(np.asarray(r, dtype=float), self.r, self.K)


def _ernst_rhs(r: float, y: np.ndarray, c: float) -> np.ndarray:
    """Right-hand side of the Ernst ODE rewritten as a first-order system.

    State y = [F, F'].
    Equation: F * (F'' + F'/r) = (F')^2 - c^2
            => F'' = ((F')^2 - c^2) / F  -  F'/r.
    """
    F, dF = y
    if F == 0.0:
        # coordinate singularity; let the integrator catch via its event handler
        return np.array([dF, 0.0])
    d2F = ((dF * dF - c * c) / F) - dF / r
    return np.array([dF, d2F])


def _make_F_zero_event(c: float):
    """Construct a non-terminal event recording F = 0 zero crossings.

    F = 0 is a *removable* singularity of the Ernst ODE in Case III: the
    solution passes through with F' = +/- c; the metric becomes degenerate
    only as a coordinate effect ("ergosurface"). We record crossings for
    diagnostic use but do not terminate.
    """
    def event(r: float, y: np.ndarray) -> float:
        return y[0]
    event.terminal = False  # type: ignore[attr-defined]
    event.direction = 0.0  # type: ignore[attr-defined]
    return event


def integrate_lp_exterior(
    omega_dust: float,
    R: float,
    r_max: float,
    n_samples: int = 4001,
    rtol: float = 1e-10,
    atol: float = 1e-13,
) -> LPSolution:
    """Integrate the vacuum Lewis-Papapetrou exterior outward from r = R.

    Parameters
    ----------
    omega_dust : float
        Interior van Stockum dust angular velocity.
    R : float
        Cylinder radius. Integration domain is (R, r_max].
    r_max : float
        Outer integration bound.
    n_samples : int
        Output grid density (linspace from R to r_max).
    rtol, atol : float
        ODE tolerances (Radau).

    Returns
    -------
    LPSolution
    """
    if omega_dust < 0:
        raise ValueError("omega_dust must be non-negative")
    if R <= 0:
        raise ValueError("R must be positive")
    if r_max <= R:
        raise ValueError("r_max must exceed R")

    c = 2.0 * omega_dust
    y0 = np.array([1.0, 0.0])  # F(R) = 1, F'(R) = 0

    r_eval = np.linspace(R, r_max, n_samples)

    sol = solve_ivp(
        fun=lambda r, y: _ernst_rhs(r, y, c),
        t_span=(R, r_max),
        y0=y0,
        t_eval=r_eval,
        method="Radau",
        rtol=rtol,
        atol=atol,
        events=_make_F_zero_event(c),
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"LP exterior integration failed: {sol.message}")

    r_arr = sol.t
    F_arr = sol.y[0]
    Fp_arr = sol.y[1]
    F_zeros_arr = np.asarray(sol.t_events[0], dtype=float)

    # omega_metric by quadrature: omega'(r) = c r / F^2
    # Use cumulative trapezoidal integration on the same grid.
    omega_metric_init = omega_dust * R * R  # interior matching value at r=R
    integrand = c * r_arr / (F_arr * F_arr)
    omega_metric = omega_metric_init + np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r_arr))]
    )

    K_arr = F_arr * omega_metric
    L_arr = (r_arr * r_arr - K_arr * K_arr) / F_arr

    # h by quadrature (standard cylindrical-Weyl reduction):
    #   d ln h / dr = (r / 2) * [ (F'/F)^2 + (c/F)^2 ]
    # Diverges at F = 0; we mask out grid points where |F| is below a small
    # threshold and freeze ln h there. The h component is unphysical at F = 0
    # (coordinate ergosurface) so this restriction is appropriate.
    h_init = float(np.exp(-omega_dust * omega_dust * R * R))
    F_safe = np.where(np.abs(F_arr) > 1e-3, F_arr, np.sign(F_arr + 1e-300) * 1e-3)
    dlnh_dr = 0.5 * r_arr * ((Fp_arr / F_safe) ** 2 + (c / F_safe) ** 2)
    # Cap to prevent runaway across F-zero crossings
    dlnh_dr = np.clip(dlnh_dr, 0.0, 1e6)
    ln_h = np.log(h_init) + np.concatenate(
        [[0.0], np.cumsum(0.5 * (dlnh_dr[1:] + dlnh_dr[:-1]) * np.diff(r_arr))]
    )
    h_arr = np.exp(np.clip(ln_h, -700.0, 700.0))

    return LPSolution(
        omega_dust=omega_dust,
        R=R,
        r=r_arr,
        F=F_arr,
        K=K_arr,
        L=L_arr,
        h=h_arr,
        omega_metric=omega_metric,
        c=c,
        F_zeros=F_zeros_arr,
    )
