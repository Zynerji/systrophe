"""Geodesics in stationary axisymmetric cylindrical spacetimes.

For a metric of the form

    ds^2 = -F(r) dt^2 + 2 K(r) dt dphi + L(r) dphi^2 + h(r) (dr^2 + dz^2)

with two Killing vectors d/dt and d/dphi, the conserved quantities are

    E   = -p_t = F t' - K phi'           (energy per unit mass)
    ell =  p_phi = K t' + L phi'         (angular momentum per unit mass)

Solving for (t', phi') with FL + K^2 = r^2:

    t'   =  (E L + K ell) / r^2
    phi' = -(E K - F ell) / r^2

The radial equation comes from the four-velocity normalization
g_{mu nu} u^mu u^nu = -1 (timelike) or 0 (null):

    h (r')^2 + h (z')^2 = -kappa - V_eff(r)

where kappa = +1 for timelike, 0 for null, and

    V_eff(r) = (F ell^2 - 2 K E ell - L E^2) / r^2.

For axisymmetric motion (z' = 0) the radial integration gives r(tau).

Circular orbits at fixed r (the case relevant to CTCs)
-----------------------------------------------------
At fixed r and z, a phi-orbit with angular velocity Omega = dphi/dt has
line element

    ds^2 = (-F + 2 K Omega + L Omega^2) dt^2.

The orbit is timelike iff F - 2 K Omega - L Omega^2 > 0. The boundary

    Omega_+- = -(K +- r) / L

splits the Omega axis into timelike and spacelike sectors. In the CTC
region (L < 0), the closed phi-orbit at fixed t (Omega = infinity) is
timelike with finite proper time per revolution sqrt(-L) * 2 pi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class CircularOrbit:
    """A constant-r, constant-z phi-orbit in the cylindrical spacetime.

    Attributes
    ----------
    r : float
        Radial coordinate.
    F, K, L : float
        Metric components at r.
    Omega : float
        Coordinate angular velocity dphi/dt. Caller's choice; must lie in
        the timelike sector to be a physical timelike orbit.
    """

    r: float
    F: float
    K: float
    L: float
    Omega: float

    @property
    def is_timelike(self) -> bool:
        """True iff F - 2 K Omega - L Omega^2 > 0."""
        return (self.F - 2.0 * self.K * self.Omega - self.L * self.Omega ** 2) > 0.0

    @property
    def is_in_ctc_region(self) -> bool:
        """True iff L < 0 at the orbit's radius (CTC region marker)."""
        return self.L < 0.0

    @property
    def coord_dt_per_revolution(self) -> float:
        """Coordinate-time advance over one full revolution (Delta phi = 2 pi)."""
        return 2.0 * np.pi / self.Omega

    @property
    def proper_dtau_per_revolution(self) -> float:
        """Proper-time advance over one full revolution.

        Equals sqrt(F - 2 K Omega - L Omega^2) * |Delta t|. Negative under
        the square root indicates a spacelike orbit; we return NaN there.
        """
        s = self.F - 2.0 * self.K * self.Omega - self.L * self.Omega ** 2
        if s <= 0.0:
            return float("nan")
        return float(np.sqrt(s) * abs(self.coord_dt_per_revolution))

    @property
    def coord_time_advance_per_proper_revolution(self) -> float:
        """Net coordinate time gained by the particle for one proper-time loop.

        Negative values mean the particle has moved BACK in coordinate time
        relative to a static observer at the same r, while advancing in its
        own proper time. This is the time-travel diagnostic.
        """
        return self.coord_dt_per_revolution


def timelike_omega_bounds(F: float, K: float, L: float, r: float) -> tuple[float, float]:
    """Compute the two roots Omega_+- of F - 2 K Omega - L Omega^2 = 0.

    Returned as (Omega_lower, Omega_upper).
    """
    if abs(L) < 1e-300:
        # Linear: F - 2 K Omega = 0 -> Omega = F / (2K). Single root.
        if abs(K) < 1e-300:
            raise ValueError("degenerate metric (K = L = 0)")
        return (F / (2.0 * K), F / (2.0 * K))
    op = -(K + r) / L
    om = -(K - r) / L
    return (min(op, om), max(op, om))


def is_omega_timelike(F: float, K: float, L: float, r: float, Omega: float) -> bool:
    """Whether the phi-orbit with given Omega is timelike at (r, ...)."""
    return (F - 2.0 * K * Omega - L * Omega ** 2) > 0.0


def omega_for_zero_coord_time(F: float, K: float, L: float, r: float) -> float:
    """Limit of Omega for which Delta t -> 0 per revolution.

    Strictly Omega = +infinity. Returns a large finite proxy (1e6 / r)
    for numerical use. This corresponds to the closed-phi-loop CTC at
    fixed t.
    """
    return 1.0e6 / max(r, 1.0e-12)


def omega_for_target_coord_time(target_dt: float) -> float:
    """Omega producing a target coordinate-time advance per revolution."""
    if target_dt == 0.0:
        return float("inf")
    return 2.0 * np.pi / target_dt


def integrate_geodesic(
    F_fn: Callable[[float], float],
    K_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    h_fn: Callable[[float], float],
    E: float,
    ell: float,
    r0: float,
    rdot0: float,
    tau_max: float,
    n_samples: int = 2001,
    kappa: float = 1.0,
) -> dict:
    """Integrate a planar (z = const) geodesic.

    Parameters
    ----------
    F_fn, K_fn, L_fn, h_fn : callables
        Metric components at radius r.
    E, ell : float
        Conserved energy and angular momentum per unit mass.
    r0 : float
        Initial radius.
    rdot0 : float
        Initial dr/dtau.
    tau_max : float
        Final proper-time horizon.
    n_samples : int
        Output grid density.
    kappa : float
        +1 for timelike, 0 for null (default timelike).

    Returns
    -------
    dict with keys 'tau', 't', 'r', 'phi'.
    """
    from scipy.integrate import solve_ivp

    def rhs(tau: float, y: np.ndarray) -> np.ndarray:
        t, r, phi, rdot = y
        F = F_fn(r)
        K = K_fn(r)
        L = L_fn(r)
        h = h_fn(r)
        r2 = r * r
        tdot = (E * L + K * ell) / r2
        phidot = (F * ell - E * K) / r2
        # r'' from differentiating (r')^2 = -(kappa + V_eff)/h
        # Easier: numerical derivative of V_eff
        eps = 1e-6 * max(r, 1.0)
        V_plus = (F_fn(r + eps) * ell ** 2 - 2 * K_fn(r + eps) * E * ell - L_fn(r + eps) * E ** 2) / (r + eps) ** 2
        V_minus = (F_fn(r - eps) * ell ** 2 - 2 * K_fn(r - eps) * E * ell - L_fn(r - eps) * E ** 2) / (r - eps) ** 2
        h_plus = h_fn(r + eps)
        h_minus = h_fn(r - eps)
        # d/dr [-(kappa + V)/h] = -(V'/h - (kappa+V) h'/h^2) = (-(kappa+V)·(-h'/h²) - V'/h)
        dVdr = (V_plus - V_minus) / (2 * eps)
        dhdr = (h_plus - h_minus) / (2 * eps)
        # rdot^2 = -(kappa + V) / h. d/dtau (rdot^2) = 2 rdot rddot.
        # Differentiating with chain rule: 2 rdot rddot = -(dVdr/h - (kappa+V) dhdr/h^2) * rdot
        # so rddot = -(dVdr / h - (kappa+V) * dhdr / h^2) / 2
        kappa_plus_V = kappa + (F * ell ** 2 - 2 * K * E * ell - L * E ** 2) / r2
        rddot = -(dVdr / h - kappa_plus_V * dhdr / (h * h)) / 2.0
        return np.array([tdot, rdot, phidot, rddot])

    y0 = np.array([0.0, r0, 0.0, rdot0])
    tau_eval = np.linspace(0.0, tau_max, n_samples)
    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, tau_max),
        y0=y0,
        t_eval=tau_eval,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"geodesic integration failed: {sol.message}")
    return {
        "tau": sol.t,
        "t": sol.y[0],
        "r": sol.y[1],
        "phi": sol.y[2],
        "rdot": sol.y[3],
    }
