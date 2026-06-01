"""Geodesic dynamics under rigid vs. time-dependent Tipler rotation.

This is the *conservative* half of the story. It establishes, by direct
numerical integration, the two first-principles facts that frame the
Lorenz-attractor result:

H0  A single **rigidly**-rotating cylinder (omega = const) has an extra
    Killing vector d/dt, hence p_t = -E is conserved. Together with
    p_phi = ell and the Hamiltonian H this gives three integrals for a
    3-DOF system -> **integrable**, regular orbits, ~zero Lyapunov.

H2  Making the rotation **time-dependent** (omega = omega(t), a "wobbling"
    column) breaks the d/dt symmetry: p_t is no longer conserved, energy
    is pumped in and out, and the geodesic flow becomes **chaotic**
    (positive finite-time Lyapunov exponent, mixed Poincare section).

    BUT the flow is still **Hamiltonian**, so by Liouville's theorem the
    phase-space divergence is identically zero and **no attractor can
    form** -- in stark contrast to the dissipative Lorenz rotation
    (rotating_dust_lorenz.py), whose divergence tr(J) = -(sigma+1+b) < 0.

Hamiltonian (interior van Stockum, z frozen, (t, r, phi) block)
--------------------------------------------------------------
Metric block g_{mu nu} with omega = omega(t):

    g_tt = -1,  g_tphi = omega r^2,  g_phiphi = r^2(1 - omega^2 r^2),
    g_rr = exp(-omega^2 r^2).

Its inverse gives the geodesic Hamiltonian H = 1/2 g^{mu nu} p_mu p_nu:

    H = 1/2 [ (omega^2 r^2 - 1) p_t^2 + 2 omega p_t p_phi
              + p_phi^2 / r^2 + exp(omega^2 r^2) p_r^2 ].

Because phi is cyclic, p_phi = ell is conserved. H itself is conserved
along the flow (no explicit dependence on the affine parameter tau), and
fixes the mass shell H = -1/2 (timelike). When omega is constant, p_t is
also conserved and the system is integrable.

CAVEAT (stated up front): a genuinely time-dependent omega(t) makes the
metric a *prescribed, non-vacuum* background, not an exact Einstein
solution. This module is therefore a test-particle-in-prescribed-field
toy whose purpose is the dynamical-systems statement (symmetry breaking
-> conservative chaos; Liouville -> no attractor), not a claim about an
exact rotating-dust spacetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class GeodesicRotation:
    """Test-particle geodesics in the interior van Stockum (t, r, phi) block
    with a prescribed (possibly time-dependent) angular velocity omega(t).

    Parameters
    ----------
    omega_fn : callable t -> omega(t)
    omega_dot_fn : callable t -> d omega / dt  (analytic derivative)
    ell : float
        Conserved angular momentum p_phi.
    """

    omega_fn: Callable[[float], float]
    omega_dot_fn: Callable[[float], float]
    ell: float

    def hamiltonian(self, state: np.ndarray) -> float:
        """H = 1/2 g^{mu nu} p_mu p_nu at (t, r, phi, p_t, p_r, p_phi)."""
        t, r, phi, p_t, p_r, p_phi = state
        w = self.omega_fn(t)
        e = np.exp(w * w * r * r)
        return 0.5 * (
            (w * w * r * r - 1.0) * p_t * p_t
            + 2.0 * w * p_t * p_phi
            + p_phi * p_phi / (r * r)
            + e * p_r * p_r
        )

    def rhs(self, tau: float, state: np.ndarray) -> np.ndarray:
        """Hamilton's equations d state / d tau."""
        t, r, phi, p_t, p_r, p_phi = state
        w = self.omega_fn(t)
        wd = self.omega_dot_fn(t)
        e = np.exp(w * w * r * r)

        tdot = (w * w * r * r - 1.0) * p_t + w * p_phi
        rdot = e * p_r
        phidot = w * p_t + p_phi / (r * r)

        # p_t' = -dH/dt = -wd * d/dw [ ... ]
        # dH/dw = w r^2 p_t^2 + p_t p_phi + w r^2 e p_r^2
        dH_dw = w * r * r * p_t * p_t + p_t * p_phi + w * r * r * e * p_r * p_r
        p_t_dot = -wd * dH_dw

        # p_r' = -dH/dr
        p_r_dot = -(
            w * w * r * p_t * p_t
            - p_phi * p_phi / (r * r * r)
            + w * w * r * e * p_r * p_r
        )

        p_phi_dot = 0.0  # phi cyclic
        return np.array([tdot, rdot, phidot, p_t_dot, p_r_dot, p_phi_dot])

    def initial_state_from_E(
        self, r0: float, E: float, p_r_sign: float = 1.0
    ) -> np.ndarray:
        """Build an on-shell timelike initial state (H = -1/2) at r0.

        Sets t=phi=0, p_phi=ell, p_t=-E, and solves for p_r so that
        H = -1/2. Returns the 6-vector; raises if no real p_r exists.
        """
        w = self.omega_fn(0.0)
        e = np.exp(w * w * r0 * r0)
        p_t = -E
        # H = -1/2  =>  e p_r^2 = -1 - [ (w^2 r^2 -1)p_t^2 + 2 w p_t ell + ell^2/r^2 ]
        bracket = (
            (w * w * r0 * r0 - 1.0) * p_t * p_t
            + 2.0 * w * p_t * self.ell
            + self.ell * self.ell / (r0 * r0)
        )
        p_r2 = (-1.0 - bracket) / e
        if p_r2 < 0:
            raise ValueError(
                f"No real p_r at r0={r0}, E={E}, ell={self.ell} (p_r^2={p_r2:.3e})"
            )
        p_r = p_r_sign * np.sqrt(p_r2)
        return np.array([0.0, r0, 0.0, p_t, p_r, self.ell])

    def integrate(
        self, state0: np.ndarray, tau_max: float, n_samples: int = 4001
    ) -> dict:
        tau_eval = np.linspace(0.0, tau_max, n_samples)
        sol = solve_ivp(
            self.rhs, (0.0, tau_max), np.asarray(state0, dtype=float),
            t_eval=tau_eval, method="DOP853", rtol=1e-10, atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"geodesic integration failed: {sol.message}")
        y = sol.y
        return {
            "tau": sol.t, "t": y[0], "r": y[1], "phi": y[2],
            "p_t": y[3], "p_r": y[4], "p_phi": y[5],
        }


def constant_omega(omega0: float) -> GeodesicRotation:
    return GeodesicRotation(
        omega_fn=lambda t: omega0, omega_dot_fn=lambda t: 0.0, ell=0.0
    )


def driven_omega(
    omega0: float, eps: float, drive_freq: float, ell: float
) -> GeodesicRotation:
    """omega(t) = omega0 (1 + eps cos(drive_freq * t)) -- a wobbling column."""
    return GeodesicRotation(
        omega_fn=lambda t: omega0 * (1.0 + eps * np.cos(drive_freq * t)),
        omega_dot_fn=lambda t: -omega0 * eps * drive_freq * np.sin(drive_freq * t),
        ell=ell,
    )


def finite_time_lyapunov(
    geo: GeodesicRotation,
    state0: np.ndarray,
    tau_max: float = 400.0,
    d0: float = 1e-9,
    n_renorm: int = 400,
) -> float:
    """Largest finite-time Lyapunov exponent of the geodesic flow.

    Two-trajectory rescaling in the 6-D phase space. ~0 for the integrable
    (constant-omega) case, > 0 for the driven case.
    """
    seg = tau_max / n_renorm
    keys = ("t", "r", "phi", "p_t", "p_r", "p_phi")

    def step(state):
        out = geo.integrate(state, seg, n_samples=2)
        return np.array([out[k][-1] for k in keys])

    s = np.asarray(state0, dtype=float).copy()
    sp = s.copy()
    sp[4] += d0  # perturb in p_r (a physical phase-space direction)
    log_growth = 0.0
    for _ in range(n_renorm):
        s = step(s)
        sp = step(sp)
        diff = sp - s
        dist = float(np.linalg.norm(diff))
        if dist == 0.0:
            continue
        log_growth += np.log(dist / d0)
        sp = s + (d0 / dist) * diff
    return float(log_growth / tau_max)


def phase_volume_divergence(
    geo: GeodesicRotation, state: np.ndarray, h: float = 1e-6
) -> float:
    """Numerical phase-space divergence sum_i d(state_dot_i)/d(state_i).

    For ANY Hamiltonian flow this is identically zero (Liouville). Computing
    it for the driven geodesic confirms no attractor can form, however
    chaotic the motion; contrast with RotatingDustLorenz.divergence < 0.
    """
    div = 0.0
    for i in range(6):
        sp = state.copy(); sp[i] += h
        sm = state.copy(); sm[i] -= h
        div += (geo.rhs(0.0, sp)[i] - geo.rhs(0.0, sm)[i]) / (2.0 * h)
    return float(div)
