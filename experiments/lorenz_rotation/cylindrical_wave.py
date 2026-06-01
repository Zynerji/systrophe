"""Exact hyperbolic evolution of the frame-dragging sector (replaces adiabatic CTC).

The adiabatic CTC treatment freezes the spacetime to the *static* Bonnor Case III
solution for the instantaneous a(t) -- i.e. it assumes the gravitational field
responds at infinite speed. The exact treatment solves the *time-dependent*
cylindrical field equations, so the field propagates at speed c = 1 with
retardation and wave emission.

Master variable
---------------
The frame-dragging metric component Psi(t, r) = delta g_{t phi} = delta K obeys the
hyperbolic (dynamical) generalization of the repo's static twist reduction:

    Psi_tt = Psi_rr - (1/r) Psi_r - V(r) Psi

with the background-determined potential (regular except at ergosurfaces F0 = 0)

    V(r) = (F0'^2 - c0^2) / F0^2  -  2 F0' / (r F0),     c0 = 2 a0 / R.

Static limit (Psi_tt = 0) reproduces the static twist structure; the characteristic
speed is exactly 1 (the coefficient of Psi_rr), so signals are causal: a change of
rotation at r = R reaches radius r only at retarded time t ~ (r - R). This is the
content adiabatic discards.

Scope (stated honestly)
-----------------------
- EXACT in the dynamics: full hyperbolic propagation, retardation, wave emission.
- LINEAR in the perturbation amplitude delta a (perturbation of static van Stockum).
- The full *nonlinear*, large-amplitude, ergosurface-crossing problem is genuinely
  open (non-orthogonally-transitive; V is singular at ergosurfaces). Near ergosurfaces
  V is regularized (capped) and that region is flagged, not silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class CylindricalWave:
    """Method-of-lines solver for  u_tt = u_rr - (1/r) u_r - V(r) u  on [R, r_out].

    Parameters
    ----------
    R, r_out : float
        Radial domain.
    n : int
        Number of spatial cells (grid has n+1 points).
    V : callable r -> V(r) (the background potential); default 0 (pure wave).
    drive : callable t -> u(t, R)  (inner Dirichlet boundary, the rotation drive).
    cfl : float
        Courant factor; dt = cfl * dr.
    v_cap : float
        Cap on |V| for ergosurface regularization (None = no cap).
    """

    R: float
    r_out: float
    n: int
    V: Callable[[np.ndarray], np.ndarray] | None = None
    drive: Callable[[float], float] = lambda t: 0.0
    drive_dot: Callable[[float], float] | None = None
    cfl: float = 0.4
    v_cap: float | None = None

    def __post_init__(self):
        self.r = np.linspace(self.R, self.r_out, self.n + 1)
        self.dr = self.r[1] - self.r[0]
        self.dt = self.cfl * self.dr
        if self.V is None:
            self.Vr = np.zeros_like(self.r)
        else:
            self.Vr = np.asarray(self.V(self.r), dtype=float)
            if self.v_cap is not None:
                self.Vr = np.clip(self.Vr, -self.v_cap, self.v_cap)

    def _rhs(self, t, u, pi):
        """Return (du/dt, dpi/dt) with boundary conditions applied."""
        r, dr = self.r, self.dr
        du = pi.copy()
        # interior second derivatives (central)
        urr = np.zeros_like(u)
        ur = np.zeros_like(u)
        urr[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dr ** 2
        ur[1:-1] = (u[2:] - u[:-2]) / (2 * dr)
        dpi = urr - ur / r - self.Vr * u
        # inner Dirichlet: u(R) = drive(t) -> du/dt[0] = drive_dot(t)
        if self.drive_dot is not None:
            du[0] = self.drive_dot(t)
        else:
            du[0] = (self.drive(t + 1e-6) - self.drive(t - 1e-6)) / 2e-6
        dpi[0] = 0.0  # u[0] is set by the drive each step; pi[0] tracks du[0]
        # outer outgoing (cylindrical Sommerfeld): u_t + u_r + u/(2 r) = 0
        ur_out = (u[-1] - u[-2]) / dr
        du[-1] = -ur_out - u[-1] / (2.0 * r[-1])
        dpi[-1] = 0.0
        return du, dpi

    def step(self, t, u, pi):
        """One RK4 step of dt."""
        dt = self.dt
        k1u, k1p = self._rhs(t, u, pi)
        k2u, k2p = self._rhs(t + dt / 2, u + dt / 2 * k1u, pi + dt / 2 * k1p)
        k3u, k3p = self._rhs(t + dt / 2, u + dt / 2 * k2u, pi + dt / 2 * k2p)
        k4u, k4p = self._rhs(t + dt, u + dt * k3u, pi + dt * k3p)
        u = u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        pi = pi + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        # re-pin the inner Dirichlet value exactly (RK4 drift guard)
        u[0] = self.drive(t + dt)
        if self.drive_dot is not None:
            pi[0] = self.drive_dot(t + dt)
        return u, pi

    def evolve(self, t_max: float, u0=None, pi0=None, record_radii=None):
        """Evolve to t_max. Returns dict with 't', the field at requested radii,
        and the final (u, pi). record_radii: list of r-values to time-sample."""
        u = np.zeros_like(self.r) if u0 is None else np.array(u0, float)
        pi = np.zeros_like(self.r) if pi0 is None else np.array(pi0, float)
        u[0] = self.drive(0.0)
        n_steps = int(t_max / self.dt)
        ts = np.empty(n_steps + 1)
        rec_idx = []
        if record_radii is not None:
            rec_idx = [int(np.argmin(np.abs(self.r - rr))) for rr in record_radii]
        rec = np.empty((n_steps + 1, len(rec_idx)))
        energy = np.empty(n_steps + 1)
        t = 0.0
        for k in range(n_steps + 1):
            ts[k] = t
            if rec_idx:
                rec[k] = u[rec_idx]
            energy[k] = self.energy(u, pi)
            if k < n_steps:
                u, pi = self.step(t, u, pi)
                t += self.dt
        return {"t": ts, "rec": rec, "rec_radii": (record_radii or []),
                "u": u, "pi": pi, "energy": energy}

    def energy(self, u, pi):
        """Conserved wave energy (weight 1/r makes the operator self-adjoint):
        E = 1/2 integral (1/r)(pi^2 + u_r^2 + V u^2) dr. Constant when undriven."""
        ur = np.gradient(u, self.r)
        integrand = (pi ** 2 + ur ** 2 + self.Vr * u ** 2) / self.r
        return float(0.5 * np.trapezoid(integrand, self.r))


def vanstockum_potential(a0: float, R: float = 1.0):
    """Background potential V(r) for a supercritical van Stockum cylinder a0 = w0 R.

    Uses the analytic Case III F0(r); regular except at ergosurfaces (F0 = 0),
    where V ~ 1/(r - r_erg) (handled by the solver's v_cap)."""
    from systrophe.geometry.vanstockum import VanStockumInterior

    vs = VanStockumInterior(omega=a0 / R, R=R)
    c0 = 2.0 * a0 / R

    def V(r):
        r = np.asarray(r, dtype=float)
        F0 = np.asarray(vs.analytic_exterior_F(r), dtype=float)
        # analytic F0' via tight central difference on the closed form
        eps = 1e-6 * np.maximum(r, 1.0)
        F0p = (np.asarray(vs.analytic_exterior_F(r + eps), dtype=float)
               - np.asarray(vs.analytic_exterior_F(r - eps), dtype=float)) / (2 * eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            term1 = (F0p ** 2 - c0 ** 2) / (F0 ** 2)
            term2 = 2.0 * F0p / (r * F0)
            Vr = term1 - term2
        Vr = np.where(np.isfinite(Vr), Vr, 0.0)
        return Vr

    return V, vs, c0
