"""Gauge-fixed, constraint-damped NONLINEAR cylindrical-vacuum evolution.

This is the nonlinear capstone. The linear coupled analysis (derive_coupled / run_coupled)
showed the supercritical "instability" was a single-variable artifact; here we evolve the
full *nonlinear* cylindrical Einstein vacuum in a well-posed, gauge-fixed, constraint-damped
formulation and confirm stability + the standard nonlinear physics.

Formulation (Jordan-Ehlers-Kundt / Einstein-Rosen, two polarizations with rotation)
-----------------------------------------------------------------------------------
    ds^2 = e^{2(gamma-psi)}(-dt^2+dr^2) + e^{2psi}(dz + omega dphi)^2 + r^2 e^{-2psi} dphi^2

GAUGE FIXING (no residual gauge freedom):
  * areal radial coordinate r  (the phi-z determinant is fixed = r^2, built into the ansatz)
  * conformal time gauge  (the (t,r) block is e^{2(gamma-psi)}(-dt^2+dr^2))
  => the coordinate light speed is exactly 1; there are no lapse/shift dynamics to evolve.

Physical DOF: psi(t,r), omega(t,r)  -- two nonlinear wave-map fields (omega is the
gravitomagnetic twist = nonlinear frame-dragging polarization). gamma is fixed by the
constraints:
  HAMILTONIAN:  gamma_r = r(psi_r^2 + psi_t^2) + (e^{4psi}/(4r))(omega_r^2 + omega_t^2)
  MOMENTUM:     gamma_t = 2 r psi_r psi_t   + (e^{4psi}/(2r)) omega_r omega_t

Nonlinear evolution equations (exact vacuum):
  psi_tt   = psi_rr + psi_r/r + (e^{4psi}/(2r^2))(omega_t^2 - omega_r^2)
  omega_tt = omega_rr - omega_r/r + 4(psi_t omega_t - psi_r omega_r)

CONSTRAINT DAMPING (Z4-style): gamma is evolved with the momentum constraint and damped
toward the Hamiltonian-constraint value gamma_H(r) = gamma(R) + integral of gamma_r:
  gamma_t = [momentum RHS] - kappa (gamma - gamma_H)
so the constraint  C = gamma - gamma_H  obeys  C_t = -kappa C  (exponential damping).

References: Einstein-Rosen 1937; Jordan-Ehlers-Kundt; Thorne C-energy 1965;
Piran-Safier-Stark (nonlinear cylindrical NR); Gundlach et al. (constraint damping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _dr(y, dr):
    d = np.empty_like(y)
    d[1:-1] = (y[2:] - y[:-2]) / (2 * dr)
    d[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * dr)
    d[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * dr)
    return d


def _drr(y, dr):
    d = np.empty_like(y)
    d[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dr ** 2
    d[0] = d[1]; d[-1] = d[-2]
    return d


def _ko(y, sigma):
    """Kreiss-Oliger 4th-order dissipation (numerical hygiene)."""
    d = np.zeros_like(y)
    d[2:-2] = (y[:-4] - 4 * y[1:-3] + 6 * y[2:-2] - 4 * y[3:-1] + y[4:])
    return -sigma * d


@dataclass
class NonlinearCylindrical:
    """Method-of-lines evolver for the nonlinear JEK cylindrical-vacuum system.

    Parameters
    ----------
    r_in, r_out, n : radial domain / resolution
    psi_drive, omega_drive : callables t -> boundary value at r_in (None = static end)
    kappa : constraint-damping rate
    sigma_ko : KO dissipation strength
    cfl : Courant factor
    """

    r_in: float = 1.0
    r_out: float = 21.0
    n: int = 2000
    psi_drive: Callable[[float], float] | None = None
    omega_drive: Callable[[float], float] | None = None
    kappa: float = 1.0
    sigma_ko: float = 0.02
    cfl: float = 0.4

    def __post_init__(self):
        self.r = np.linspace(self.r_in, self.r_out, self.n + 1)
        self.dr = self.r[1] - self.r[0]
        self.dt = self.cfl * self.dr

    # -- right-hand sides --------------------------------------------------- #
    def _rhs(self, t, psi, Pps, om, Pom, gam):
        r, dr = self.r, self.dr
        psi_r = _dr(psi, dr); psi_rr = _drr(psi, dr)
        om_r = _dr(om, dr); om_rr = _drr(om, dr)
        e4 = np.exp(4.0 * psi)

        dpsi = Pps.copy()
        dPps = psi_rr + psi_r / r + (e4 / (2 * r ** 2)) * (Pom ** 2 - om_r ** 2)
        dom = Pom.copy()
        dPom = om_rr - om_r / r + 4.0 * (Pps * Pom - psi_r * om_r)

        # gamma: momentum-constraint evolution + damping toward Hamiltonian value
        gam_t_mom = 2 * r * psi_r * Pps + (e4 / (2 * r)) * om_r * Pom
        gam_H = self._gamma_hamiltonian(psi, Pps, om, Pom, gam[0])
        dgam = gam_t_mom - self.kappa * (gam - gam_H)

        # KO dissipation
        dpsi += _ko(psi, self.sigma_ko); dPps += _ko(Pps, self.sigma_ko)
        dom += _ko(om, self.sigma_ko); dPom += _ko(Pom, self.sigma_ko)

        # boundaries
        self._apply_bc(t, dpsi, dPps, dom, dPom, psi, Pps, om, Pom, psi_r, om_r)
        dgam[0] = 0.0; dgam[-1] = gam_t_mom[-1]  # gamma inner pinned, outer free
        return dpsi, dPps, dom, dPom, dgam

    def _gamma_hamiltonian(self, psi, Pps, om, Pom, gam_in):
        r, dr = self.r, self.dr
        psi_r = _dr(psi, dr); om_r = _dr(om, dr)
        e4 = np.exp(4.0 * psi)
        integrand = r * (psi_r ** 2 + Pps ** 2) + (e4 / (4 * r)) * (om_r ** 2 + Pom ** 2)
        gam = np.empty_like(r)
        gam[0] = gam_in
        gam[1:] = gam_in + np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dr)
        return gam

    def _apply_bc(self, t, dpsi, dPps, dom, dPom, psi, Pps, om, Pom, psi_r, om_r):
        r, dr = self.r, self.dr
        # inner: Dirichlet drive (or static)
        if self.psi_drive is not None:
            dpsi[0] = self._ddt(self.psi_drive, t); dPps[0] = 0.0
        else:
            dpsi[0] = 0.0; dPps[0] = 0.0
        if self.omega_drive is not None:
            dom[0] = self._ddt(self.omega_drive, t); dPom[0] = 0.0
        else:
            dom[0] = 0.0; dPom[0] = 0.0
        # outer: cylindrical outgoing (Sommerfeld) f_t = -(f_r + f/(2r))
        dpsi[-1] = -(psi_r[-1] + psi[-1] / (2 * r[-1]))
        dPps[-1] = 0.0
        dom[-1] = -(om_r[-1] + om[-1] / (2 * r[-1]))
        dPom[-1] = 0.0

    @staticmethod
    def _ddt(fn, t, h=1e-6):
        return (fn(t + h) - fn(t - h)) / (2 * h)

    # -- time stepping ------------------------------------------------------ #
    def step(self, t, state):
        dt = self.dt

        def add(s, k, f):
            return tuple(si + f * dt * ki for si, ki in zip(s, k))

        k1 = self._rhs(t, *state)
        k2 = self._rhs(t + dt / 2, *add(state, k1, 0.5))
        k3 = self._rhs(t + dt / 2, *add(state, k2, 0.5))
        k4 = self._rhs(t + dt, *add(state, k3, 1.0))
        new = tuple(s + dt / 6 * (a + 2 * b + 2 * c + d)
                    for s, a, b, c, d in zip(state, k1, k2, k3, k4))
        # re-pin inner Dirichlet exactly
        psi, Pps, om, Pom, gam = new
        if self.psi_drive is not None:
            psi[0] = self.psi_drive(t + dt)
        if self.omega_drive is not None:
            om[0] = self.omega_drive(t + dt)
        return (psi, Pps, om, Pom, gam)

    def evolve(self, t_max, psi0=None, om0=None, Pps0=None, Pom0=None,
               record_radii=None):
        r = self.r
        psi = np.zeros_like(r) if psi0 is None else np.array(psi0, float)
        om = np.zeros_like(r) if om0 is None else np.array(om0, float)
        Pps = np.zeros_like(r) if Pps0 is None else np.array(Pps0, float)
        Pom = np.zeros_like(r) if Pom0 is None else np.array(Pom0, float)
        if self.psi_drive is not None:
            psi[0] = self.psi_drive(0.0)
        if self.omega_drive is not None:
            om[0] = self.omega_drive(0.0)
        gam = self._gamma_hamiltonian(psi, Pps, om, Pom, 0.0)
        state = (psi, Pps, om, Pom, gam)

        nsteps = int(t_max / self.dt)
        rec_idx = ([int(np.argmin(np.abs(r - rr))) for rr in record_radii]
                   if record_radii is not None else [])
        ts = np.empty(nsteps + 1)
        c_energy = np.empty(nsteps + 1)
        cviol = np.empty(nsteps + 1)
        rec = np.empty((nsteps + 1, len(rec_idx), 2))
        t = 0.0
        for s in range(nsteps + 1):
            psi, Pps, om, Pom, gam = state
            ts[s] = t
            c_energy[s] = self.c_energy(psi, Pps, om, Pom)
            cviol[s] = self.constraint_violation(psi, Pps, om, Pom, gam)
            for j, idx in enumerate(rec_idx):
                rec[s, j] = (psi[idx], om[idx])
            if s < nsteps:
                state = self.step(t, state)
                t += self.dt
        return {"t": ts, "c_energy": c_energy, "constraint_violation": cviol,
                "rec": rec, "rec_radii": (record_radii or []), "state": state}

    # -- diagnostics -------------------------------------------------------- #
    def c_energy(self, psi, Pps, om, Pom):
        """Total C-energy = integral of gamma_r (Thorne). Conserved up to flux."""
        r, dr = self.r, self.dr
        psi_r = _dr(psi, dr); om_r = _dr(om, dr)
        e4 = np.exp(4.0 * psi)
        integrand = r * (psi_r ** 2 + Pps ** 2) + (e4 / (4 * r)) * (om_r ** 2 + Pom ** 2)
        return float(np.trapezoid(integrand, r))

    def constraint_violation(self, psi, Pps, om, Pom, gam):
        gam_H = self._gamma_hamiltonian(psi, Pps, om, Pom, gam[0])
        return float(np.sqrt(np.mean((gam - gam_H) ** 2)))


def levi_civita_static(r, sigma=0.2):
    """Static cylindrical vacuum (Levi-Civita): psi = sigma ln r, omega = 0.
    Satisfies psi_rr + psi_r/r = 0 -> exact static solution (fixed point)."""
    return sigma * np.log(r), np.zeros_like(r)
