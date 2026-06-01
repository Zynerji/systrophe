"""Evolve the coupled linearized (dF,dK,dL,dh,dS) vacuum system on van Stockum.

Settles whether the single-variable frame-dragging instability (rotating_dust_lorenz
/ cylindrical_wave) was physical or a truncation artifact, by evolving the FULL
coupled metric-perturbation system derived in derive_coupled.py.

Method (free evolution of the linearized equations)
---------------------------------------------------
State q = (dF,dK,dL,dh,dS)(t,r), momentum p = q_t. The five EVOLUTION components of
the linearized Einstein tensor (tt, tphi, rr, phiphi, zz) are linear in the second
time derivatives:  M(r) . q_tt + rest(q, q_t, q_r, q_rr, q_tr) = 0, where M(r) is
BACKGROUND-determined (computed once by probing the equations with q_tt basis
vectors). We solve q_tt = -pinv(M) . rest and step with RK4. The two momentum
constraints (tr, rphi) are monitored, not imposed.

Ergosurface regularization: the conformal factor h = g_rr = g_zz diverges (~e^700)
at F0 = 0 in Lewis-Papapetrou coordinates; every h-term in the equations carries
1/h or 1/h^2 and -> 0 there, so capping h at H_CAP realises that regular limit
without float overflow (verified in validate_coupled.py).
"""

from __future__ import annotations

import warnings

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.geometry.lewis_papapetrou import integrate_lp_exterior

from derive_coupled import get_system

warnings.filterwarnings("ignore")
H_CAP = 1e40

EVOL_KEYS = [(0, 0), (0, 2), (1, 1), (2, 2), (3, 3)]   # tt, tphi, rr, phiphi, zz
CONSTR_KEYS = [(0, 1), (1, 2)]                          # tr, rphi
FIELDS = ["dF", "dK", "dL", "dh", "dS"]


def _ddr(y, r):
    return np.gradient(y, r, edge_order=2)


def _d2dr(y, dr):
    """Second derivative via the standard 3-point stencil (stable)."""
    d2 = np.empty_like(y)
    d2[1:-1] = (y[2:] - 2 * y[1:-1] + y[:-2]) / dr ** 2
    d2[0] = d2[1]; d2[-1] = d2[-2]
    return d2


def background(a0, R, r):
    vs = VanStockumInterior(omega=a0 / R, R=R)
    F0 = np.asarray(vs.analytic_exterior_F(r), float)
    K0 = np.asarray(vs.analytic_exterior_K(r), float)
    L0 = np.asarray(vs.analytic_exterior_L(r), float)
    lp = integrate_lp_exterior(omega_dust=a0 / R, R=R, r_max=float(r[-1]) + 0.5,
                               n_samples=8000)
    h0 = np.clip(np.interp(r, lp.r, lp.h), None, H_CAP)
    S0 = h0.copy()
    d = {}
    for nm, y in [("F0", F0), ("K0", K0), ("L0", L0), ("h0", h0), ("S0", S0)]:
        d[nm] = y; d[nm + "r"] = _ddr(y, r); d[nm + "rr"] = _ddr(_ddr(y, r), r)
    d["_vs"] = vs
    return d


class CoupledEvolver:
    def __init__(self, a0=1.5, R=1.0, r_min=1.1, r_max=6.0, n=240):
        self.r = np.linspace(r_min, r_max, n)
        self.dr = self.r[1] - self.r[0]
        self.bg = background(a0, R, self.r)
        sysd = get_system()
        self.order = sysd["order"]
        self.lambdas = sysd["lambdas"]
        self.M = self._build_M()           # (n, 5, 5) background mass matrix
        self.Minv = np.linalg.pinv(self.M)  # tolerant to gauge rank-deficiency
        # diagnostic: rank of M
        self.rankM = int(np.median([np.linalg.matrix_rank(self.M[i], tol=1e-8)
                                    for i in range(0, len(self.r), 10)]))

    # ---- argument assembly ------------------------------------------------ #
    def _args(self, pert):
        """Build the 45-symbol positional arg list from background + perturbation
        dict `pert` (keys like dF, dF_t, dF_r, dF_tt, dF_rr, dF_tr)."""
        val = {o: self.bg[o] for o in self.order if o in self.bg}
        z = np.zeros_like(self.r)
        for nm in FIELDS:
            for suf in ["", "_t", "_r", "_tt", "_rr", "_tr"]:
                val[nm + suf] = pert.get(nm + suf, z)
        return [val[o] for o in self.order]

    def _build_M(self):
        n = len(self.r)
        M = np.zeros((n, 5, 5))
        for j, fj in enumerate(FIELDS):          # column: unit q_tt for field j
            pert = {fj + "_tt": np.ones_like(self.r)}
            args = self._args(pert)
            for i, key in enumerate(EVOL_KEYS):
                M[:, i, j] = np.asarray(self.lambdas[key](*args), float)
        return M

    def _rest(self, q, p):
        """Lower-order part of the 5 evolution equations (q_tt = 0)."""
        pert = {}
        for k, nm in enumerate(FIELDS):
            pert[nm] = q[k]
            pert[nm + "_t"] = p[k]
            pert[nm + "_r"] = _ddr(q[k], self.r)
            pert[nm + "_rr"] = _d2dr(q[k], self.dr)   # stable 3-point 2nd deriv
            pert[nm + "_tr"] = _ddr(p[k], self.r)
        args = self._args(pert)
        rest = np.zeros((len(self.r), 5))
        for i, key in enumerate(EVOL_KEYS):
            rest[:, i] = np.asarray(self.lambdas[key](*args), float)
        return rest, pert

    def constraints(self, q, p):
        _, pert = self._rest(q, p)
        out = {}
        args = self._args(pert)
        for key, nm in zip(CONSTR_KEYS, ["tr", "rphi"]):
            out[nm] = np.asarray(self.lambdas[key](*args), float)
        return out

    @staticmethod
    def _ko(u, sigma):
        """Kreiss-Oliger dissipation: -sigma * (4th difference), damps the
        high-frequency grid modes that the rank-deficient (gauge) free-evolution
        excites, without touching the resolved physical modes."""
        d4 = np.zeros_like(u)
        d4[2:-2] = (u[:-4] - 4 * u[1:-3] + 6 * u[2:-2] - 4 * u[3:-1] + u[4:])
        return -sigma * d4

    def _rhs(self, q, p, sigma_ko=0.02):
        rest, _ = self._rest(q, p)               # (n,5)
        qtt = -np.einsum("nij,nj->ni", self.Minv, rest)  # (n,5)
        dq = p.copy()
        dp = qtt.T.copy()                         # (5,n)
        for k in range(5):                        # KO dissipation + Dirichlet ends
            dq[k] += self._ko(q[k], sigma_ko)
            dp[k] += self._ko(p[k], sigma_ko)
            dq[k, 0] = dq[k, -1] = 0.0
            dp[k, 0] = dp[k, -1] = 0.0
        return dq, dp

    def evolve(self, t_max, dt=None, q0=None, c_every=25):
        if dt is None:
            dt = 0.4 * self.dr
        n = len(self.r)
        q = np.zeros((5, n)) if q0 is None else q0.copy()
        p = np.zeros((5, n))
        nsteps = int(t_max / dt)
        amp = np.empty(nsteps + 1)
        cnorm = np.full(nsteps + 1, np.nan)
        for s in range(nsteps + 1):
            amp[s] = float(np.max(np.abs(q)))
            if s % c_every == 0 or s == nsteps:        # constraint monitor (sparse)
                c = self.constraints(q, p)
                cnorm[s] = float(np.sqrt(np.mean(c["tr"] ** 2 + c["rphi"] ** 2)))
            if s < nsteps:
                k1q, k1p = self._rhs(q, p)
                k2q, k2p = self._rhs(q + 0.5 * dt * k1q, p + 0.5 * dt * k1p)
                k3q, k3p = self._rhs(q + 0.5 * dt * k2q, p + 0.5 * dt * k2p)
                k4q, k4p = self._rhs(q + dt * k3q, p + dt * k3p)
                q = q + dt / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
                p = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        return {"amp": amp, "constraint_norm": cnorm, "n_steps": nsteps, "dt": dt}

    def gaussian_id(self, field="dK", r0=3.0, width=0.4, amp=1e-3):
        """Small localized initial perturbation in one metric component."""
        q0 = np.zeros((5, len(self.r)))
        k = FIELDS.index(field)
        q0[k] = amp * np.exp(-((self.r - r0) / width) ** 2)
        # taper to zero at boundaries
        q0[k, 0] = q0[k, -1] = 0.0
        return q0
