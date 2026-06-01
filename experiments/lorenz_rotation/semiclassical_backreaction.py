"""Self-consistent semiclassical backreaction at the chronology horizon.

The previous step (chronology_horizon.py) showed the renormalized <T_mu nu> diverges at
the forming chronology horizon while the classical curvature stays finite. This module
pushes to the *self-consistent* semiclassical question: does the divergent backreaction
destroy the perturbative (classical) horizon -- i.e. does quantum backreaction protect
chronology?

A full 4D self-consistent G_mu nu[g] = 8 pi <T_mu nu>[g] near a CTC-forming horizon is an
open research problem (and a CTC spacetime has no global Cauchy surface). What is rigorous
and computable is the *breakdown of the semiclassical expansion*: the dimensionless
backreaction strength

    beta(r) = 8 pi ell^2 |<T_tt>(r)| / kappa^2          (ell^2 = Planck area ~ hbar G)

(quantum energy vs the classical curvature scale kappa^2 at the horizon) diverges at the
horizon because <T_tt> ~ 1/(r - r_H) while kappa is finite. Hence for ANY ell > 0 there is
a shell of width eps_bd around the classical horizon where beta >= 1: the perturbative
vacuum geometry is invalid there and is replaced by a strong-backreaction (quantum-gravity)
region. The classical chronology horizon is therefore never reached -- chronology is
protected by the breakdown of the semiclassical vacuum, and the protection shell closes
(eps_bd -> 0) only in the classical limit ell -> 0, exactly where CTCs would form.

This is the standard chronology-protection mechanism (Hawking 1992; Kim-Thorne 1991;
Visser); here it is made quantitative on the Tipler/van Stockum background using the
validated Phase-2a <T_mu nu>. The 'self-consistent effective horizon' is the fixed point
r_eff = r_H - eps_bd(ell): the quantum-corrected boundary of the usable spacetime.

Honest scope: 2D Polyakov <T> (massless conformal scalar); the breakdown/scaling analysis
is rigorous; a full nonlinear self-consistent metric is beyond scope (and ill-posed once
CTCs form).
"""

from __future__ import annotations

import warnings

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.ctc.stress_energy_ctc import boulware_stress_tensor
from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate, surface_gravity_at_horizon

warnings.filterwarnings("ignore")
A_CRIT = 0.5


def _regular_side_grid(vs, rH, n=150, eps_min=2e-4, eps_max=0.4):
    """Geometric grid approaching r_H from the side where F > 0 (Boulware regular)."""
    eps = np.geomspace(eps_min, eps_max, n)
    if float(vs.analytic_exterior_F(rH - eps_min)) > 0:
        return rH - eps, eps      # approach from inside
    return rH + eps, eps          # approach from outside


def backreaction_profile(a0: float, ell2: float, R: float = 1.0) -> dict:
    """Backreaction strength beta(r) = 8 pi ell^2 |<T_tt>| / kappa^2 near r_H."""
    vs = VanStockumInterior(omega=a0 / R, R=R)
    rH = float(cauchy_horizon_estimate(vs)[0])
    kappa = float(surface_gravity_at_horizon(vs, rH))
    r, eps = _regular_side_grid(vs, rH)
    Ttt = np.array([abs(boulware_stress_tensor(vs, float(rr))["T_tt"]) for rr in r])
    beta = 8.0 * np.pi * ell2 * Ttt / (kappa ** 2)
    return {"r": r, "eps": eps, "T_tt": Ttt, "beta": beta,
            "r_horizon": rH, "kappa": kappa}


def breakdown_shell(a0: float, ell2: float, R: float = 1.0) -> dict:
    """Width eps_bd of the shell where beta >= 1 (semiclassical breakdown).

    Inside eps_bd of the classical horizon the perturbative vacuum is invalid: the
    quantum-corrected ('effective') horizon sits at r_eff = r_H - eps_bd."""
    prof = backreaction_profile(a0, ell2, R=R)
    eps, Ttt, kappa, rH = prof["eps"], prof["T_tt"], prof["kappa"], prof["r_horizon"]
    # <T_tt> ~ A / eps near the horizon (power -1); amplitude A = <T_tt> * eps
    inner = eps < 0.05
    A = float(np.median((Ttt * eps)[inner])) if inner.any() else float(np.median(Ttt * eps))
    # beta = 8 pi ell^2 (A/eps) / kappa^2 = 1  ->  eps_bd = 8 pi ell^2 A / kappa^2
    eps_bd = 8.0 * np.pi * ell2 * A / (kappa ** 2)
    return {
        "ell2": ell2, "r_horizon": rH, "kappa": kappa, "Ttt_amplitude": A,
        "eps_breakdown": eps_bd, "r_effective_horizon": rH - eps_bd,
        "protected": bool(eps_bd > 0.0),
    }


def breakdown_scaling(a0: float, ell2_values, R: float = 1.0) -> dict:
    """eps_bd vs ell^2 -> power law (eps_bd ~ ell^2 for the 1/(r-r_H) divergence)."""
    ell2_values = np.asarray(ell2_values, float)
    eps_bd = np.array([breakdown_shell(a0, float(e), R=R)["eps_breakdown"]
                       for e in ell2_values])
    good = eps_bd > 0
    if good.sum() >= 2:
        p, logA = np.polyfit(np.log(ell2_values[good]), np.log(eps_bd[good]), 1)
    else:
        p = float("nan")
    return {"ell2": ell2_values, "eps_breakdown": eps_bd, "power_vs_ell2": float(p)}


def self_consistent_effective_horizon(a0: float, ell2: float, R: float = 1.0,
                                      n_iter: int = 12) -> dict:
    """Fixed-point iteration for the quantum-corrected horizon.

    The effective horizon is where beta(r) = 1 (backreaction becomes O(1)); iterate
    r_eff^(n+1) = r_H - eps_bd evaluated with <T> recomputed at the current r_eff. For
    the static profile this converges in one step, but we iterate to expose the fixed
    point and confirm the classical horizon r_H is never the self-consistent endpoint."""
    vs = VanStockumInterior(omega=a0 / R, R=R)
    rH = float(cauchy_horizon_estimate(vs)[0])
    kappa = float(surface_gravity_at_horizon(vs, rH))
    r_eff = rH - 0.1
    history = []
    for _ in range(n_iter):
        eps = rH - r_eff
        if eps <= 0:
            break
        Ttt = abs(boulware_stress_tensor(vs, float(r_eff))["T_tt"])
        beta = 8.0 * np.pi * ell2 * Ttt / kappa ** 2
        # move r_eff to where beta = 1: eps_new = 8 pi ell^2 |T_tt| eps / kappa^2
        eps_new = beta * eps                      # since T_tt ~ A/eps -> eps_new = 8pi ell2 A/kappa2
        r_eff = rH - eps_new
        history.append(float(r_eff))
    return {"r_horizon": rH, "r_effective_horizon": float(r_eff),
            "eps_breakdown": float(rH - r_eff), "history": history,
            "horizon_shrouded": bool(rH - r_eff > 0)}


def _shell_fast(a0: float, ell2: float, R: float = 1.0) -> float:
    """Fast eps_bd: amplitude A from 3 <T_tt> evals near r_H (no full profile)."""
    vs = VanStockumInterior(omega=a0 / R, R=R)
    hz = cauchy_horizon_estimate(vs)
    if len(hz) == 0:
        return 0.0
    rH = float(hz[0])
    kappa = float(surface_gravity_at_horizon(vs, rH))
    sgn = -1.0 if float(vs.analytic_exterior_F(rH - 1e-3)) > 0 else 1.0
    epss = np.array([5e-3, 1e-2, 2e-2])
    A = float(np.median([abs(boulware_stress_tensor(vs, rH + sgn * e)["T_tt"]) * e
                         for e in epss]))
    return 8.0 * np.pi * ell2 * A / (kappa ** 2)


def chaotic_breakdown(ell2: float = 1e-3, a_center: float = 0.7, amp: float = 0.18,
                      lorenz_t_max: float = 70.0, dt: float = 0.02,
                      stride: int = 12) -> dict:
    """Lorenz a(t) across the threshold -> time series of the protection-shell width."""
    from rotating_dust_lorenz import RotatingDustLorenz, rotation_parameter_timeseries

    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=lorenz_t_max, dt=dt, t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=a_center, eps=amp / a_center, clip_min=0.05)
    t = (rot["t"] - rot["t"][0])[::stride]
    a_t = rot["a"][::stride]
    shell = np.array([_shell_fast(float(a), ell2) if a > A_CRIT else 0.0 for a in a_t])
    return {"t": t, "a": a_t, "shell_width": shell,
            "fraction_protected": float(np.mean(shell > 0)),
            "max_shell": float(shell.max())}
