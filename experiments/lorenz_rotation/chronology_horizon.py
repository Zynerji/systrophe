"""Dynamical approach to the chronology horizon (Hawking chronology protection).

The fully nonlinear evolution of a CTC-containing spacetime is impossible as a Cauchy
problem (no global Cauchy surface). The well-posed substitute is the *approach* to the
chronology horizon: spin the cylinder up toward and through the supercritical threshold
a = 1/2 and ask whether the renormalized quantum stress tensor diverges at the forming
chronology horizon (chronology protection), while the classical geometry stays smooth.

Diagnostics per rotation a (reusing the validated Phase-2a machinery):
  * chronology / Cauchy horizon r_H = first zero of F = -g_tt (exists iff a > 1/2);
  * CTC band: first radius where L = g_phiphi < 0;
  * Boulware <T_tt> divergence power near r_H (the protection signal; ~ -1 = 1/(r-r_H));
  * surface gravity kappa = (1/2)|F'(r_H)| (sets the divergence amplitude / T_Hawking);
  * classical Kretschmann scalar near r_H (must stay BOUNDED -> classically smooth).

Result: for a <= 1/2 there is no horizon and <T> is finite (no protection needed); the
instant a crosses 1/2 the horizon appears and <T_tt> diverges as 1/(r-r_H) (power ~ -1,
alpha-universal) while the Kretschmann stays finite -- exactly Hawking's chronology
protection. Driving a(t) chaotically across 1/2 gives a flickering protection barrier.
"""

from __future__ import annotations

import warnings

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.ctc.stress_energy_ctc import divergence_rate_at_horizon
from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate, surface_gravity_at_horizon
from systrophe.qftcs.point_splitting import kretschmann_scalar

warnings.filterwarnings("ignore")

A_CRIT = 0.5   # supercritical (Bonnor Case III) threshold a = omega R


def first_ctc_radius(vs: VanStockumInterior, r_min=1.001, r_max=12.0, n=4000):
    """First radius where L = g_phiphi < 0 (CTC band inner edge), or None."""
    r = np.linspace(r_min, r_max, n)
    L = np.asarray(vs.analytic_exterior_L(r), float)
    neg = np.where(L < 0)[0]
    return float(r[neg[0]]) if len(neg) else None


def chronology_signal(a: float, R: float = 1.0, n_fit: int = 12) -> dict:
    """Chronology-protection diagnostics at rotation a = omega R."""
    vs = VanStockumInterior(omega=a / R, R=R)
    out = {"a": a, "supercritical": a > A_CRIT}
    if a <= A_CRIT:
        # subcritical: no Cauchy horizon, no CTC, <T> finite everywhere
        out.update({"r_horizon": None, "kappa": None, "T_div_power": None,
                    "kretschmann_at_rH": None, "kretschmann_bounded": True,
                    "ctc_radius": None, "protected": False})
        return out
    horizons = cauchy_horizon_estimate(vs)
    rH = float(horizons[0])
    kappa = float(surface_gravity_at_horizon(vs, rH))
    fit = divergence_rate_at_horizon(vs, rH, n_samples=n_fit)
    Kr = float(kretschmann_scalar(vs, rH + 0.02))
    out.update({
        "r_horizon": rH, "kappa": kappa,
        "T_div_power": float(fit.power), "T_diverges": bool(fit.diverges),
        "kretschmann_at_rH": Kr, "kretschmann_bounded": bool(np.isfinite(Kr)),
        "ctc_radius": first_ctc_radius(vs),
        # protection = quantum <T> diverges while classical curvature stays finite
        "protected": bool(fit.diverges and np.isfinite(Kr)),
    })
    return out


def approach_sequence(a_values: np.ndarray) -> list:
    """Quasi-static spin-up sequence across the threshold."""
    return [chronology_signal(float(a)) for a in a_values]


def chaotic_barrier(a_center: float = 0.62, amp: float = 0.18,
                    lorenz_t_max: float = 80.0, dt: float = 0.02) -> dict:
    """Lorenz a(t) wandering across the threshold a = 1/2 -> flickering chronology-
    protection barrier. Barrier 'strength' = surface gravity kappa when supercritical
    (the divergence amplitude), zero when subcritical (no horizon)."""
    from rotating_dust_lorenz import RotatingDustLorenz, rotation_parameter_timeseries

    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=lorenz_t_max, dt=dt, t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=a_center, eps=amp / a_center, clip_min=0.05)
    t = rot["t"] - rot["t"][0]
    a_t = rot["a"]
    barrier = np.zeros_like(a_t)
    horizon = np.full_like(a_t, np.nan)
    for i, a in enumerate(a_t):
        if a > A_CRIT:
            vs = VanStockumInterior(omega=a, R=1.0)
            hz = cauchy_horizon_estimate(vs)
            if len(hz):
                barrier[i] = float(surface_gravity_at_horizon(vs, float(hz[0])))
                horizon[i] = float(hz[0])
    frac_super = float(np.mean(a_t > A_CRIT))
    return {
        "t": t, "a": a_t, "barrier_strength": barrier, "horizon_radius": horizon,
        "fraction_supercritical": frac_super,
        "n_threshold_crossings": int(np.sum(np.abs(np.diff(np.sign(a_t - A_CRIT))) > 0)),
        "a_min": float(a_t.min()), "a_max": float(a_t.max()),
    }
