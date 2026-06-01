"""Nonlinear gravitational-wave emission from a chaotically-rotating cylinder.

Next step beyond the nonlinear evolution (nonlinear_cylindrical.py): instead of just
showing the spacetime is stable, EXTRACT the radiation and ask whether it carries the
strange-attractor fingerprint -- including through nonlinear mode-mixing.

The cylinder's twist omega(t,R) (the gravitomagnetic/rotational polarization) is driven by
the chaotic Lorenz rotation a(t). The full nonlinear cylindrical-vacuum evolution then
radiates two polarizations to a far detector:
  * omega : the directly-driven twist (a smooth observable of the attractor)
  * psi   : generated ONLY nonlinearly (the omega^2 source term) -- absent in linear theory

Results computed here:
  1. Takens correlation dimension D2 of the RADIATED omega(t) and psi(t) at a far detector
     -- does the gravitational radiation reconstruct the Lorenz attractor (D2 ~ 2.06)?
  2. Nonlinear cross-polarization conversion efficiency vs drive amplitude (~A^2).
  3. The radiated C-energy (Thorne) flux time series and its chaos.

Honest scope: this is the well-posed (z,phi)-twist sector (omega gravitomagnetic, g_phiphi>0,
no CTC). The fully nonlinear (t,phi)-CTC frame-dragging evolution is blocked at a deeper
level -- a spacetime with CTCs has no global Cauchy surface, so it is not a well-posed
initial-value problem (see FINDINGS).
"""

from __future__ import annotations

import numpy as np

from nonlinear_cylindrical import NonlinearCylindrical
from rotating_dust_lorenz import RotatingDustLorenz, rotation_parameter_timeseries
from deep_tests import correlation_dimension_scalar


def lorenz_twist_drive(eps=0.04, a0=1.5, lorenz_t_max=140.0, dt=0.01):
    """Lorenz a(t) -> twist boundary drive omega(t,R) = eps*(a(t)-a0)."""
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=lorenz_t_max, dt=dt, t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=a0, eps=eps, clip_min=0.55)
    t_l = rot["t"] - rot["t"][0]
    sig = (rot["a"] - a0)
    return t_l, sig


def emit(eps=0.04, r_out=120.0, n=4000, t_max=100.0, r_detector=90.0):
    """Evolve the nonlinear cylinder driven by the chaotic twist; return the
    radiated waves at a far detector + the C-energy time series."""
    t_l, sig = lorenz_twist_drive(eps=eps)
    amp = float(np.max(np.abs(sig)))
    drive = lambda t: float(np.interp(t, t_l, sig))
    ev = NonlinearCylindrical(r_in=1.0, r_out=r_out, n=n, omega_drive=drive,
                              kappa=1.0, sigma_ko=0.02)
    res = ev.evolve(t_max=t_max, record_radii=[r_detector])
    t = res["t"]
    om_det = res["rec"][:, 0, 1]
    psi_det = res["rec"][:, 0, 0]
    # restrict to after the signal front has reached the detector (retardation)
    t_arr = (r_detector - 1.0)
    mask = t > t_arr + 5.0
    return {
        "t": t[mask], "omega_det": om_det[mask], "psi_det": psi_det[mask],
        "c_energy": res["c_energy"], "t_full": t,
        "drive_amplitude": amp, "r_detector": r_detector,
        "max_omega_det": float(np.max(np.abs(om_det[mask]))),
        "max_psi_det": float(np.max(np.abs(psi_det[mask]))),
    }


def attractor_in_radiation(eps=0.04, **kw):
    """D2 of the radiated waves vs the Lorenz attractor (~2.06)."""
    out = emit(eps=eps, **kw)
    d2_om = correlation_dimension_scalar(out["omega_det"], m=5)
    d2_psi = correlation_dimension_scalar(out["psi_det"], m=5)
    return {
        "drive_amplitude": out["drive_amplitude"],
        "D2_radiated_omega": d2_om["D2"],
        "D2_radiated_psi": d2_psi["D2"],
        "max_omega_det": out["max_omega_det"],
        "max_psi_det": out["max_psi_det"],
        "n_samples": len(out["omega_det"]),
    }


def cross_polarization_scaling(amps=(0.02, 0.04, 0.08), r_out=60.0, n=2500,
                               t_max=60.0, r_detector=45.0):
    """Nonlinear conversion: max|psi_radiated| should scale ~ A^2 (omega -> psi via
    the omega^2 source); linear theory gives zero psi."""
    rows = []
    for A in amps:
        out = emit(eps=A, r_out=r_out, n=n, t_max=t_max, r_detector=r_detector)
        rows.append({"eps": A,
                     "max_omega": out["max_omega_det"],
                     "max_psi": out["max_psi_det"],
                     "ratio_psi_over_omega": out["max_psi_det"] / (out["max_omega_det"] + 1e-30)})
    return rows
