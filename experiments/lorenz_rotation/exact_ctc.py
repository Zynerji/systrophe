"""Exact (retarded) vs adiabatic CTC structure under a chaotic rotation a(t).

Uses the hyperbolic frame-dragging solver (cylindrical_wave.py) to evolve the EXACT
time-dependent metric response to a chaotic a(t), and compares it to the adiabatic
(quasi-static) prediction the original construct used.

For a fixed observation radius r_obs in the CTC region:
  K_exact(t, r)     = K0(r) + dK(t, r)             dK from the hyperbolic evolution
  K_adiabatic(t, r) = K0(r) + (dK0/da)(r) * da(t)  instant (infinite-speed) response
  L(t, r)           = (r^2 - K^2) / F0(r)          CTC iff L < 0

The headline number is the fraction of time the two DISAGREE about whether a CTC is
open at r_obs -- i.e. how often adiabatic mispredicts the time-machine state purely by
ignoring the gravitational light-travel delay (r_obs - R).

Linear in da; F0 frozen at the background (leading frame-dragging response).
"""

from __future__ import annotations

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior

from cylindrical_wave import CylindricalWave, vanstockum_potential
from rotating_dust_lorenz import RotatingDustLorenz, rotation_parameter_timeseries


def dK0_da(a0: float, r: np.ndarray, R: float = 1.0, da: float = 1e-4) -> np.ndarray:
    """Static linear response (dK0/da)(r) by central difference in a."""
    vp = VanStockumInterior(omega=(a0 + da) / R, R=R)
    vm = VanStockumInterior(omega=(a0 - da) / R, R=R)
    return (np.asarray(vp.analytic_exterior_K(r)) - np.asarray(vm.analytic_exterior_K(r))) / (2 * da)


def exact_vs_adiabatic_ctc(
    a0: float = 1.5, R: float = 1.0, eps: float = 0.05,
    r_out: float = 30.0, n: int = 2500, t_max: float = 80.0,
    lorenz_t_max: float = 100.0, v_cap: float = 50.0,
    r_obs_list=(2.0, 3.5, 5.0),
    use_potential: bool = True,
) -> dict:
    """Evolve the exact retarded frame-dragging under a Lorenz a(t) and compare to
    adiabatic at several observation radii. Returns time series + disagreement stats."""
    # --- chaotic rotation drive da(t) (small amplitude -> linear regime) ---
    m = RotatingDustLorenz(sigma=10.0, r=28.0, b=8.0 / 3.0)
    traj = m.integrate(np.array([1.0, 1.0, 1.0]), t_max=lorenz_t_max, dt=0.01, t_transient=20.0)
    rot = rotation_parameter_timeseries(traj, a0=a0, eps=eps, clip_min=0.55)
    t_l = rot["t"] - rot["t"][0]
    da_series = rot["a"] - a0

    def drive(t):
        return float(np.interp(t, t_l, da_series))

    def drive_dot(t):
        dt = 1e-3
        return (drive(t + dt) - drive(t - dt)) / (2 * dt)

    # --- exact hyperbolic evolution of dK ---
    V, vs, c0 = vanstockum_potential(a0=a0, R=R)
    # use_potential=False isolates the (stable) causal-retardation physics by
    # dropping the near-field potential (which separately destabilises the
    # supercritical background -- see FINDINGS).
    V_use = V if use_potential else None
    cw = CylindricalWave(R=R, r_out=r_out, n=n, V=V_use, drive=drive,
                         drive_dot=drive_dot, cfl=0.4, v_cap=v_cap)
    res = cw.evolve(t_max=t_max, record_radii=list(r_obs_list))
    t = res["t"]
    r = cw.r

    # background fields at the observation radii
    K0 = np.asarray(vs.analytic_exterior_K(np.array(r_obs_list)))
    F0 = np.asarray(vs.analytic_exterior_F(np.array(r_obs_list)))
    dKda = dK0_da(a0, np.array(r_obs_list), R=R)

    rows = []
    da_t = np.array([drive(tt) for tt in t])
    L_exact_series = np.empty((len(t), len(r_obs_list)))
    L_adiab_series = np.empty((len(t), len(r_obs_list)))
    for j, r_obs in enumerate(r_obs_list):
        dK_exact = res["rec"][:, j]
        dK_adiab = dKda[j] * da_t
        K_ex = K0[j] + dK_exact
        K_ad = K0[j] + dK_adiab
        L_ex = (r_obs ** 2 - K_ex ** 2) / F0[j]
        L_ad = (r_obs ** 2 - K_ad ** 2) / F0[j]
        L_exact_series[:, j] = L_ex
        L_adiab_series[:, j] = L_ad
        ctc_ex = L_ex < 0
        ctc_ad = L_ad < 0
        # only compare after the signal has had time to arrive (retardation window)
        valid = t > (r_obs - R)
        disagree = np.mean(ctc_ex[valid] != ctc_ad[valid]) if valid.any() else float("nan")
        # cross-correlation lag between exact and adiabatic dK
        a = dK_exact[valid] - dK_exact[valid].mean()
        b = dK_adiab[valid] - dK_adiab[valid].mean()
        if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
            corr = np.correlate(a, b, mode="full")
            lags = np.arange(-len(a) + 1, len(a)) * (t[1] - t[0])
            lag = float(lags[np.argmax(corr)])
            rms = float(np.sqrt(np.mean((dK_exact[valid] - dK_adiab[valid]) ** 2))
                        / (np.std(dK_adiab[valid]) + 1e-30))
        else:
            lag, rms = float("nan"), float("nan")
        rows.append({
            "r_obs": r_obs,
            "retardation_travel_time": r_obs - R,
            "measured_lag": lag,
            "ctc_state_disagreement_fraction": float(disagree),
            "rms_diff_over_adiab_std": rms,
        })

    return {
        "a0": a0, "eps": eps, "c0": c0,
        "da_amplitude": float(np.max(np.abs(da_series))),
        "t": t, "da_t": da_t,
        "rows": rows,
        "rec": res["rec"], "r_obs_list": list(r_obs_list),
        "L_exact": L_exact_series, "L_adiab": L_adiab_series,
        "energy_drift": float(np.max(np.abs(res["energy"] - res["energy"][0]))
                              / (abs(res["energy"][0]) + 1e-30)),
    }


def background_instability_probe(
    a0: float = 1.5, R: float = 1.0, r_out: float = 14.0, n: int = 2800,
    t_max: float = 20.0, v_cap: float = 50.0,
) -> dict:
    """Drive the FULL (with-potential) supercritical background and report whether
    it stays bounded. The reduced frame-dragging potential is tachyonic (V < 0)
    near ergosurfaces, so the driven evolution diverges -- i.e. the supercritical
    CTC background is NOT quasi-statically stable in this reduction. Reported as a
    factual numerical observation, with the truncation caveat in FINDINGS."""
    V, vs, c0 = vanstockum_potential(a0=a0, R=R)
    Vr = V(np.linspace(R + 1e-3, r_out, 4000))
    v_min = float(np.min(Vr[np.isfinite(Vr)]))
    drive = lambda t: 1e-3 * np.sin(2.0 * t)   # small periodic kick
    ddot = lambda t: 2e-3 * np.cos(2.0 * t)
    cw = CylindricalWave(R=R, r_out=r_out, n=n, V=V, drive=drive,
                         drive_dot=ddot, cfl=0.3, v_cap=v_cap)
    res = cw.evolve(t_max=t_max)
    E = res["energy"]
    drift = float(np.max(np.abs(E - E[0])) / (abs(E[0]) + 1e-30))
    return {
        "a0": a0, "most_negative_V": v_min,
        "tachyonic_rate_estimate": float(np.sqrt(abs(min(v_min, 0.0)))),
        "energy_drift": drift,
        "diverged": bool(drift > 1e3 or not np.isfinite(res["u"]).all()),
    }
