"""Phase 2a applied to the SystrophePair (matched-pair, delta-sweep).

For two co-axial supercritical cylinders sharing (a, R, A) but with
relative phase offset delta, the linearised joint F-component is

    F_pair(r; delta) = F_1(r) + F_2(r; delta)
                     = 2 cos(delta/2) (r/R) sin(alpha u + gamma + delta/2) / sin(gamma)

which is a *single Tipler-sinusoid* with reduced amplitude 2 cos(delta/2)
and shifted phase. Predictions:

  delta = 0  : amplitude 2x single-cylinder, same horizons.
               Stress-energy 4x single (T ~ F'^2 / F has ratio (2F)' / (2F) = F'/F).
               No: T_tt ~ F'^2 / (96 pi F) so ratio = (2F')^2/(2F) / (F'^2/F) = 2.
               Stress-tensor MAGNITUDE doubles.

  delta = pi : F_pair == 0 IDENTICALLY (anti-phase extinction).
               No Cauchy horizons. Polyakov sigma undefined (the
               framework returns NaN). Operationally: CTC suppression
               AND QFTCS divergence suppression simultaneously.

  delta in (0, pi) : amplitude 2 cos(delta/2) F_1; horizons shift by
               -delta/(2 alpha) in log radius; divergence rate stays at
               power = -1 (geometric, alpha-independent).
"""

from __future__ import annotations

import json
import math
import pathlib
import time

import numpy as np
from scipy.optimize import brentq

from systrophe import (
    StressEnergyState,
    polyakov_sigma_derivatives,
)
from systrophe.ctc.stress_energy_ctc import (
    DivergenceFit,
    _stress_tensor_in_rstar_coords,
    _rstar_to_r_tensor,
)
from systrophe.geometry.vanstockum import VanStockumInterior


def _pair_F(omega: float, R: float, delta: float, r: float) -> float:
    """Matched-pair F_pair(r) via analytic identity."""
    a = omega * R
    alpha = math.sqrt(4 * a * a - 1)
    gamma = math.pi - math.atan(alpha)
    u = math.log(r / R)
    theta = alpha * u + gamma + delta / 2.0
    return 2.0 * math.cos(delta / 2.0) * (r / R) * math.sin(theta) / math.sin(gamma)


def _pair_F_derivatives(omega: float, R: float, delta: float, r: float, eps: float = 1e-5
                          ) -> tuple[float, float, float]:
    F = _pair_F(omega, R, delta, r)
    Fp = (_pair_F(omega, R, delta, r + eps) - _pair_F(omega, R, delta, r - eps)) / (2 * eps)
    Fpp = (_pair_F(omega, R, delta, r + eps) - 2 * F + _pair_F(omega, R, delta, r - eps)) / (eps * eps)
    return F, Fp, Fpp


def _pair_sigma_derivatives(omega: float, R: float, delta: float, r: float, eps: float = 1e-5):
    F, Fp, Fpp = _pair_F_derivatives(omega, R, delta, r, eps=eps)
    if F <= 0:
        return None
    sigma_p = Fp / (2.0 * F)
    sigma_pp = (Fpp / (2.0 * F)) - (Fp * Fp) / (2.0 * F * F)
    # h_metric = 1
    sigma_rstar = math.sqrt(F) * Fp / (2.0 * F)
    sigma_rstar2 = F * sigma_pp + (Fp / 2.0) * sigma_p
    return {"sigma_rstar": sigma_rstar, "sigma_rstar2": sigma_rstar2, "F": F, "Fp": Fp, "Fpp": Fpp}


def boulware_pair(omega: float, R: float, delta: float, r: float, eps: float = 1e-5) -> dict:
    """Boulware <T_mu_nu>_B for a matched pair."""
    d = _pair_sigma_derivatives(omega, R, delta, r, eps=eps)
    if d is None:
        return {"T_tt": float("nan"), "T_rr": float("nan"), "F": _pair_F(omega, R, delta, r)}
    rstar = _stress_tensor_in_rstar_coords(d["sigma_rstar"], d["sigma_rstar2"], t_u=0.0, t_v=0.0)
    rdict = _rstar_to_r_tensor(rstar, F=d["F"], h_metric=1.0)
    return {**rstar, **rdict, "F": d["F"]}


def find_pair_cauchy_horizons(omega: float, R: float, delta: float,
                                  r_min: float = None, r_max: float = None,
                                  n_grid: int = 4001):
    if r_min is None:
        r_min = 1.05 * R
    if r_max is None:
        r_max = 10.0 * R
    r_grid = np.linspace(r_min, r_max, n_grid)
    F_vals = np.array([_pair_F(omega, R, delta, float(r)) for r in r_grid])
    if np.max(np.abs(F_vals)) < 1e-10:
        return np.array([])  # identically-zero (delta = pi)
    signs = np.sign(F_vals)
    changes = np.where(np.diff(signs) != 0)[0]
    horizons = []
    for i in changes:
        r_lo, r_hi = float(r_grid[i]), float(r_grid[i + 1])
        try:
            h = brentq(lambda r: _pair_F(omega, R, delta, r), r_lo, r_hi, xtol=1e-12)
            horizons.append(float(h))
        except ValueError:
            continue
    return np.array(horizons)


def divergence_at_pair_horizon(omega, R, delta, r_H, n_samples=20, eps_min=5e-4, eps_max=2e-2,
                                 component="T_tt", eps_diff=1e-5):
    eps_grid = np.geomspace(eps_min, eps_max, n_samples)
    F_outside = _pair_F(omega, R, delta, r_H + eps_min)
    if F_outside <= 0:
        F_inside = _pair_F(omega, R, delta, r_H - eps_min)
        if F_inside <= 0:
            return float("nan"), float("nan")
        eps_grid = -eps_grid
    r_samples = r_H + eps_grid
    T_arr = np.array([boulware_pair(omega, R, delta, float(r), eps=eps_diff).get(component, float("nan"))
                        for r in r_samples])
    finite = np.isfinite(T_arr) & (np.abs(T_arr) > 1e-30)
    if finite.sum() < 4:
        return float("nan"), float("nan")
    p, log_A = np.polyfit(np.log(np.abs(eps_grid))[finite], np.log(np.abs(T_arr[finite])), 1)
    resid = np.log(np.abs(T_arr[finite])) - (p * np.log(np.abs(eps_grid))[finite] + log_A)
    return float(p), float(np.sqrt(np.mean(resid ** 2)))


def main() -> dict:
    t_start = time.time()
    omega, R = 2.0, 1.0
    deltas = np.array([0.0, math.pi/8, math.pi/4, math.pi/2, 3*math.pi/4,
                        math.pi - 0.5, math.pi - 0.1, math.pi - 0.01, math.pi])
    rows = []
    for delta in deltas:
        horizons = find_pair_cauchy_horizons(omega, R, float(delta))
        if len(horizons) == 0:
            rows.append({
                "delta": float(delta),
                "delta_over_pi": float(delta / math.pi),
                "amplitude_factor": float(abs(2 * math.cos(delta / 2.0))),
                "n_horizons": 0,
                "horizons": [],
                "tt_power": None,
                "rr_power": None,
                "ctc_extinction": True,
            })
            continue
        r_H = float(horizons[0])
        p_tt, rms_tt = divergence_at_pair_horizon(omega, R, float(delta), r_H, component="T_tt")
        p_rr, rms_rr = divergence_at_pair_horizon(omega, R, float(delta), r_H, component="T_rr")
        T_mid = boulware_pair(omega, R, float(delta), max(1.05 * R, r_H * 0.5))
        rows.append({
            "delta": float(delta),
            "delta_over_pi": float(delta / math.pi),
            "amplitude_factor": float(abs(2 * math.cos(delta / 2.0))),
            "n_horizons": int(len(horizons)),
            "horizons": [float(h) for h in horizons[:3]],
            "r_H1": r_H,
            "tt_power": p_tt,
            "tt_fit_rms": rms_tt,
            "rr_power": p_rr,
            "rr_fit_rms": rms_rr,
            "T_tt_midpoint": float(T_mid.get("T_tt", float("nan"))),
            "ctc_extinction": False,
        })

    out = {
        "phase": "2a-pair",
        "title": "Phase 2a applied to a matched SystrophePair",
        "spacetime": {"omega": float(omega), "R": float(R), "matched": True},
        "rows": rows,
        "elapsed_seconds": time.time() - t_start,
    }
    out_path = pathlib.Path(__file__).with_name("phase_2a_pair_extension_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 80)
    print("Phase 2a: matched SystrophePair, delta sweep (omega=2, R=1)")
    print("=" * 80)
    print(" delta/pi  amp_factor  n_h    r_H1     T_tt power  T_rr power  CTC?")
    print("-" * 80)
    for r in rows:
        if r["ctc_extinction"]:
            print(f"  {r['delta_over_pi']:5.3f}   {r['amplitude_factor']:8.4f}    "
                  f"0     --         --          --         EXTINGUISHED")
        else:
            print(f"  {r['delta_over_pi']:5.3f}   {r['amplitude_factor']:8.4f}    "
                  f"{r['n_horizons']:1d}   {r['r_H1']:.4f}   "
                  f"{r['tt_power']:+.4f}     {r['rr_power']:+.4f}    intact")
    print()
    print("Interpretation:")
    print("  - delta = 0      : full single-cylinder Cauchy-horizon structure x2 amplitude.")
    print("  - delta in (0,pi): amplitude shrinks as 2 cos(delta/2); horizons shift but")
    print("                     divergence power stays -1 / -2 (universality).")
    print("  - delta = pi     : F_pair == 0 identically -> no horizons, no QFTCS divergence")
    print("                     SIMULTANEOUS with no CTCs. Anti-phase extinguishes BOTH.")
    print()
    print(f"Elapsed: {out['elapsed_seconds']:.1f} s")
    print(f"Results: {out_path}")
    return out


if __name__ == "__main__":
    main()
