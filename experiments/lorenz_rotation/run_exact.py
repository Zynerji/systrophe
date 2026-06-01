"""Orchestrator: EXACT (hyperbolic) time-varying metric vs the adiabatic CTC treatment.

Replaces the quasi-static (infinite-speed) CTC treatment with the exact hyperbolic
evolution of the frame-dragging sector (cylindrical_wave.py), and quantifies what
adiabatic gets wrong. Writes lorenz_exact_results.json + the mandatory novelty-catcher
verdict.

Sections
--------
V1  Causal propagation at speed 1 (a rotation pulse reaches r at retarded time r-R).
V2  Energy conservation (undriven) -> well-posed scheme.
V3  Adiabaticity criterion made rigorous: lag = travel time; adiabatic valid iff
    Omega (r-R) << 1.
CTC Exact-retarded vs adiabatic CTC state under a chaotic a(t): retardation makes
    adiabatic mispredict the time-machine open/closed state near band edges.
INST The full supercritical background is NOT quasi-statically stable (tachyonic
    frame-dragging potential at ergosurfaces).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from systrophe.catchers.novelty_catcher import scan_novelty, summarize_novelty_for_report

from cylindrical_wave import CylindricalWave
from exact_ctc import exact_vs_adiabatic_ctc, background_instability_probe


def v1_causal_speed():
    print("\n=== V1: causal propagation at speed 1 (retardation) ===")
    t0, w = 3.0, 0.6
    drive = lambda t: np.exp(-((t - t0) / w) ** 2)
    ddot = lambda t: -2 * (t - t0) / w ** 2 * np.exp(-((t - t0) / w) ** 2)
    cw = CylindricalWave(R=1.0, r_out=21.0, n=2000, V=None, drive=drive,
                         drive_dot=ddot, cfl=0.4)
    radii = [2.0, 5.0, 9.0, 13.0]
    res = cw.evolve(t_max=18.0, record_radii=radii)
    t = res["t"]
    rows = []
    for j, rr in enumerate(radii):
        k = int(np.argmax(np.abs(res["rec"][:, j])))
        rows.append({"r": rr, "t_arrival": float(t[k]),
                     "predicted": t0 + (rr - 1.0)})
        print(f"  r={rr:5.1f}: arrival t={t[k]:6.2f}  predicted={t0 + rr - 1.0:6.2f}")
    return rows


def v2_energy_conservation():
    print("\n=== V2: energy conservation (undriven) ===")
    cw = CylindricalWave(R=1.0, r_out=41.0, n=4000, V=None,
                         drive=lambda t: 0.0, drive_dot=lambda t: 0.0, cfl=0.4)
    r = cw.r
    u0 = np.exp(-((r - 20.0) / 1.0) ** 2)
    res = cw.evolve(t_max=10.0, u0=u0, pi0=np.zeros_like(r))
    E = res["energy"]
    drift = float(np.max(np.abs(E - E[0])) / E[0])
    print(f"  energy rel. drift = {drift:.2e}")
    return {"energy_rel_drift": drift}


def v3_adiabaticity():
    print("\n=== V3: adiabaticity criterion (lag = travel time) ===")
    r_star, A = 6.0, 1e-3
    rows = []
    for Om in [0.1, 0.3, 0.6]:
        drive = lambda t, Om=Om: A * np.sin(Om * t)
        ddot = lambda t, Om=Om: A * Om * np.cos(Om * t)
        cw = CylindricalWave(R=1.0, r_out=40.0, n=3000, V=None, drive=drive,
                             drive_dot=ddot, cfl=0.4)
        res = cw.evolve(t_max=80.0, record_radii=[r_star])
        t, sig = res["t"], res["rec"][:, 0]
        mask = t > 40.0
        tt, ss = t[mask], sig[mask]
        dd = A * np.sin(Om * tt)
        ss = ss - ss.mean(); dd = dd - dd.mean()
        corr = np.correlate(ss, dd, mode="full")
        lags = np.arange(-len(tt) + 1, len(tt)) * (tt[1] - tt[0])
        lag = float(abs(lags[np.argmax(corr)]))
        rows.append({"Omega": Om, "period": 2 * np.pi / Om, "lag": lag,
                     "adiabatic_phase_error": Om * lag})
        print(f"  Omega={Om:.2f} period={2*np.pi/Om:5.1f}: lag={lag:5.2f} "
              f"(travel {r_star-1.0:.0f})  adiab phase err={Om*lag:.2f} rad")
    return {"r_star_minus_R": r_star - 1.0, "rows": rows}


def ctc_exact_vs_adiabatic():
    print("\n=== CTC: exact-retarded vs adiabatic time-machine state ===")
    radii = (2.0, 4.0, 7.0, 10.0)
    res = exact_vs_adiabatic_ctc(a0=1.5, R=1.0, eps=0.05, r_out=30.0, n=2500,
                                 t_max=70.0, lorenz_t_max=90.0,
                                 r_obs_list=radii, use_potential=False)
    for row in res["rows"]:
        print(f"  r_obs={row['r_obs']:5.1f} travel={row['retardation_travel_time']:4.1f} "
              f"lag={row['measured_lag']:5.2f}  CTC-state disagreement="
              f"{100*row['ctc_state_disagreement_fraction']:4.1f}%")
    # mandatory novelty catcher: scan observation radius; output = L_exact histogram
    t = res["t"]
    L_exact = res["L_exact"]

    def fingerprint(r_obs):
        j = int(np.argmin(np.abs(np.array(radii) - r_obs)))
        col = L_exact[:, j]
        return np.quantile(col, np.linspace(0.0, 1.0, 16))

    scan = scan_novelty(np.array(radii, dtype=float), fingerprint, n_bits=32,
                        parameter_label="r_obs")
    print(f"  novelty catcher verdict = {scan.verdict}; "
          f"sharp at r_obs = {[f['parameter_value'] for f in scan.sharp_features]}")
    return {
        "rows": res["rows"], "da_amplitude": res["da_amplitude"],
        "energy_drift": res["energy_drift"],
        "catcher_verdict": scan.verdict,
        "catcher_sharp": scan.sharp_features,
    }, scan


def instability():
    print("\n=== INST: is the supercritical background quasi-statically stable? ===")
    out = background_instability_probe(a0=1.5, r_out=14.0, n=2800, t_max=20.0)
    print(f"  most-negative V = {out['most_negative_V']:.2f} (tachyonic); "
          f"driven energy drift = {out['energy_drift']:.2e}; diverged = {out['diverged']}")
    print("  -> the adiabatic 'frozen flickering background' assumption is "
          "qualitatively invalid here (caveat: single-variable reduction).")
    return out


def main():
    t0 = time.time()
    results = {}
    results["V1_causal_speed"] = v1_causal_speed()
    results["V2_energy_conservation"] = v2_energy_conservation()
    results["V3_adiabaticity"] = v3_adiabaticity()
    ctc_out, scan = ctc_exact_vs_adiabatic()
    results["CTC_exact_vs_adiabatic"] = ctc_out
    results["INST_instability"] = instability()

    print("\n=== Novelty catcher report (mandatory) ===")
    print(summarize_novelty_for_report(scan))

    results["runtime_seconds"] = round(time.time() - t0, 1)
    out_path = Path(__file__).with_name("lorenz_exact_results.json")
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out_path}  ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
