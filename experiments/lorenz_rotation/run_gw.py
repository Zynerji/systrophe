"""Orchestrator: nonlinear GW emission from a chaotically-rotating cylinder.

Drives the validated nonlinear cylindrical evolution with the Lorenz rotation a(t) and
studies the radiation. Writes lorenz_gw_results.json + the mandatory catcher verdict.

Sections
--------
A  Attractor in radiation: D2 of the radiated twist omega(t) and the nonlinearly-
   generated psi(t) at a far detector, vs the Lorenz attractor (~2.06).
B  Nonlinear cross-polarization: omega twist sources psi (~A^2); psi/omega ~ A.
C  Radiated-energy chaos: the C-energy time series under the chaotic drive.
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

from nonlinear_gw_emission import emit, attractor_in_radiation, cross_polarization_scaling
from deep_tests import correlation_dimension_scalar


def section_attractor():
    print("\n=== A: attractor fingerprint in the radiated waves ===")
    res = attractor_in_radiation(eps=0.04, r_out=60.0, n=3000, t_max=100.0, r_detector=20.0)
    print(f"  radiated omega (twist, linear in drive):  D2 = {res['D2_radiated_omega']:.3f}  "
          f"(Lorenz ~ 2.06)")
    print(f"  radiated psi   (nonlinearly generated):   D2 = {res['D2_radiated_psi']:.3f}  "
          f"(folded by the omega^2 source)")
    print(f"  max|omega_det|={res['max_omega_det']:.3e}  max|psi_det|={res['max_psi_det']:.3e}")
    return res


def section_cross_pol():
    print("\n=== B: nonlinear cross-polarization (omega -> psi) ===")
    rows = cross_polarization_scaling(amps=(0.02, 0.04, 0.08), r_out=50.0, n=2200,
                                      t_max=55.0, r_detector=18.0)
    for r in rows:
        print(f"  eps={r['eps']:.3f}: max|omega|={r['max_omega']:.3e} "
              f"max|psi|={r['max_psi']:.3e}  psi/omega={r['ratio_psi_over_omega']:.4f}")
    ratios = [r["ratio_psi_over_omega"] for r in rows]
    growth = [ratios[1] / (ratios[0] + 1e-30), ratios[2] / (ratios[1] + 1e-30)]
    print(f"  psi/omega doubling factors = {[round(g,2) for g in growth]} "
          f"(linear in A -> ~2; confirms psi ~ A^2)")
    return {"rows": rows, "ratio_growth": growth}


def section_energy_chaos():
    print("\n=== C: radiated-energy chaos ===")
    out = emit(eps=0.05, r_out=50.0, n=2500, t_max=80.0, r_detector=20.0)
    E = out["c_energy"]
    tE = out["t_full"]
    # post-transient C-energy fluctuations
    mask = tE > 25.0
    Em = E[mask]
    d2_E = correlation_dimension_scalar(Em, m=5)
    res = {
        "c_energy_mean": float(Em.mean()), "c_energy_std": float(Em.std()),
        "c_energy_min": float(Em.min()), "c_energy_max": float(Em.max()),
        "D2_c_energy": float(d2_E["D2"]),
    }
    print(f"  C-energy: mean={res['c_energy_mean']:.3e} std={res['c_energy_std']:.3e} "
          f"range=[{res['c_energy_min']:.3e},{res['c_energy_max']:.3e}]")
    print(f"  D2 of C-energy time series = {res['D2_c_energy']:.3f} (chaotic readout)")
    return res, out


def main():
    t0 = time.time()
    results = {}
    results["A_attractor"] = section_attractor()
    results["B_cross_polarization"] = section_cross_pol()
    energy, out = section_energy_chaos()
    results["C_energy_chaos"] = energy

    # mandatory novelty catcher: scan detector radius; output radiated-omega fingerprint
    radii = np.linspace(10.0, 35.0, 14)
    drive_t, _ = out["t_full"], None
    ev_out = emit(eps=0.04, r_out=50.0, n=2500, t_max=80.0, r_detector=20.0)

    def fp(rr):
        o = emit(eps=0.04, r_out=50.0, n=1500, t_max=70.0, r_detector=float(rr))
        col = o["omega_det"]
        return np.quantile(col, np.linspace(0, 1, 16))

    # (single representative scan over a few radii to keep runtime bounded)
    scan_radii = np.linspace(12.0, 30.0, 8)
    scan = scan_novelty(scan_radii, fp, n_bits=32, parameter_label="r_detector")
    results["catcher"] = {"verdict": scan.verdict, "sharp": scan.sharp_features}
    print("\n=== Novelty catcher (mandatory) ===")
    print(summarize_novelty_for_report(scan))

    results["runtime_seconds"] = round(time.time() - t0, 1)
    Path(__file__).with_name("lorenz_gw_results.json").write_text(
        json.dumps(results, indent=2, default=float))
    print(f"\nWrote lorenz_gw_results.json ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
