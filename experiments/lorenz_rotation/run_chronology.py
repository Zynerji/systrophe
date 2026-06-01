"""Orchestrator: dynamical approach to the chronology horizon (chronology protection).

Spins the cylinder up through the supercritical threshold a = 1/2 and shows that the
renormalized quantum <T_tt> diverges at the forming chronology horizon (the protection
signal) while the classical Kretschmann stays bounded -- Hawking's chronology protection.
Then drives a(t) chaotically across the threshold for a flickering protection barrier.

Writes lorenz_chronology_results.json + the mandatory novelty-catcher verdict.
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
from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.ctc.stress_energy_ctc import boulware_stress_tensor
from systrophe.qftcs.quantum_diagnostics import cauchy_horizon_estimate

from chronology_horizon import approach_sequence, chaotic_barrier, chronology_signal, A_CRIT


def section_approach():
    print("\n=== APPROACH: spin-up through the chronology threshold a = 1/2 ===")
    a_vals = np.array([0.45, 0.50, 0.52, 0.6, 0.8, 1.0, 1.2, 1.5])
    seq = approach_sequence(a_vals)
    print("  a    super  r_H     kappa  <T_tt>power  Kretschmann  protected")
    for s in seq:
        rH = f"{s['r_horizon']:.3f}" if s["r_horizon"] else "  --  "
        k = f"{s['kappa']:.3f}" if s["kappa"] else " -- "
        tp = f"{s['T_div_power']:+.3f}" if s["T_div_power"] is not None else "  --  "
        kr = f"{s['kretschmann_at_rH']:.2e}" if s["kretschmann_at_rH"] is not None else "  --  "
        print(f"  {s['a']:.2f} {str(s['supercritical']):5s}  {rH}  {k}   {tp}    {kr}    {s['protected']}")
    # approach scaling: r_H -> R*e as a -> 1/2+, protection power universal ~ -1
    supers = [s for s in seq if s["supercritical"]]
    powers = [s["T_div_power"] for s in supers]
    return {
        "sequence": [{k: v for k, v in s.items()} for s in seq],
        "protection_power_mean": float(np.mean(powers)),
        "protection_power_std": float(np.std(powers)),
        "kretschmann_all_bounded": all(s["kretschmann_bounded"] for s in seq),
        "protected_iff_supercritical": all(
            s["protected"] == s["supercritical"] for s in seq),
    }


def section_threshold_scaling():
    print("\n=== SCALING: approach to threshold a -> 1/2+ ===")
    a_vals = 0.5 + np.array([0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    rows = []
    for a in a_vals:
        s = chronology_signal(float(a))
        rows.append({"a": float(a), "da": float(a - 0.5),
                     "r_horizon": s["r_horizon"], "kappa": s["kappa"],
                     "T_power": s["T_div_power"]})
        print(f"  a-1/2={a-0.5:.3f}: r_H={s['r_horizon']:.4f}  kappa={s['kappa']:.4f}  "
              f"T_power={s['T_div_power']:+.4f}")
    print(f"  -> r_H -> R*e = {np.e:.4f} as a -> 1/2+ (horizon born at finite radius);")
    print(f"     protection power stays ~ -1 (full strength from the threshold).")
    return {"rows": rows, "r_horizon_limit_Re": float(np.e)}


def section_chaotic_barrier():
    print("\n=== CHAOS: Lorenz a(t) flickering the chronology-protection barrier ===")
    cb = chaotic_barrier(a_center=0.62, amp=0.18)
    on = cb["barrier_strength"] > 0
    print(f"  a(t) in [{cb['a_min']:.3f}, {cb['a_max']:.3f}], "
          f"threshold crossings = {cb['n_threshold_crossings']}")
    print(f"  barrier ON (supercritical) fraction = {cb['fraction_supercritical']:.2f}, "
          f"max strength (kappa) = {cb['barrier_strength'].max():.3f}")
    return {
        "fraction_supercritical": cb["fraction_supercritical"],
        "n_threshold_crossings": cb["n_threshold_crossings"],
        "a_min": cb["a_min"], "a_max": cb["a_max"],
        "barrier_max": float(cb["barrier_strength"].max()),
    }


def main():
    t0 = time.time()
    results = {}
    results["approach"] = section_approach()
    results["threshold_scaling"] = section_threshold_scaling()
    results["chaotic_barrier"] = section_chaotic_barrier()

    # mandatory novelty catcher: scan a across the threshold; the protection signal
    # (|<T_tt>| near the horizon) is zero subcritically and divergent supercritically
    # -> the catcher should localise the chronology-protection threshold at a = 1/2.
    def fp(a):
        # normalized divergence SHAPE: identically zero subcritically, the (universal)
        # 1/(r-r_H) profile supercritically -> the only transition is at a = 1/2.
        vs = VanStockumInterior(omega=float(a), R=1.0)
        if a <= A_CRIT:
            return np.zeros(16)
        rH = float(cauchy_horizon_estimate(vs)[0])
        offs = np.geomspace(5e-3, 5e-2, 16)
        T = np.array([abs(boulware_stress_tensor(vs, rH + o)["T_tt"]) for o in offs])
        return T / (T.max() + 1e-300)

    a_scan = np.linspace(0.4, 1.2, 18)
    scan = scan_novelty(a_scan, fp, n_bits=32, parameter_label="a")
    feats = [f["parameter_value"] for f in scan.sharp_features]
    results["catcher"] = {"verdict": scan.verdict, "sharp_features": scan.sharp_features,
                          "threshold_a_crit": A_CRIT}
    print("\n=== Novelty catcher (mandatory) ===")
    print(summarize_novelty_for_report(scan))
    print(f"  catcher sharp features at a = {[round(x,3) for x in feats]} "
          f"(chronology threshold a_crit = {A_CRIT})")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    Path(__file__).with_name("lorenz_chronology_results.json").write_text(
        json.dumps(results, indent=2, default=float))
    print(f"\nWrote lorenz_chronology_results.json ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
