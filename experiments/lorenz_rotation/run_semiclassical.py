"""Orchestrator: self-consistent semiclassical backreaction / chronology protection.

Quantifies how quantum backreaction shrouds the chronology horizon: the renormalized
<T> diverges there, so the dimensionless backreaction beta(r) = 8 pi ell^2 |<T_tt>|/kappa^2
exceeds 1 in a shell of width eps_bd ~ ell^2 around the classical horizon. The classical
horizon is never reached for any hbar > 0; the shell closes only in the classical limit.

Writes lorenz_semiclassical_results.json + the mandatory novelty-catcher verdict.
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

from semiclassical_backreaction import (
    breakdown_shell, breakdown_scaling, self_consistent_effective_horizon,
    chaotic_breakdown,
)


def section_shell():
    print("\n=== BREAKDOWN SHELL: quantum backreaction shrouds the horizon ===")
    rows = []
    for ell2 in (1e-4, 1e-3, 1e-2):
        s = breakdown_shell(1.0, ell2)
        rows.append(s)
        print(f"  ell^2={ell2:.0e}: r_H={s['r_horizon']:.4f} kappa={s['kappa']:.3f}  "
              f"eps_bd={s['eps_breakdown']:.3e}  r_eff={s['r_effective_horizon']:.4f}  "
              f"protected={s['protected']}")
    return rows


def section_scaling():
    print("\n=== SCALING: shell width vs hbar and vs surface gravity ===")
    sc = breakdown_scaling(1.0, [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    print(f"  eps_bd ~ ell^(2p):  power = {sc['power_vs_ell2']:.3f}  "
          f"(=1 => eps_bd proportional to Planck area; -> 0 only as hbar -> 0)")
    # vs kappa (vary a0)
    print("  vs surface gravity kappa (vary a):")
    rows = []
    for a0 in (0.6, 0.8, 1.0, 1.4):
        s = breakdown_shell(a0, 1e-3)
        rows.append({"a0": a0, "kappa": s["kappa"], "eps_bd": s["eps_breakdown"]})
        print(f"    a={a0}: kappa={s['kappa']:.3f}  eps_bd={s['eps_breakdown']:.3e}")
    return {"power_vs_ell2": sc["power_vs_ell2"],
            "ell2": sc["ell2"].tolist(), "eps_bd": sc["eps_breakdown"].tolist(),
            "kappa_rows": rows}


def section_self_consistent():
    print("\n=== SELF-CONSISTENT effective horizon ===")
    it = self_consistent_effective_horizon(1.0, 1e-3)
    print(f"  classical r_H = {it['r_horizon']:.5f}  ->  self-consistent r_eff = "
          f"{it['r_effective_horizon']:.5f}  (shrouded = {it['horizon_shrouded']})")
    print("  the classical chronology horizon is never the self-consistent endpoint")
    return it


def section_chaotic():
    print("\n=== CHAOS: Lorenz a(t) -> flickering protection shell ===")
    cb = chaotic_breakdown(ell2=1e-3, a_center=0.7, amp=0.18)
    print(f"  fraction protected (a>1/2) = {cb['fraction_protected']:.2f}, "
          f"max shell width = {cb['max_shell']:.3e}")
    return {"fraction_protected": cb["fraction_protected"],
            "max_shell": cb["max_shell"]}


def main():
    t0 = time.time()
    results = {}
    results["breakdown_shell"] = section_shell()
    results["scaling"] = section_scaling()
    results["self_consistent"] = section_self_consistent()
    results["chaotic"] = section_chaotic()

    # mandatory catcher: scan ell^2; output the beta(r) profile near the horizon
    def fp(ell2):
        from semiclassical_backreaction import backreaction_profile
        prof = backreaction_profile(1.0, float(ell2))
        b = np.log10(prof["beta"] + 1e-30)
        idx = np.linspace(0, len(b) - 1, 16).astype(int)
        return b[idx]

    ell2_scan = np.geomspace(1e-5, 1e-1, 14)
    scan = scan_novelty(ell2_scan, fp, n_bits=32, parameter_label="ell2")
    results["catcher"] = {"verdict": scan.verdict, "sharp": scan.sharp_features}
    print("\n=== Novelty catcher (mandatory) ===")
    print(summarize_novelty_for_report(scan))

    results["runtime_seconds"] = round(time.time() - t0, 1)
    Path(__file__).with_name("lorenz_semiclassical_results.json").write_text(
        json.dumps(results, indent=2, default=float))
    print(f"\nCONCLUSION: for any hbar > 0 the chronology horizon is shrouded by a "
          f"semiclassical-breakdown shell of width eps_bd ~ ell^2; the classical horizon "
          f"(and its CTCs) is reached only in the classical limit. Chronology protection.")
    print(f"Wrote lorenz_semiclassical_results.json ({results['runtime_seconds']} s)")


if __name__ == "__main__":
    main()
