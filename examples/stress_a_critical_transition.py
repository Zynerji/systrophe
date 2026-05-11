"""Stress the supercritical/subcritical transition in the Tipler LP envelope.

Tipler regime parameter a = omega R has three regimes:
  - subcritical (a < 1/2): F, K, L given by hyperbolic forms
  - critical    (a = 1/2): degenerate polynomial forms
  - supercritical (a > 1/2): the canonical log-periodic Bonnor Case III

The transition at a = 1/2 is a non-analytic regime change. We test
the catcher's ability to detect this transition by sweeping a fine
grid across a in [0.3, 0.7].

Each probe samples the L metric component, the Tipler tilt, and a
selection of derived diagnostics. Sharp Hamming transitions in the
catcher's hash of the same-quantity-across-conditions should
appear near a = 1/2.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import (
    catch_novelty_in_named_arrays,
    catch_novelty_per_quantity,
)
from systrophe.tipler_krasnikov_hybrid import tipler_tilt_at
from systrophe.vanstockum import VanStockumInterior

A_GRID = np.linspace(0.3, 0.7, 41)  # 41 points across critical


def main() -> None:
    print("=" * 70)
    print("Stress test: a-critical transition (subcritical -> supercritical)")
    print("=" * 70)
    print(f"a grid: {len(A_GRID)} points over [{A_GRID[0]:.3f}, {A_GRID[-1]:.3f}]")
    print(f"Critical at a = 0.5")
    print()

    r_test = 2.0  # fixed test radius outside R=1
    L_values = []
    F_values = []
    K_values = []
    tilt_values = []
    L_at_r3 = []
    L_at_r5 = []

    for a in A_GRID:
        vs = VanStockumInterior(omega=float(a), R=1.0)
        try:
            L = float(vs.analytic_exterior_L(r_test))
        except Exception:
            L = 0.0
        try:
            F = float(vs.analytic_exterior_F(r_test))
        except Exception:
            F = 0.0
        try:
            K = float(vs.analytic_exterior_K(r_test))
        except Exception:
            K = 0.0
        try:
            tilt = float(tipler_tilt_at(vs, r_test))
        except Exception:
            tilt = 0.0
        L_values.append(L if math.isfinite(L) else 0.0)
        F_values.append(F if math.isfinite(F) else 0.0)
        K_values.append(K if math.isfinite(K) else 0.0)
        tilt_values.append(tilt if math.isfinite(tilt) else 0.0)
        # Also sample L at two other radii to enrich the feature space
        try:
            L_at_r3.append(float(vs.analytic_exterior_L(3.0)))
        except Exception:
            L_at_r3.append(0.0)
        try:
            L_at_r5.append(float(vs.analytic_exterior_L(5.0)))
        except Exception:
            L_at_r5.append(0.0)

    L_arr = np.array(L_values)
    F_arr = np.array(F_values)
    K_arr = np.array(K_values)
    tilt_arr = np.array(tilt_values)
    L3 = np.array(L_at_r3)
    L5 = np.array(L_at_r5)

    print(f"{'a':6s} {'F(r=2)':10s} {'K(r=2)':10s} {'L(r=2)':10s} {'tilt':10s}")
    for i in range(0, len(A_GRID), 4):
        print(f"  {A_GRID[i]:.3f}  {F_arr[i]:+9.4f}  {K_arr[i]:+9.4f}  "
              f"{L_arr[i]:+9.4f}  {tilt_arr[i]:9.4f}")
    print()

    # Split a-grid into subcritical / supercritical halves
    mid = len(A_GRID) // 2
    subcr = slice(0, mid)
    super_ = slice(mid, len(A_GRID))

    per_q = {
        "F_r2": {"sub": F_arr[subcr], "super": F_arr[super_]},
        "K_r2": {"sub": K_arr[subcr], "super": K_arr[super_]},
        "L_r2": {"sub": L_arr[subcr], "super": L_arr[super_]},
        "tilt_r2": {"sub": tilt_arr[subcr], "super": tilt_arr[super_]},
        "L_r3": {"sub": L3[subcr], "super": L3[super_]},
        "L_r5": {"sub": L5[subcr], "super": L5[super_]},
    }
    nov = catch_novelty_per_quantity(per_q)
    print(f"Catcher per-quantity aggregate verdict: {nov['aggregate_verdict']}")
    print(f"Novel quantities: {nov['novel_quantities']}")
    for q, r in nov["per_quantity"].items():
        v = r.get("verdict")
        n = len(r.get("sharp_features", []))
        if v == "novel_structure":
            print(f"  {q}: novel, n_sharp={n}")
            for sf in r["sharp_features"][:3]:
                print(f"    between={sf.get('between')}  step={sf.get('hamming_step')}")
        else:
            print(f"  {q}: {v}")

    # Also test by individual hash: pair (sub, super) for each quantity
    print()
    print("Full series catch_novelty_in_named_arrays:")
    arrs = {
        "F_r2_series": F_arr,
        "K_r2_series": K_arr,
        "L_r2_series": L_arr,
        "tilt_r2_series": tilt_arr,
    }
    nov2 = catch_novelty_in_named_arrays(arrs)
    print(f"  verdict={nov2['verdict']}, n_sharp={len(nov2['sharp_features'])}")
    for sf in nov2["sharp_features"][:5]:
        print(f"  between={sf.get('between')}  step={sf.get('hamming_step')}")

    out_path = Path(__file__).parent / "stress_a_critical_transition_results.json"
    out_path.write_text(json.dumps({
        "a_grid": A_GRID.tolist(),
        "L_at_r2": L_arr.tolist(),
        "K_at_r2": K_arr.tolist(),
        "F_at_r2": F_arr.tolist(),
        "tilt_at_r2": tilt_arr.tolist(),
        "novelty_catcher": nov,
        "novelty_series": nov2,
    }, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
