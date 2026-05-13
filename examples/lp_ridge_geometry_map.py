"""Map every LP-ridge catcher peak to a named geometric feature of the
bare LP supercritical exterior metric (F, K, L from VanStockumInterior).

This is the punchline of the multi-observable address-space novelty
program: the catcher localizes the metric's special-locus structure
from physics observables alone, without ever reading the metric.

Result (omega=1.0, R=1.0 LP):
  r = 1.8305:  F = 0  AND  dK/dr = 0  (chronology horizon, K extremum)
               --> Catcher peak r=1.840 from 11 modules
  r = 2.2903:  F*L - K^2 = 0           (t-phi sector causal sign change)
               --> Catcher peak r=2.20 from 4 modules
  r = 2.4766:  d^2 F / dr^2 = 0        (F inflection)
               --> Catcher peak r=2.500 from 5 modules
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from systrophe.vanstockum import VanStockumInterior


def find_zeros_dense(arr: np.ndarray, r_grid: np.ndarray) -> list[float]:
    out = []
    for i in range(len(arr) - 1):
        if arr[i] * arr[i + 1] < 0 and np.isfinite(arr[i]) and np.isfinite(arr[i + 1]):
            rz = r_grid[i] - arr[i] * (r_grid[i + 1] - r_grid[i]) / (arr[i + 1] - arr[i])
            out.append(float(rz))
    return out


def main():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        raise SystemExit("VS subcritical; LP exterior horizons do not exist")

    r_grid = np.linspace(1.05, 6.0, 4951)  # dr = 0.001
    F = vs.analytic_exterior_F(r_grid)
    K = vs.analytic_exterior_K(r_grid)
    L = vs.analytic_exterior_L(r_grid)
    dF = np.gradient(F, r_grid)
    dK = np.gradient(K, r_grid)
    dL = np.gradient(L, r_grid)
    d2F = np.gradient(dF, r_grid)
    d2K = np.gradient(dK, r_grid)
    d2L = np.gradient(dL, r_grid)
    disc = F * L - K**2  # null Killing condition

    features = {
        "F = 0 (chronology horizon)": find_zeros_dense(F, r_grid),
        "K = 0": find_zeros_dense(K, r_grid),
        "L = 0": find_zeros_dense(L, r_grid),
        "dF/dr = 0 (F extremum)": find_zeros_dense(dF, r_grid),
        "dK/dr = 0 (K extremum)": find_zeros_dense(dK, r_grid),
        "dL/dr = 0 (L extremum)": find_zeros_dense(dL, r_grid),
        "d2F/dr2 = 0 (F inflection)": find_zeros_dense(d2F, r_grid),
        "d2K/dr2 = 0 (K inflection)": find_zeros_dense(d2K, r_grid),
        "d2L/dr2 = 0 (L inflection)": find_zeros_dense(d2L, r_grid),
        "F*L - K^2 = 0 (sectoral flip)": find_zeros_dense(disc, r_grid),
    }
    print("Geometric special loci of bare LP supercritical exterior")
    print("=" * 70)
    for name, locs in features.items():
        if not locs: continue
        print(f"  {name:40s}: {[round(x, 4) for x in locs]}")

    # Now map each catcher peak to nearest geometric locus
    catcher_peaks = [1.620, 1.840, 1.860, 2.000, 2.200, 2.500]
    print()
    print("Catcher-peak --> Nearest LP-metric special locus")
    print("=" * 70)
    mapping = {}
    all_special = []
    for name, locs in features.items():
        for r in locs:
            if 1.4 <= r <= 2.7:
                all_special.append((r, name))
    for peak in catcher_peaks:
        best = min(all_special, key=lambda rn: abs(rn[0] - peak))
        gap = abs(best[0] - peak)
        mapping[peak] = {"locus_r": best[0], "feature": best[1],
                         "gap": float(gap),
                         "match": "EXACT" if gap < 0.02 else "near" if gap < 0.05 else "unclear"}
        print(f"  r={peak:.3f}  -->  {best[1]:40s}  at r={best[0]:.4f}  "
              f"(gap={gap:.4f}, {mapping[peak]['match']})")

    out_path = Path(__file__).parent / "lp_ridge_geometry_map.json"
    out_path.write_text(json.dumps({"features": features,
                                     "catcher_peak_mapping": mapping},
                                    indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
