"""Cheap rigorous recheck of A003002 (largest 3-AP-free subset of [1,n],
Erdos problems 3 and 142) -- is the sweep's omega ~ 5.95 peak a real
complex dimension or red-noise?

High n_boot AR(1) significance + detrend-degree sensitivity. A real
log-periodic mode is robust to detrend choice; a trend artifact moves.
Reference frequency: the GREEDY 3-AP-free set (base-3, no digit 2; Cantor
set) has exact discrete-scale-invariance at omega = 2 pi / ln 3 = 5.7196.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from erdos_dsi_sweep import dsi_scan, _detrend_logpower

CANTOR_OMEGA = 2 * math.pi / math.log(3)  # 5.7196: greedy/base-3 AP-free DSI


def load_a003002():
    txt = (Path(__file__).parent / "oeis_cache" / "A003002.txt").read_text()
    n, a = [], []
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        i, v = line.split()[:2]
        i, v = int(i), int(v)
        if i >= 1 and v > 0:
            n.append(i)
            a.append(v)
    return np.array(n, float), np.array(a, float)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    n, a = load_a003002()
    t = np.log(n)
    logy = np.log(a)
    print("=" * 70)
    print("A003002 recheck (largest 3-AP-free subset of [1,n])")
    print("=" * 70)
    print(f"  terms n>=1: {len(n)},  n range [{int(n[0])},{int(n[-1])}],  "
          f"ln n span = {t[-1]-t[0]:.2f}")
    print(f"  reference: greedy/base-3 Cantor DSI omega = "
          f"2pi/ln3 = {CANTOR_OMEGA:.4f}")
    print("  " + "-" * 64)
    print(f"  {'detrend deg':>11}  {'peak_omega':>10}  {'ratio':>6}  "
          f"{'power':>6}  {'p_value':>9}")
    rows = []
    for deg in (2, 3, 4):
        y = _detrend_logpower(t, logy, deg=deg)
        scan = dsi_scan(t, y, omega=(2.0, 30.0), n_freq=2000, n_boot=5000,
                        seed=deg)
        rows.append({"detrend_deg": deg, **scan})
        print(f"  {deg:>11}  {scan['peak_omega']:>10.3f}  "
              f"{scan['geometric_ratio']:>6.3f}  {scan['peak_power']:>6.3f}  "
              f"{scan['p_value']:>9.4f}")
    print()
    p_best = min(r["p_value"] for r in rows)
    print(f"  best p across detrend choices: {p_best:.4f}  "
          f"(Bonferroni-in-sweep threshold was 0.00065; uncorrected 0.05)")
    verdict = ("ROBUST signal" if p_best <= 0.01 and
               len({round(r['peak_omega']) for r in rows}) == 1
               else "noise / not robust")
    print(f"  verdict: {verdict}")
    out = {"n_terms": len(n), "ln_n_span": float(t[-1] - t[0]),
           "cantor_omega": CANTOR_OMEGA, "rows": rows,
           "best_p": float(p_best), "verdict": verdict}
    p = Path(__file__).parent / "erdos_a003002_recheck_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Wrote {p}")


if __name__ == "__main__":
    main()
