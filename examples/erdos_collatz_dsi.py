"""Collatz (Erdos problem #1135, $500): DSI / log-periodicity in the total
stopping time.

Erdos offered $500 on the 3x+1 problem ("mathematics is not yet ready").
The total stopping time sigma(n) (steps to reach 1) is computable by us to
large n -- unlike most prize sequences (Ramsey/Golomb), this one is not
length-walled. We test the smoothed sigma for discrete-scale invariance:
geometric-bin-average sigma(n), detrend the C*ln n trend, and run the
validated DSI detector (Lomb-Scargle in ln n + AR(1) red-noise significance).

Honest premise: a clean significant log-periodic peak would be a real
structural lead (Collatz log-periodicity is a contested topic), NOT a proof
or a route to the $500. Candidate special frequencies if 2-adic / 3-adic
self-similarity leaks through:
    2pi/ln2     = 9.064
    2pi/ln(4/3) = 21.843   (4/3 = geometric mean growth per step)
    2pi/ln3     = 5.719
Reference growth: average sigma(n) ~ (2/ln(4/3)) ln n = 6.952 ln n.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from erdos_dsi_sweep import dsi_scan, _detrend_logpower

C_THEORY = 2.0 / math.log(4.0 / 3.0)  # ~6.952
SPECIAL = {"2pi/ln2": 2 * math.pi / math.log(2),
           "2pi/ln(4/3)": 2 * math.pi / math.log(4 / 3),
           "2pi/ln3": 2 * math.pi / math.log(3)}


def total_stopping_times(N: int) -> np.ndarray:
    ts = np.full(N + 1, -1, dtype=np.int64)
    ts[1] = 0
    for n in range(2, N + 1):
        if ts[n] != -1:
            continue
        seq = []
        m = n
        while m != 1 and (m > N or ts[m] == -1):
            seq.append(m)
            m = (m >> 1) if (m & 1) == 0 else (3 * m + 1)
        base = 0 if m == 1 else int(ts[m])
        L = len(seq)
        for j, v in enumerate(seq):
            if v <= N:
                ts[v] = base + (L - j)
    return ts


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    N = 2_000_000
    ts = total_stopping_times(N)

    # geometric bins -> average sigma per bin (smooths per-n noise, keeps
    # any oscillation coherent in ln n)
    n_bins = 400
    edges = np.unique(np.exp(np.linspace(math.log(1000), math.log(N),
                                         n_bins + 1)).astype(np.int64))
    centers, avg = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        seg = ts[lo:hi]
        seg = seg[seg >= 0]
        if len(seg) > 0:
            centers.append(math.sqrt(lo * hi))
            avg.append(float(seg.mean()))
    centers = np.array(centers)
    avg = np.array(avg)
    t = np.log(centers)

    slope = float(np.polyfit(t, avg, 1)[0])
    y = _detrend_logpower(t, avg, deg=2)  # remove smooth C*ln n trend
    scan = dsi_scan(t, y, omega=(2.0, 40.0), n_freq=2500, n_boot=400)

    print("=" * 72)
    print("Collatz total stopping time -- DSI test (Erdos #1135, $500)")
    print("=" * 72)
    print(f"  N = {N:,},  bins = {len(centers)},  ln n span = {t[-1]-t[0]:.2f}")
    print(f"  growth slope (sigma vs ln n) = {slope:.3f}  "
          f"(theory 2/ln(4/3) = {C_THEORY:.3f})")
    print("  " + "-" * 64)
    print(f"  peak omega = {scan['peak_omega']:.4f}  "
          f"(ratio {scan['geometric_ratio']:.3f})")
    print(f"  LS power = {scan['peak_power']:.3f},  p_value = "
          f"{scan['p_value']:.4f}  (AR(1) red-noise null)")
    print("  nearest special frequency:")
    for name, w in SPECIAL.items():
        print(f"    {name:>12} = {w:7.3f}   (delta = "
              f"{scan['peak_omega']-w:+.3f})")
    verdict = ("DSI-lead" if scan["p_value"] <= 0.01 else "null")
    print(f"\n  verdict: {verdict}")

    out = {"N": N, "n_bins": len(centers), "ln_n_span": float(t[-1]-t[0]),
           "growth_slope": slope, "growth_theory": C_THEORY,
           "special_freqs": SPECIAL, **scan, "verdict": verdict}
    p = Path(__file__).parent / "erdos_collatz_dsi_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Wrote {p}")


if __name__ == "__main__":
    main()
