"""Erdos multiplication-table problem: the growth_catcher mode.

How many DISTINCT products are in the n x n multiplication table?
Erdos showed it is o(n^2); Ford pinned the order:

    M(n) ~ n^2 / (ln n)^delta (ln ln n)^{3/2},   delta = 1 - (1+ln ln 2)/ln 2
                                                       = 0.0860713...

This is the campaign's GROWTH-exponent target (the DSI catcher's companion
mode). We compute M(n) exactly to large n and:
  (1) growth_catcher on M(n) vs ln n  -> leading exponent (should be ~2);
  (2) a log-log fit of M(n)/n^2 to recover Ford's delta -- honestly, this
      sits in the same loglog-asymptopia regime that made r_3(n) hard, so
      the recovered delta is expected to be biased; we report it with the
      caveat rather than pretending precision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from systrophe.growth_catcher import catch_growth, summarize_growth_for_report

DELTA_FORD = 1.0 - (1.0 + math.log(math.log(2))) / math.log(2)  # 0.0860713


def distinct_products(n_max: int) -> np.ndarray:
    """M(n) = # distinct values in {i*j : 1<=i,j<=n}, incremental."""
    seen = np.zeros(n_max * n_max + 1, dtype=bool)
    M = np.zeros(n_max + 1, dtype=np.int64)
    for n in range(1, n_max + 1):
        prod = np.arange(1, n + 1) * n
        new = ~seen[prod]
        seen[prod] = True
        M[n] = M[n - 1] + int(np.count_nonzero(new))
    return M


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    n_max = 8000
    M = distinct_products(n_max)
    ns = np.arange(200, n_max + 1)
    Mn = M[ns].astype(float)
    lnn = np.log(ns.astype(float))

    print("=" * 70)
    print("Erdos multiplication-table problem -- growth_catcher mode")
    print("=" * 70)
    print(f"  n_max = {n_max},  M(n_max) = {M[n_max]:,}  "
          f"(table has {n_max**2:,} cells)")
    print(f"  Ford delta = {DELTA_FORD:.7f}")

    # (1) leading exponent via growth_catcher (log M vs ln n)
    gc = catch_growth(lnn, Mn, n_perm=400, parameter_label="ln n")
    print("\n  (1) leading exponent (growth_catcher, log M vs ln n):")
    print(f"      {summarize_growth_for_report(gc)}")
    print(f"      -> exponent {gc.growth_exponent:.4f} (expected ~2 for "
          f"M ~ n^2/polylog)")

    # (2) recover Ford's delta from M/n^2 ~ (ln n)^{-delta}(ln ln n)^{3/2}
    R = Mn / ns.astype(float) ** 2
    logR = np.log(R)
    L1 = np.log(lnn)            # ln ln n
    L2 = np.log(L1)            # ln ln ln n
    # 1-param: logR ~ -delta * ln ln n + c
    s1 = np.polyfit(L1, logR, 1)[0]
    # 2-param: logR ~ -delta*L1 + (3/2)*L2 + c  (least squares)
    A = np.column_stack([L1, L2, np.ones_like(L1)])
    coef, *_ = np.linalg.lstsq(A, logR, rcond=None)
    print("\n  (2) Ford delta recovery from M(n)/n^2:")
    print(f"      1-param (ln M/n^2 vs ln ln n): delta_eff = {-s1:.4f}")
    print(f"      2-param (+ (3/2) ln ln ln n) : delta = {-coef[0]:.4f}, "
          f"L2-coef = {coef[1]:.3f} (theory 1.5)")
    print(f"      Ford theory delta = {DELTA_FORD:.4f}")
    print("\n  note: loglog asymptopia -- at n=8000, ln ln n only spans "
          f"[{L1[0]:.2f},{L1[-1]:.2f}]; delta is biased, not precise.")

    out = {"n_max": n_max, "M_nmax": int(M[n_max]),
           "ford_delta": DELTA_FORD,
           "leading_exponent": gc.growth_exponent,
           "leading_exponent_z": gc.significance_z,
           "delta_1param": float(-s1), "delta_2param": float(-coef[0]),
           "L2_coef_2param": float(coef[1]),
           "lnln_span": [float(L1[0]), float(L1[-1])]}
    p = Path(__file__).parent / "erdos_multiplication_table_growth_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Wrote {p}")


if __name__ == "__main__":
    main()
