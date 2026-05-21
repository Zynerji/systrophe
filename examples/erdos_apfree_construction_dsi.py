"""(a) Where the complex dimension actually lives: the AP-free CONSTRUCTION.

The exact extremal sequence r_3(n) (A003002) is length-walled at ~211 terms
-- too short to confirm a complex dimension (see erdos_a003002_recheck.py:
omega~5.95 but p~0.08, and span only resolves to +-1.17). But the GREEDY
3-term-AP-free set -- the Stanley sequence A005836 = nonneg integers with no
digit 2 in base 3 -- is computable to arbitrarily large N, and its counting
function is the textbook Cantor / complex-dimension object:

    C(N) = #{m <= N : base-3(m) has no digit 2}
         ~ N^{ln2/ln3} * P(ln N),   P log-periodic with period ln 3.

So it has EXACT discrete-scale invariance with

    omega_theory = 2*pi / ln 3 = 5.7192,
    geometric ratio lambda = exp(2*pi/omega) = 3.000  (the base).

This is the cleanest possible positive for the DSI catcher: a provable
complex dimension at a known frequency. Recovering omega = 2pi/ln3 (and
ratio = 3) to high precision and significance:
  (i) validates "AP-free constructions carry discrete-scale invariance"
      -- the legitimate, true version of "make it a sinusoid";
  (ii) and the recovered frequency is within one resolution element of the
       exact-r_3(n) tickle at 5.95, consistent with (but not proof of) the
       extremal sequence inheriting a faint echo of the greedy structure.

It is NOT a route to the growth exponent or the $10k problem -- the greedy
set is far from extremal (Behrend beats it) -- it just shows the catcher
correctly reads a real complex dimension in the AP world.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from erdos_dsi_sweep import dsi_scan, _detrend_logpower

OMEGA_THEORY = 2 * math.pi / math.log(3)
RATIO_THEORY = 3.0


def count_base3_no2(N: int) -> int:
    """#{m in [0, N] : base-3 representation of m has no digit 2}."""
    if N < 0:
        return 0
    digs = []
    x = N
    while x > 0:
        digs.append(x % 3)
        x //= 3
    digs = digs[::-1] or [0]
    L = len(digs)
    count = 0
    for i, d in enumerate(digs):
        rem = L - i - 1
        nlt = sum(1 for c in (0, 1) if c < d)   # valid digits below d
        count += nlt * (2 ** rem)
        if d > 1:                                # prefix can't stay equal
            break
    else:
        count += 1                               # N itself is valid
    return count


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    Nmax = 3 ** 20                               # 3,486,784,401
    Nmin = 1000
    n_samples = 2500
    Ns = np.unique(np.exp(np.linspace(math.log(Nmin), math.log(Nmax),
                                      n_samples)).astype(np.int64))
    C = np.array([count_base3_no2(int(N)) for N in Ns], dtype=float)

    t = np.log(Ns.astype(float))
    logy = np.log(C)
    # power-law (N^{ln2/ln3}) + curvature removed; residual = the log-periodic P
    y = _detrend_logpower(t, logy, deg=3)
    scan = dsi_scan(t, y, omega=(2.0, 30.0), n_freq=3000, n_boot=500)

    print("=" * 72)
    print("AP-free CONSTRUCTION DSI: greedy 3-AP-free set (base-3 no-2, "
          "A005836)")
    print("=" * 72)
    print(f"  N range [{Nmin}, {Nmax}],  points {len(Ns)},  "
          f"ln N span {t[-1]-t[0]:.2f},  Rayleigh dω {2*math.pi/(t[-1]-t[0]):.3f}")
    print(f"  empirical exponent (slope log C vs ln N): "
          f"{np.polyfit(t, logy, 1)[0]:.4f}  (theory ln2/ln3 = "
          f"{math.log(2)/math.log(3):.4f})")
    print("  " + "-" * 64)
    print(f"  recovered omega = {scan['peak_omega']:.4f}   "
          f"(theory 2pi/ln3 = {OMEGA_THEORY:.4f}, "
          f"err {100*(scan['peak_omega']-OMEGA_THEORY)/OMEGA_THEORY:+.2f}%)")
    print(f"  recovered ratio = {scan['geometric_ratio']:.4f}   "
          f"(theory = {RATIO_THEORY:.4f})")
    print(f"  LS power = {scan['peak_power']:.3f},  p_value = "
          f"{scan['p_value']:.4f}  (floor 1/501 = 0.002)")
    print()
    verdict = ("DSI confirmed (complex dimension recovered)"
               if scan["p_value"] <= 0.01 and
               abs(scan["peak_omega"] - OMEGA_THEORY) < 0.3
               else "not recovered")
    print(f"  verdict: {verdict}")

    out = {"Nmax": Nmax, "n_points": len(Ns), "ln_N_span": float(t[-1]-t[0]),
           "empirical_exponent": float(np.polyfit(t, logy, 1)[0]),
           "exponent_theory": math.log(2)/math.log(3),
           "omega_theory": OMEGA_THEORY, "ratio_theory": RATIO_THEORY,
           **scan, "verdict": verdict}
    p = Path(__file__).parent / "erdos_apfree_construction_dsi_results.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Wrote {p}")


if __name__ == "__main__":
    main()
