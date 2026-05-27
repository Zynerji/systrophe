"""Millennium-problem exploration: Birch-Swinnerton-Dyer via the
local data (a_p sequences) of small elliptic curves.

The BSD conjecture relates the rank of E(Q) (the Mordell-Weil group)
to the order of vanishing of the L-function L(E, s) at s = 1. In
practice, for low-rank curves, the local data a_p = p + 1 - #E(F_p)
encode the rank: rank-0 curves have L(1) > 0, rank-1 curves have
L(1) = 0 with L'(1) > 0, etc.

A partial-Euler-product approximation:

    L(E, s) ~ prod_{p prime, p <= N} (1 - a_p p^{-s} + p^{1-2s})^{-1}

evaluated at s = 1, truncated at small N, gives an estimator that:
  - converges to a positive limit for rank-0 curves
  - "diverges" (the partial product grows without bound) for rank-1+

This script:
  1. Computes a_p for primes p up to P_MAX for several known
     elliptic curves of rank 0, 1, 2.
  2. Estimates log L(E, 1) via the partial Euler product.
  3. Applies the Systrophe catcher to the (rank, partial_log_L)
     pairs to ask: does the catcher distinguish ranks from local data?

This is a tool exploration of BSD-adjacent local-to-global structure;
NOT a proof of BSD.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

from systrophe.catchers.derivative_catcher import catch_smooth_transition
from systrophe.catchers.novelty_catcher import (
    catch_novelty_per_quantity,
    scan_novelty,
)


def sieve_primes(n_max: int) -> list[int]:
    sieve = bytearray([1]) * (n_max + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n_max ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, n_max + 1, i)))
    return [i for i in range(n_max + 1) if sieve[i]]


def count_points_E_Fp(a4: int, a6: int, p: int) -> int:
    """Count points on the elliptic curve E: y^2 = x^3 + a4*x + a6 over F_p
    (including the point at infinity).

    For each x in F_p, count solutions y such that y^2 = x^3 + a4*x + a6
    mod p. Add 1 for the point at infinity. Naive O(p^2) implementation
    suitable for small p only.
    """
    count = 1  # point at infinity
    # Precompute quadratic residues mod p
    qr = set()
    for y in range(p):
        qr.add((y * y) % p)
    for x in range(p):
        rhs = (x * x * x + a4 * x + a6) % p
        if rhs == 0:
            count += 1  # y = 0 unique
        elif rhs in qr:
            count += 2  # two square roots
    return count


def ap_sequence(a4: int, a6: int, primes: list[int]) -> dict[int, int]:
    """Return dict {p: a_p} for primes p in `primes` (skipping bad reduction)."""
    out = {}
    discriminant = -16 * (4 * a4 ** 3 + 27 * a6 ** 2)
    for p in primes:
        if discriminant % p == 0:
            continue  # bad reduction; skip
        if p == 2:
            continue  # avoid pathology at p=2
        nE = count_points_E_Fp(a4, a6, p)
        ap = p + 1 - nE
        out[p] = ap
    return out


def partial_log_L(ap: dict[int, int]) -> dict:
    """Approximate log L(E, 1) via the partial Euler product:

      log L(E, 1) ~ sum_p log(1 - a_p / p + 1/p)^{-1}
                  = -sum_p log(1 - a_p / p + 1/p)
                  ~ sum_p (a_p / p - 1/p)  (leading order in 1/p)
    """
    s_full = 0.0  # full log of (1/L_p(s=1))
    primes_used = sorted(ap.keys())
    cumulative = []
    for p in primes_used:
        a_p = ap[p]
        factor = 1.0 - a_p / p + 1.0 / p
        if factor <= 0:
            # Local L-factor is non-positive; treat as truncation error
            cumulative.append(s_full)
            continue
        s_full += -math.log(factor)
        cumulative.append(s_full)
    return {
        "primes": primes_used,
        "log_L_partial_at_each_p": cumulative,
        "log_L_final": s_full,
    }


def main() -> None:
    # Extended set of short Weierstrass curves y^2 = x^3 + a4*x + a6.
    # Ranks are from the literature (Cremona tables, congruent-number
    # results, twist families). This is still a small sample, but
    # broader than the initial 6-curve probe.
    curves = [
        # (label, a4, a6, expected_rank)
        # --- Rank 0 ---
        ("y2=x3+1",      0,    1,    0),   # j=0 torsion-only
        ("y2=x3-1",      0,   -1,    0),
        ("y2=x3+x",      1,    0,    0),   # CM by Z[i]
        ("y2=x3+x+1",    1,    1,    0),
        ("y2=x3-x+1",   -1,    1,    0),
        ("y2=x3+2x+1",   2,    1,    0),
        ("y2=x3+4",      0,    4,    0),
        ("y2=x3-4",      0,   -4,    0),
        # --- Rank 1 ---
        ("y2=x3-x",     -1,    0,    1),   # congruent-number n=1
        ("y2=x3-4x",    -4,    0,    1),   # n=2
        ("y2=x3-9x",    -9,    0,    1),   # n=3
        ("y2=x3-25x",  -25,    0,    1),   # n=5
        ("y2=x3-2",      0,   -2,    1),
        ("y2=x3-7",      0,   -7,    1),
        # --- Rank 2 ---
        ("y2=x3+17",     0,   17,    2),
        ("y2=x3-49x",  -49,    0,    2),   # n=7 (congruent)
        ("y2=x3-225x",-225,    0,    2),   # n=15 (congruent)
    ]
    P_MAX = 500
    primes = sieve_primes(P_MAX)
    print(f"Computing a_p for primes <= {P_MAX} ({len(primes)} primes)...")
    print()

    log_L_per_rank = {0: [], 1: [], 2: []}
    per_curve = []
    for label, a4, a6, rank in curves:
        ap = ap_sequence(a4, a6, primes)
        ll = partial_log_L(ap)
        per_curve.append({
            "label": label, "a4": a4, "a6": a6,
            "expected_rank": rank,
            "n_primes_used": len(ll["primes"]),
            "log_L_final": ll["log_L_final"],
            "ap_sample": {p: ap[p] for p in ll["primes"][:5]},
        })
        log_L_per_rank[rank].append(ll["log_L_final"])
        print(f"  {label} (rank={rank}, a4={a4}, a6={a6}): "
              f"log L_partial = {ll['log_L_final']:+.4f}, "
              f"#primes = {len(ll['primes'])}")
    print()

    # Catcher pass: do the rank-stratified distributions show novel structure?
    by_rank = {f"rank_{k}": np.array(v) for k, v in log_L_per_rank.items()
               if len(v) > 0}
    pq = catch_novelty_per_quantity({"log_L_partial": by_rank}, n_bins=32)

    print("Catcher per-rank verdict:")
    print(f"  aggregate: {pq['aggregate_verdict']}")
    for q, r in pq["per_quantity"].items():
        print(f"    {q}: verdict={r['verdict']}")

    # Mean per rank
    print()
    print("Mean partial log L(E, 1) by rank:")
    for k in (0, 1, 2):
        vs = log_L_per_rank[k]
        if vs:
            print(f"  rank {k}: mean = {np.mean(vs):+.4f}, n = {len(vs)}")
    print()
    print("BSD prediction: rank up => log L_partial up (since the partial product")
    print("approximates an L-function that vanishes more strongly at higher rank).")

    out_path = Path(__file__).parent / "millennium_bsd_catcher_results.json"
    out_path.write_text(json.dumps({
        "P_MAX": P_MAX,
        "per_curve": per_curve,
        "log_L_per_rank": {k: list(map(float, v))
                            for k, v in log_L_per_rank.items()},
        "catcher_verdict": pq["aggregate_verdict"],
        "catcher_per_rank": {
            q: r["verdict"] for q, r in pq["per_quantity"].items()
        },
    }, indent=2))
    print()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
