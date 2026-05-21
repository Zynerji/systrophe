"""Sign-aware DSI: log-periodicity in SIGNED summatory functions.

Closes the documented sign-aware gap (the OEIS sweep skipped signed
sequences because it took logs). Signed oscillators that cross zero are
handled by normalising by their conjectured power x^theta and running the
detector directly (no log).

Targets:
  * Mertens   M(x) = sum_{n<=x} mu(n)        -- explicit formula gives
        M(x) ~ sum_rho x^rho/(rho zeta'(rho)), i.e. x^{1/2} oscillations at
        the zeta zeros gamma_k. Expect a DSI comb at omega = gamma_1 = 14.13.
  * Liouville L(x) = sum_{n<=x} lambda(n)     -- Dirichlet series
        zeta(2s)/zeta(s); leading sqrt(x)/zeta(1/2) trend + zeta-zero
        oscillations. Expect DSI at gamma_1 after removing the mean.
  * A140462 (#500 prize, the one signed prize sequence) -- honest test.

This extends the campaign's structural finding -- log-periodicity tracks
Euler-product structure -- to two more classic multiplicative summatory
functions, and tests the remaining signed prize sequence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from erdos_dsi_sweep import dsi_scan, _detrend_logpower, _mobius
from erdos_oeis_dsi_sweep import fetch_bfile

GAMMA1 = 14.134725


def _primes_to(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.flatnonzero(s)


def mertens_summatory(N):
    return np.cumsum(_mobius(N))


def liouville_summatory(N):
    Omega = np.zeros(N + 1, dtype=np.int64)
    for p in _primes_to(N):
        pk = int(p)
        while pk <= N:
            Omega[pk::pk] += 1
            pk *= int(p)
    lam = np.where(Omega % 2 == 0, 1, -1).astype(float)
    lam[0] = 0.0
    return np.cumsum(lam)


def scan_signed_summatory(S, x_min, x_max, n, theta, omega=(5.0, 40.0)):
    t = np.linspace(math.log(x_min), math.log(x_max), n)
    x = np.exp(t)
    xi = np.floor(x).astype(np.int64)
    y = S[xi] / x ** theta            # normalise the power-law envelope
    return dsi_scan(t, y, omega=omega, n_freq=2000, n_boot=300)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    N = 2_000_000
    print("=" * 70)
    print("Sign-aware DSI: Mertens, Liouville, prize A140462")
    print(f"  (gamma_1 = {GAMMA1};  N = {N:,})")
    print("=" * 70)

    results = {}

    M = mertens_summatory(N)
    sm = scan_signed_summatory(M, 1e3, N, 4000, theta=0.5)
    results["mertens"] = sm
    print(f"\n  Mertens M(x)/sqrt(x):  peak omega = {sm['peak_omega']:.4f}  "
          f"(gamma_1 = {GAMMA1}, delta {sm['peak_omega']-GAMMA1:+.3f})")
    print(f"    power = {sm['peak_power']:.3f}, p_value = {sm['p_value']:.4f}")

    L = liouville_summatory(N)
    sl = scan_signed_summatory(L, 1e3, N, 4000, theta=0.5)
    results["liouville"] = sl
    print(f"\n  Liouville L(x)/sqrt(x):  peak omega = {sl['peak_omega']:.4f}  "
          f"(gamma_1 = {GAMMA1}, delta {sl['peak_omega']-GAMMA1:+.3f})")
    print(f"    power = {sl['peak_power']:.3f}, p_value = {sl['p_value']:.4f}")

    # A140462 prize sequence (signed)
    terms = fetch_bfile("A140462")
    if terms:
        idx = np.array([n for n, _ in terms], float)
        val = np.array([v for _, v in terms], float)
        keep = idx > 0
        idx, val = idx[keep], val[keep]
        t = np.log(idx)
        y = _detrend_logpower(t, val, deg=3)   # signed: detrend, no log
        sa = dsi_scan(t, y, omega=(2.0, 40.0), n_freq=2000, n_boot=300)
        results["A140462"] = {**sa, "n_terms": int(len(idx))}
        print(f"\n  A140462 (#500 prize, signed, n={len(idx)}):  "
              f"peak omega = {sa['peak_omega']:.4f}")
        print(f"    power = {sa['peak_power']:.3f}, p_value = {sa['p_value']:.4f}"
              f"  -> {'DSI-lead' if sa['p_value']<=0.01 else 'null'}")
    else:
        print("\n  A140462: fetch failed")

    def verdict(s):
        return ("DSI" if s["p_value"] <= 0.01 else "null")
    print("\n  summary:")
    print(f"    Mertens   : {verdict(sm)} (omega {sm['peak_omega']:.2f})")
    print(f"    Liouville : {verdict(sl)} (omega {sl['peak_omega']:.2f})")

    p = Path(__file__).parent / "erdos_signed_dsi_results.json"
    p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  Wrote {p}")


if __name__ == "__main__":
    main()
