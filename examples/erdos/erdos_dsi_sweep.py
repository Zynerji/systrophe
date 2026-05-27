"""Erdos-problem DSI discovery sweep: scan integer-sequence error terms for
hidden discrete-scale-invariance (log-periodic structure), with red-noise
significance and battery-level multiple-testing control.

Motivation
----------
Per Bloom & Tao, a large fraction of Erdos problems are about an integer
sequence -- typically the extremal size of some structure -- and these are
being linked to the OEIS. The Systrophe DSI / log-periodic stack
(`systrophe.catchers.dsi_observables`, `tipler_fractal`) plus the mandated catcher
are built to detect discrete-scale invariance: oscillations periodic in
ln(x), i.e. f(x) ~ x^c [1 + A cos(omega ln x + phi)], with geometric ratio
lambda = exp(2 pi / omega).

This is a DISCOVERY NET. For each sequence we:
  1. reduce it to an oscillation series y(t), t = ln x (signed error term
     normalised by its conjectured power, or log-power-detrended count);
  2. run a Lomb-Scargle periodogram in t and take the peak;
  3. assess significance against an AR(1) (red-noise) surrogate null using
     the MAX periodogram power (within-sequence look-elsewhere);
  4. Bonferroni-correct across the battery.

A verdict of `DSI` means a log-periodic peak survives correction; most
sequences are expected to be `null`. Two synthetic controls (one
log-periodic, one white noise) validate the net before any hit is trusted.

Expected, as a sanity frame (NOT assumed -- printed from the run):
  * prime psi(x)-x: DSI, omega ~ 14.13 (first zeta zero; Euler-product /
    multiplicative origin gives genuine log-periodicity);
  * squarefree-count error: DSI at HALF the frequencies (~7.07 = gamma_1/2,
    from zeros of zeta(2s));
  * Dirichlet divisor / Gauss circle errors: null here -- they oscillate in
    sqrt(x) (Voronoi/Hardy), not ln x, so they are NOT log-periodic;
  * multiplication-table count (Erdos): null -- its (ln n)^-delta correction
    is a smooth log-power TREND, not an oscillation (growth_catcher's job).

Usage
-----
    python examples/erdos_dsi_sweep.py
    python examples/erdos_dsi_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, lombscargle

from millennium_primes_dsi_inverse import prime_power_steps

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# Sequence / error-term generators -> (t = ln x, y = oscillation series)
# --------------------------------------------------------------------------

def _uniform_log_x(x_min: float, x_max: float, n: int):
    t = np.linspace(math.log(x_min), math.log(x_max), n)
    return t, np.exp(t)


def gen_control_logperiodic(n=1500, omega0=8.0):
    t, _ = _uniform_log_x(1e3, 1e7, n)
    rng = np.random.default_rng(0)
    y = 0.05 * np.cos(omega0 * t + 0.7) + 0.01 * rng.standard_normal(n)
    return t, y, {"kind": "control", "note": f"synthetic log-periodic omega={omega0}"}


def gen_control_noise(n=1500):
    t, _ = _uniform_log_x(1e3, 1e7, n)
    rng = np.random.default_rng(1)
    # AR(1) red noise -- the hardest honest negative control.
    e = rng.standard_normal(n)
    y = lfilter([1.0], [1.0, -0.6], e)
    return t, y, {"kind": "control", "note": "AR(1) red noise"}


def gen_psi_error(n=1500, x_max=1e7):
    t, x = _uniform_log_x(1e3, x_max, n)
    pos, cum = prime_power_steps(int(x_max))
    idx = np.searchsorted(pos, x, side="right")
    psi = np.where(idx > 0, cum[np.clip(idx - 1, 0, len(cum) - 1)], 0.0)
    y = (psi - x) / np.sqrt(x)            # theta = 1/2
    return t, y, {"kind": "erdos-adjacent", "note": "prime psi(x)-x (Chebyshev)"}


def gen_squarefree_error(n=1500, x_max=1e7):
    t, x = _uniform_log_x(1e3, x_max, n)
    K = int(math.isqrt(int(x_max))) + 1
    mu = _mobius(K)
    ks = np.arange(1, K + 1)
    mu_over = mu[1:]                       # mu[1..K]
    Q = np.array([np.sum(mu_over * np.floor(xx / (ks ** 2))) for xx in x])
    err = Q - x * (6.0 / math.pi ** 2)
    y = err / x ** 0.25                    # conjectured theta = 1/4
    return t, y, {"kind": "erdos-adjacent",
                  "note": "squarefree count Q(x) - 6x/pi^2"}


def gen_divisor_error(n=1500, N=1_000_000):
    d = np.zeros(N + 1)
    for k in range(1, N + 1):
        d[k::k] += 1.0
    D = np.cumsum(d)
    t, x = _uniform_log_x(1e3, N, n)
    xi = np.floor(x).astype(int)
    err = D[xi] - x * (np.log(x) + 2 * EULER_GAMMA - 1.0)
    y = err / x ** 0.25                    # Dirichlet divisor problem
    return t, y, {"kind": "erdos-adjacent",
                  "note": "Dirichlet divisor error Delta(x)"}


def gen_circle_error(n=1500, x_max=1e7):
    t, x = _uniform_log_x(1e4, x_max, n)   # x = area scale = R^2
    R = np.sqrt(x)
    counts = np.empty(n)
    for i, r in enumerate(R):
        ri = int(r)
        a = np.arange(0, ri + 1)
        col = np.floor(np.sqrt(np.maximum(r * r - a * a, 0.0)))
        counts[i] = 1 + 4 * np.sum(col)    # lattice points in disk radius r
    err = counts - math.pi * x
    y = err / x ** 0.25                    # Gauss circle problem
    return t, y, {"kind": "erdos-adjacent",
                  "note": "Gauss circle error P(x)"}


def gen_multiplication_table(n_max=2000):
    seen = np.zeros(n_max * n_max + 1, dtype=bool)
    C = np.zeros(n_max + 1, dtype=np.int64)
    for nn in range(1, n_max + 1):
        prod = np.arange(1, nn + 1) * nn
        new = ~seen[prod]
        seen[prod] = True
        C[nn] = C[nn - 1] + int(np.count_nonzero(new))
    ns = np.arange(50, n_max + 1)
    t = np.log(ns.astype(float))
    y = _detrend_logpower(t, np.log(C[ns].astype(float)), deg=3)
    return t, y, {"kind": "erdos",
                  "note": "distinct products in n x n table (Erdos)"}


def gen_goldbach(M=12000):
    sieve = _prime_sieve(M)
    evens = np.arange(50, M, 2)
    g = np.empty(len(evens), dtype=float)
    for i, twom in enumerate(evens):
        m = twom // 2
        ps = np.flatnonzero(sieve[:m + 1])
        g[i] = np.count_nonzero(sieve[twom - ps])
    t = np.log(evens.astype(float))
    y = _detrend_logpower(t, np.log(g), deg=3)
    return t, y, {"kind": "erdos-adjacent",
                  "note": "Goldbach representation count g(2m)"}


# --------------------------------------------------------------------------
# Small number-theory helpers
# --------------------------------------------------------------------------

def _prime_sieve(n: int) -> np.ndarray:
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if s[p]:
            s[p * p::p] = False
    return s


def _mobius(n: int) -> np.ndarray:
    mu = np.ones(n + 1, dtype=np.int64)
    mu[0] = 0
    is_comp = np.zeros(n + 1, dtype=bool)
    primes = []
    for i in range(2, n + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > n:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _detrend_logpower(t: np.ndarray, logy: np.ndarray, deg: int = 3):
    """Remove a smooth log-power trend (polynomial in t = ln x)."""
    coef = np.polyfit(t, logy, deg)
    return logy - np.polyval(coef, t)


# --------------------------------------------------------------------------
# DSI detector: Lomb-Scargle in ln x + AR(1) red-noise significance
# --------------------------------------------------------------------------

def _resample_uniform(t, y, n=None):
    n = n or len(t)
    tu = np.linspace(t.min(), t.max(), n)
    return tu, np.interp(tu, t, y)


def _ar1_surrogate(y0, rng):
    n = len(y0)
    phi = np.corrcoef(y0[1:], y0[:-1])[0, 1]
    phi = float(np.clip(phi if np.isfinite(phi) else 0.0, 0.0, 0.98))
    e = rng.standard_normal(n)
    s = lfilter([math.sqrt(1 - phi ** 2)], [1.0, -phi], e)
    s -= s.mean()
    sd = s.std()
    return s * (y0.std() / sd) if sd > 0 else s


def dsi_scan(t, y, omega=(2.0, 60.0), n_freq=1200, n_boot=200, seed=0):
    tu, yu = _resample_uniform(t, y)
    y0 = yu - yu.mean()
    freqs = np.linspace(omega[0], omega[1], n_freq)
    power = lombscargle(tu, y0, freqs, normalize=True)
    pk = int(np.argmax(power))
    peak_omega, peak_power = float(freqs[pk]), float(power[pk])

    rng = np.random.default_rng(seed)
    null_max = np.empty(n_boot)
    for b in range(n_boot):
        s = _ar1_surrogate(y0, rng)
        null_max[b] = lombscargle(tu, s, freqs, normalize=True).max()
    p_value = float((np.sum(null_max >= peak_power) + 1) / (n_boot + 1))

    return {
        "peak_omega": peak_omega,
        "geometric_ratio": float(math.exp(2 * math.pi / peak_omega)),
        "peak_power": peak_power,
        "p_value": p_value,
        "null_power_95": float(np.quantile(null_max, 0.95)),
    }


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def run(quick: bool) -> dict:
    n = 800 if quick else 1500
    xmax = 1e6 if quick else 1e7
    Ndiv = 200_000 if quick else 1_000_000
    nmt = 1000 if quick else 2000
    n_boot = 120 if quick else 200

    battery = {
        "control_logperiodic": gen_control_logperiodic(n),
        "control_rednoise":    gen_control_noise(n),
        "psi_error":           gen_psi_error(n, xmax),
        "squarefree_error":    gen_squarefree_error(n, xmax),
        "divisor_error":       gen_divisor_error(n, Ndiv),
        "circle_error":        gen_circle_error(n, xmax),
        "mult_table_erdos":    gen_multiplication_table(nmt),
        "goldbach_count":      gen_goldbach(8000 if quick else 12000),
    }

    n_tests = sum(1 for _, (_, _, m) in battery.items()
                  if m["kind"] != "control")
    bonf = 0.05 / n_tests

    results = []
    for name, (t, y, meta) in battery.items():
        scan = dsi_scan(t, y, n_boot=n_boot)
        is_ctrl = meta["kind"] == "control"
        verdict = ("DSI" if (not is_ctrl and scan["p_value"] <= bonf)
                   else ("DSI" if (is_ctrl and scan["p_value"] <= 0.05)
                         else "null"))
        results.append({"name": name, **meta, **scan, "verdict": verdict})

    results.sort(key=lambda r: r["p_value"])
    return {
        "n_noncontrol_tests": n_tests,
        "bonferroni_threshold": bonf,
        "n_points": n,
        "results": results,
    }


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("Erdos DSI discovery sweep  (log-periodic structure, AR(1) "
          "red-noise significance)")
    print("=" * 78)
    out = run(args.quick)
    print(f"  non-control tests = {out['n_noncontrol_tests']}, "
          f"Bonferroni threshold p <= {out['bonferroni_threshold']:.4f}, "
          f"n_points = {out['n_points']}")
    print("  " + "-" * 74)
    print(f"  {'sequence':>20}  {'omega':>7}  {'ratio':>6}  {'power':>6}  "
          f"{'p_value':>8}  {'verdict':>7}")
    for r in out["results"]:
        print(f"  {r['name']:>20}  {r['peak_omega']:>7.3f}  "
              f"{r['geometric_ratio']:>6.3f}  {r['peak_power']:>6.3f}  "
              f"{r['p_value']:>8.4f}  {r['verdict']:>7}")
    print()
    for r in out["results"]:
        print(f"    {r['name']:>20} : {r['note']}")
    print()

    out_path = Path(__file__).parent / "erdos_dsi_sweep_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
