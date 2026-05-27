"""sqrt(x)-variable variant: the divisor and circle errors ARE oscillatory --
just not in ln x. They are Voronoi/Hardy-periodic in sqrt(x).

In the Erdos x OEIS log-periodic sweep, the Dirichlet divisor error
Delta(x) and the Gauss circle error P(x) came back null -- correctly, since
their oscillation lives in sqrt(x), not ln x. This script points the SAME
Lomb-Scargle + AR(1) detector at the variable u = sqrt(x) and recovers the
known analytic combs:

  * Divisor (Voronoi):  Delta(x) = (x^{1/4}/(pi*sqrt2)) sum_n d(n) n^{-3/4}
                        cos(4 pi sqrt(n x) - pi/4)
        => peaks at omega = 4*pi*sqrt(n)  in u = sqrt(x); fundamental 4pi = 12.566.
  * Circle  (Hardy):    P(x)     = (x^{1/4}/pi) sum_n r2(n) n^{-3/4}
                        cos(2 pi sqrt(n x) - 3 pi/4)
        => peaks at omega = 2*pi*sqrt(n); fundamental 2pi = 6.283
           (only n that are sums of two squares contribute, r2(n) > 0).

The 4pi vs 2pi split (zeta^2 vs zeta*L functional equations) is the
discriminating prediction. Recovering both combs shows the detector reads
the right structure once pointed at the right variable -- the additive /
lattice counterpart to the ln-x (Euler-product) result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, lombscargle

from erdos_dsi_sweep import _ar1_surrogate

EG = 0.5772156649015329
FOURPI = 4 * math.pi
TWOPI = 2 * math.pi


def divisor_series(x_min, x_max, n):
    N = int(x_max)
    d = np.zeros(N + 1)
    for k in range(1, N + 1):
        d[k::k] += 1.0
    D = np.cumsum(d)
    u = np.linspace(math.sqrt(x_min), math.sqrt(x_max), n)  # u = sqrt(x)
    x = u ** 2
    xi = np.floor(x).astype(np.int64)
    err = D[xi] - x * (np.log(x) + 2 * EG - 1.0)
    return u, err / np.sqrt(u)          # divide by x^{1/4} = u^{1/2}


def circle_series(R_min, R_max, n):
    u = np.linspace(R_min, R_max, n)    # u = R = sqrt(x), x = R^2 (area)
    P = np.empty(n)
    for i, R in enumerate(u):
        ri = int(R)
        a = np.arange(0, ri + 1)
        col = np.floor(np.sqrt(np.maximum(R * R - a * a, 0.0)))
        P[i] = (1 + 4 * np.sum(col)) - math.pi * R * R
    return u, P / np.sqrt(u)            # x^{1/4} = R^{1/2} = sqrt(u)


def scan(u, y, freqs, n_boot, seed):
    y0 = y - y.mean()
    power = lombscargle(u, y0, freqs, normalize=True)
    peak_power = float(power.max())
    pk, _ = find_peaks(power, height=peak_power * 0.15)
    order = np.argsort(power[pk])[::-1]
    top = pk[order][:6]
    rng = np.random.default_rng(seed)
    nm = np.empty(n_boot)
    for b in range(n_boot):
        nm[b] = lombscargle(u, _ar1_surrogate(y0, rng), freqs,
                            normalize=True).max()
    p = float((np.sum(nm >= peak_power) + 1) / (n_boot + 1))
    return power, [(float(freqs[i]), float(power[i])) for i in top], p


def match_comb(freq, base):
    n_est = (freq / base) ** 2
    n = max(1, round(n_est))
    f_pred = base * math.sqrt(n)
    return n, f_pred, 100.0 * (freq - f_pred) / f_pred


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    n_samples, n_boot = 12000, 150
    print("=" * 74)
    print("sqrt(x)-variable DSI: divisor (Voronoi 4pi) & circle (Hardy 2pi) "
          "errors")
    print("=" * 74)

    out = {}

    # ---- Dirichlet divisor ----
    u, y = divisor_series(1e4, 1e6, n_samples)
    freqs = np.linspace(3.0, 42.0, 3000)
    power, top, p = scan(u, y, freqs, n_boot, seed=1)
    print(f"\n  DIVISOR  Delta(x), u=sqrt(x) in [{u[0]:.0f},{u[-1]:.0f}], "
          f"p_value={p:.4f}  (was null in ln x)")
    print(f"    fundamental predicted 4pi = {FOURPI:.4f}")
    print(f"    {'peak omega':>11}  {'power':>6}  {'-> n':>4}  "
          f"{'4pi*sqrt(n)':>11}  {'err':>7}")
    dts = []
    for f, pw in top:
        n, fp, e = match_comb(f, FOURPI)
        print(f"    {f:>11.3f}  {pw:>6.3f}  {n:>4d}  {fp:>11.3f}  {e:>6.2f}%")
        dts.append({"omega": f, "power": pw, "n": n, "pred": fp, "err_pct": e})
    out["divisor"] = {"p_value": p, "fundamental_4pi": FOURPI, "peaks": dts}

    # ---- Gauss circle ----
    u, y = circle_series(100.0, 1600.0, n_samples)
    freqs = np.linspace(3.0, 30.0, 3000)
    power, top, p = scan(u, y, freqs, n_boot, seed=2)
    print(f"\n  CIRCLE   P(x), u=R=sqrt(x) in [{u[0]:.0f},{u[-1]:.0f}], "
          f"p_value={p:.4f}  (was null in ln x)")
    print(f"    fundamental predicted 2pi = {TWOPI:.4f}")
    print(f"    {'peak omega':>11}  {'power':>6}  {'-> n':>4}  "
          f"{'2pi*sqrt(n)':>11}  {'err':>7}")
    cts = []
    for f, pw in top:
        n, fp, e = match_comb(f, TWOPI)
        print(f"    {f:>11.3f}  {pw:>6.3f}  {n:>4d}  {fp:>11.3f}  {e:>6.2f}%")
        cts.append({"omega": f, "power": pw, "n": n, "pred": fp, "err_pct": e})
    out["circle"] = {"p_value": p, "fundamental_2pi": TWOPI, "peaks": cts}

    p_ = Path(__file__).parent / "erdos_sqrtn_lattice_dsi_results.json"
    p_.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Wrote {p_}")


if __name__ == "__main__":
    main()
