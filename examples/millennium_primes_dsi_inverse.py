"""Millennium-adjacent experiment: recover the Riemann zeta zeros from the
PRIMES THEMSELVES, by treating the prime-counting fluctuation as a
discrete-scale-invariant (log-periodic) cascade.

Motivation (the Systrophe DSI bridge)
-------------------------------------
The Riemann-von Mangoldt explicit formula for the second Chebyshev
function psi(x) = sum_{p^k <= x} log p is

    psi_0(x) = x - sum_rho x^rho / rho - log(2 pi) - (1/2) log(1 - x^-2),

where the sum runs over the non-trivial zeta zeros rho = 1/2 + i*gamma.
Pairing each zero with its conjugate and writing u = ln x gives the
NORMALISED fluctuation

    f(u) := (psi(e^u) - e^u + log(2 pi) + (1/2) log(1 - e^{-2u})) / e^{u/2}
          = - sum_gamma  2 * [ (1/2) cos(gamma u) + gamma sin(gamma u) ]
                              / (1/4 + gamma^2).

Every term is a *log-periodic cosine* cos(gamma * ln x): exactly the
functional form of Systrophe's Tipler sinusoid F(r) = A cos(alpha ln(r/R))
(see `systrophe.tipler_fractal` / `systrophe.dsi_observables`), with the
substitution alpha <-> gamma. So the prime fluctuation IS a Systrophe
cascade-DSI signal whose log-periodic frequencies are the zeta zeros.

This script computes f(u) directly from the primes (no zeta function),
runs a Lomb-Scargle periodogram in the u = ln x variable, and checks
whether the spectral peaks land on the first few zeta zeros gamma_k.
It then (a) quantitatively confirms the bridge by correlating the
measured fluctuation against the truncated explicit-formula cascade,
and (b) passes the prime spectrum through the mandated Systrophe
address-space novelty catcher.

This is a tool-based exploration, not a proof of anything. It is the
inverse of the existing `millennium_riemann_catcher.py`: that one feeds
zeta zeros to the catcher; this one recovers the zeros from the primes.

Usage
-----
    python examples/millennium_primes_dsi_inverse.py            # X = 1e7
    python examples/millennium_primes_dsi_inverse.py --x-max 5e7
    python examples/millennium_primes_dsi_inverse.py --quick    # X = 1e6
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, lombscargle

from systrophe.novelty_catcher import scan_novelty, summarize_novelty_for_report


# --------------------------------------------------------------------------
# Prime machinery: Chebyshev psi(x) via a sieve over prime powers
# --------------------------------------------------------------------------

def primes_up_to(n: int) -> np.ndarray:
    """Boolean-sieve the primes <= n, returned as an int64 array."""
    if n < 2:
        return np.array([], dtype=np.int64)
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(math.isqrt(n)) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.flatnonzero(sieve).astype(np.int64)


def prime_power_steps(x_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted prime-power positions p^k <= x_max and their psi-weights log p.

    psi(x) = sum over those positions <= x of log p. Returns
    (positions_sorted, cumulative_log_weight) so psi(x) is a searchsorted.
    """
    primes = primes_up_to(x_max)
    positions = [primes.astype(float)]
    weights = [np.log(primes.astype(float))]
    # Higher prime powers p^k, k >= 2, that are still <= x_max.
    k = 2
    while True:
        # primes p with p^k <= x_max  <=>  p <= x_max ** (1/k)
        limit = x_max ** (1.0 / k)
        sel = primes[primes <= limit]
        if sel.size == 0:
            break
        positions.append((sel.astype(float)) ** k)
        weights.append(np.log(sel.astype(float)))  # weight is log p, not log p^k
        k += 1
    pos = np.concatenate(positions)
    w = np.concatenate(weights)
    order = np.argsort(pos)
    pos = pos[order]
    cum = np.cumsum(w[order])
    return pos, cum


def make_psi(x_max: int):
    """Return a vectorised psi(x) callable for x in (0, x_max]."""
    pos, cum = prime_power_steps(x_max)

    def psi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(pos, x, side="right")  # count of steps <= x
        out = np.where(idx > 0, cum[np.clip(idx - 1, 0, len(cum) - 1)], 0.0)
        return out

    return psi


# --------------------------------------------------------------------------
# Normalised prime fluctuation f(u) and the explicit-formula cascade
# --------------------------------------------------------------------------

def prime_fluctuation(psi, u: np.ndarray) -> np.ndarray:
    """f(u) = (psi(e^u) - e^u + log(2pi) + 0.5 log(1 - e^{-2u})) / e^{u/2}."""
    x = np.exp(u)
    corr = math.log(2.0 * math.pi) + 0.5 * np.log1p(-np.exp(-2.0 * u))
    return (psi(x) - x + corr) / np.sqrt(x)


def explicit_formula_cascade(u: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """The truncated explicit-formula reconstruction of f(u) from the zeros.

    f_model(u) = - sum_gamma 2 [ 0.5 cos(gamma u) + gamma sin(gamma u) ]
                              / (0.25 + gamma^2)
    """
    g = gammas[:, None]
    terms = 2.0 * (0.5 * np.cos(g * u[None, :]) + g * np.sin(g * u[None, :])) \
        / (0.25 + g ** 2)
    return -terms.sum(axis=0)


# --------------------------------------------------------------------------
# Zeta zeros (ground truth, only used for scoring the recovery)
# --------------------------------------------------------------------------

def zeta_zeros(n_zeros: int) -> np.ndarray:
    from mpmath import mp, zetazero
    mp.dps = 15
    return np.array([float(zetazero(k).imag) for k in range(1, n_zeros + 1)])


def lombscargle_chunked(u: np.ndarray, y: np.ndarray, freqs: np.ndarray,
                        chunk: int = 1000) -> np.ndarray:
    """scipy.signal.lombscargle, evaluated in frequency chunks.

    Newer scipy builds an (n_samples, n_freq) matrix internally, which is
    multi-GB for fine grids on long signals. Chunking the frequency axis
    bounds peak memory to n_samples * chunk floats.
    """
    out = np.empty(len(freqs), dtype=float)
    for i in range(0, len(freqs), chunk):
        sl = slice(i, i + chunk)
        out[sl] = lombscargle(u, y, freqs[sl], normalize=True)
    return out


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def run(x_max: int, n_zeros: int, n_samples: int, x_min: float,
        gamma_lo: float, gamma_hi: float, n_freq: int) -> dict:
    u_min = math.log(x_min)
    u_max = math.log(x_max)
    u = np.linspace(u_min, u_max, n_samples)

    # Sieve once: build psi from the prime-power steps and reuse the count.
    pos, cum = prime_power_steps(x_max)

    def psi(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(pos, x, side="right")
        return np.where(idx > 0, cum[np.clip(idx - 1, 0, len(cum) - 1)], 0.0)

    f = prime_fluctuation(psi, u)
    f0 = f - f.mean()

    # Lomb-Scargle periodogram in angular frequency = gamma.
    freqs = np.linspace(gamma_lo, gamma_hi, n_freq)
    power = lombscargle_chunked(u, f0, freqs)

    # True zeros for scoring.
    gammas = zeta_zeros(n_zeros)

    # Recovered peaks: prominent local maxima of the periodogram.
    peak_idx, props = find_peaks(power, prominence=power.max() * 0.02)
    peak_freqs = freqs[peak_idx]
    peak_powers = power[peak_idx]
    order = np.argsort(peak_powers)[::-1]
    peak_freqs = peak_freqs[order]
    peak_powers = peak_powers[order]

    # Match each true zero to its nearest recovered peak.
    matches = []
    for k, g in enumerate(gammas, start=1):
        if len(peak_freqs) == 0:
            break
        j = int(np.argmin(np.abs(peak_freqs - g)))
        rec = float(peak_freqs[j])
        err = rec - float(g)
        matches.append({
            "k": k,
            "gamma_true": float(g),
            "gamma_recovered": rec,
            "abs_error": float(err),
            "pct_error": float(100.0 * err / g),
            "peak_power": float(peak_powers[j]),
        })

    # Within a Rayleigh resolution dgamma ~ 2*pi/span, a recovered peak
    # "hits" the zero. Report how many of the first n_match land within it.
    span = u_max - u_min
    rayleigh = 2.0 * math.pi / span
    n_score = min(6, len(matches))
    hits = sum(1 for m in matches[:n_score]
               if abs(m["abs_error"]) <= rayleigh)
    mean_pct = (float(np.mean([abs(m["pct_error"]) for m in matches[:n_score]]))
                if n_score else float("nan"))

    # Forward bridge check: correlate measured fluctuation against the
    # truncated explicit-formula cascade built from the true zeros.
    f_model = explicit_formula_cascade(u, gammas)
    fm0 = f_model - f_model.mean()
    denom = np.linalg.norm(f0) * np.linalg.norm(fm0)
    correlation = float(np.dot(f0, fm0) / denom) if denom > 0 else float("nan")

    # Mandated Systrophe address-space novelty catcher on the spectrum.
    n_catch = 96
    catch_freqs = np.linspace(gamma_lo, gamma_hi, n_catch)
    catch_power = np.interp(catch_freqs, freqs, power)

    def spectrum_fn(g_val: float) -> np.ndarray:
        i = int(np.argmin(np.abs(catch_freqs - g_val)))
        return np.array([float(catch_power[i])])

    catch = scan_novelty(catch_freqs, spectrum_fn, n_bits=32,
                         parameter_label="gamma")
    catch_sharp_gammas = [round(float(s["parameter_value"]), 2)
                          for s in catch.sharp_features]

    return {
        "x_max": int(x_max),
        "x_min": float(x_min),
        "u_span": float(span),
        "rayleigh_resolution_gamma": float(rayleigh),
        "n_samples": int(n_samples),
        "n_prime_power_steps": int(len(pos)),
        "freq_grid": [float(gamma_lo), float(gamma_hi), int(n_freq)],
        "recovered_peaks_top10": [round(float(x), 4)
                                  for x in peak_freqs[:10]],
        "matches": matches,
        "hits_within_rayleigh_first6": int(hits),
        "mean_abs_pct_error_first6": mean_pct,
        "explicit_formula_correlation": correlation,
        "catcher_verdict": catch.verdict,
        "catcher_n_sharp": len(catch.sharp_features),
        "catcher_sharp_at_gamma": catch_sharp_gammas,
        "catcher_summary": summarize_novelty_for_report(catch),
    }


def main() -> None:
    import sys
    try:  # the catcher summary contains the lambda-2 glyph
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x-max", type=float, default=1e7,
                    help="upper limit X for psi(x) (default 1e7)")
    ap.add_argument("--x-min", type=float, default=1000.0,
                    help="lower x of the analysis window (default 1000)")
    ap.add_argument("--n-zeros", type=int, default=10,
                    help="number of true zeta zeros to score against")
    ap.add_argument("--n-samples", type=int, default=20000,
                    help="number of u = ln x sample points")
    ap.add_argument("--gamma-lo", type=float, default=5.0)
    ap.add_argument("--gamma-hi", type=float, default=45.0)
    ap.add_argument("--n-freq", type=int, default=8000)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run at X = 1e6")
    args = ap.parse_args()

    x_max = int(1e6) if args.quick else int(args.x_max)

    print("=" * 72)
    print("Recovering zeta zeros from the PRIMES via Systrophe DSI / "
          "log-periodic spectroscopy")
    print("=" * 72)
    print(f"  X (psi upper limit)  = {x_max:,}")
    print(f"  analysis window x in [{args.x_min:g}, {x_max:g}]  "
          f"(u = ln x span)")
    print()

    result = run(
        x_max=x_max,
        n_zeros=args.n_zeros,
        n_samples=args.n_samples,
        x_min=args.x_min,
        gamma_lo=args.gamma_lo,
        gamma_hi=args.gamma_hi,
        n_freq=args.n_freq,
    )

    print(f"  prime-power steps used : {result['n_prime_power_steps']:,}")
    print(f"  u-span                 : {result['u_span']:.3f}")
    print(f"  Rayleigh resolution    : dgamma = "
          f"{result['rayleigh_resolution_gamma']:.3f}")
    print()
    print("  Recovered spectral peaks vs true zeta zeros")
    print("  " + "-" * 60)
    print(f"  {'k':>2}  {'gamma_true':>11}  {'recovered':>11}  "
          f"{'pct_err':>8}")
    for m in result["matches"]:
        print(f"  {m['k']:>2}  {m['gamma_true']:>11.4f}  "
              f"{m['gamma_recovered']:>11.4f}  {m['pct_error']:>7.2f}%")
    print()
    print(f"  hits within Rayleigh (first 6)   : "
          f"{result['hits_within_rayleigh_first6']}/6")
    print(f"  mean |pct error| (first 6)       : "
          f"{result['mean_abs_pct_error_first6']:.2f}%")
    print(f"  explicit-formula correlation     : "
          f"{result['explicit_formula_correlation']:.4f}")
    print()
    print(f"  [catcher] {result['catcher_summary']}")
    print(f"  [catcher] sharp features at gamma = "
          f"{result['catcher_sharp_at_gamma']}")
    print()

    out_path = Path(__file__).parent / (
        f"millennium_primes_dsi_inverse_x{x_max}_results.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
