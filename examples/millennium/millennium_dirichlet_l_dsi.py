"""Dirichlet L-function zeros from primes-in-progressions, plus the
Chebyshev prime-race signature -- the character-twisted generalisation of
the zeta-only ``millennium_primes_dsi_inverse.py``.

Two experiments, one script.

(A) L-function comb from twisted primes
---------------------------------------
Clone the zeta prime-DSI inverse, but multiply each prime weight ``log p``
by a real Dirichlet character chi(p) (chi_3 mod 3, chi_4 mod 4). Form the
twisted Chebyshev function and its normalised fluctuation

    psi(x; chi) = sum_{p^k <= x} chi(p^k) log p,
    f_chi(u)    = psi(e^u; chi) / e^{u/2},   u = ln x.

Because L(s, chi) is ENTIRE (no pole at s = 1, unlike zeta), there is NO
``x`` main term and NO log corrections to subtract -- the twisted psi is
already a pure oscillation about zero, so the resulting comb is cleaner
than zeta's. A Lomb-Scargle periodogram in u recovers the imaginary parts
gamma of the L(s, chi) zeros, and the chi_3 and chi_4 combs are DISJOINT
from each other and from the zeta 14.13 fundamental.

(B) Chebyshev prime race
------------------------
Build the prime count pi(x; q, a) by residue class a mod q and race the
quadratic non-residue class against the residue class. Under the
LOGARITHMIC density measure (uniform in u = ln x; the Rubinstein-Sarnak
measure, NOT pointwise-uniform in x), the non-residue class leads almost
all of the time -- Chebyshev's bias. The one-sided log-density lead
fraction is calibrated against a residue-LABEL permutation null.

Acceptance gates (all reported with actual numbers in the result JSON):
  1. f_chi(u) correlates with the truncated explicit-formula cascade built
     from the TRUE L(s, chi) zeros (zeta version hit ~0.86).
  2. chi_3 and chi_4 return DISJOINT combs and do NOT reproduce the zeta
     14.13 peak.
  3. Phase-randomising f_chi destroys the comb (peak correlation collapses).
  4. scan_novelty on each comb returns 'novel_structure'.
  5. Race: one-sided log-density lead fraction is significantly biased vs a
     residue-label permutation null (report bias and p).

Usage
-----
    python examples/millennium/millennium_dirichlet_l_dsi.py            # X = 1e7
    python examples/millennium/millennium_dirichlet_l_dsi.py --x-max 1e8
    python examples/millennium/millennium_dirichlet_l_dsi.py --quick    # X = 1e6
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, lombscargle

from systrophe.catchers.dirichlet_l_dsi import (
    dirichlet_cascade,
    low_zeros,
    make_class_psi,
    make_twisted_psi,
    primes_up_to,
    twisted_fluctuation,
)
from systrophe.catchers.novelty_catcher import (
    scan_novelty,
    summarize_novelty_for_report,
)

#: The first zeta zero, for the "do NOT reproduce 14.13" disjointness check.
ZETA_FIRST_ZERO = 14.134725


def lombscargle_chunked(u: np.ndarray, y: np.ndarray, freqs: np.ndarray,
                        chunk: int = 1000) -> np.ndarray:
    """scipy.signal.lombscargle evaluated in frequency chunks.

    Same helper as ``millennium_primes_dsi_inverse``: chunking the
    frequency axis bounds peak memory on long signals to n_samples * chunk.
    """
    out = np.empty(len(freqs), dtype=float)
    for i in range(0, len(freqs), chunk):
        sl = slice(i, i + chunk)
        out[sl] = lombscargle(u, y, freqs[sl], normalize=True)
    return out


# --------------------------------------------------------------------------
# (A) L-function comb
# --------------------------------------------------------------------------

def comb_experiment(q: int, x_max: int, x_min: float, n_samples: int,
                    gamma_lo: float, gamma_hi: float, n_freq: int,
                    seed: int = 0) -> dict:
    """Recover the L(s, chi_q) zeros from the chi_q-twisted primes."""
    u = np.linspace(math.log(x_min), math.log(x_max), n_samples)

    psi = make_twisted_psi(x_max, q)
    f = twisted_fluctuation(psi, u)
    f0 = f - f.mean()

    # Lomb-Scargle periodogram in angular frequency = gamma.
    freqs = np.linspace(gamma_lo, gamma_hi, n_freq)
    power = lombscargle_chunked(u, f0, freqs)

    gammas = low_zeros(q)

    # Recovered peaks: prominent local maxima of the periodogram.
    peak_idx, _ = find_peaks(power, prominence=power.max() * 0.05)
    peak_freqs = freqs[peak_idx]
    peak_powers = power[peak_idx]
    order = np.argsort(peak_powers)[::-1]
    peak_freqs = peak_freqs[order]
    peak_powers = peak_powers[order]

    # Match each true zero to its nearest recovered peak.
    span = float(u.max() - u.min())
    rayleigh = 2.0 * math.pi / span
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
    n_score = len(matches)
    hits = sum(1 for m in matches if abs(m["abs_error"]) <= rayleigh)
    mean_pct = (float(np.mean([abs(m["pct_error"]) for m in matches]))
                if n_score else float("nan"))

    # Gate 1: forward bridge correlation against the explicit-formula cascade
    # built from the TRUE L(s, chi) zeros.
    f_model = dirichlet_cascade(u, gammas)
    fm0 = f_model - f_model.mean()
    denom = np.linalg.norm(f0) * np.linalg.norm(fm0)
    correlation = float(np.dot(f0, fm0) / denom) if denom > 0 else float("nan")

    # Gate 2 (partial): does the comb have power at the zeta fundamental?
    # Compare the periodogram value at 14.13 to the median; a true L-comb
    # has NO zeta peak there.
    zeta_power = float(np.interp(ZETA_FIRST_ZERO, freqs, power))
    median_power = float(np.median(power))
    zeta_peak_ratio = zeta_power / median_power if median_power > 0 else 0.0
    # The dominant recovered fundamental (top peak) and its distance to 14.13.
    top_fundamental = float(peak_freqs[0]) if len(peak_freqs) else float("nan")
    dist_to_zeta = abs(top_fundamental - ZETA_FIRST_ZERO)

    # Gate 3: phase-randomise the prime fluctuation -> comb must vanish. We
    # phase-randomise in the u-spectrum (preserve power spectrum, destroy
    # phase relationships), then recompute the cascade correlation.
    rng = np.random.default_rng(seed)
    F = np.fft.rfft(f0)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, len(F)))
    phases[0] = 1.0
    f_surr = np.fft.irfft(np.abs(F) * phases, n=len(f0))
    fs0 = f_surr - f_surr.mean()
    denom_s = np.linalg.norm(fs0) * np.linalg.norm(fm0)
    surr_correlation = (float(np.dot(fs0, fm0) / denom_s)
                        if denom_s > 0 else float("nan"))

    # Gate 4: mandated address-space novelty catcher on the spectrum.
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
        "q": int(q),
        "x_max": int(x_max),
        "u_span": span,
        "rayleigh_resolution_gamma": float(rayleigh),
        "true_zeros": [float(g) for g in gammas],
        "recovered_peaks_top8": [round(float(x), 4) for x in peak_freqs[:8]],
        "top_fundamental": top_fundamental,
        "matches": matches,
        "hits_within_rayleigh": int(hits),
        "n_scored": int(n_score),
        "mean_abs_pct_error": mean_pct,
        "explicit_formula_correlation": correlation,
        "phase_randomized_correlation": surr_correlation,
        "zeta_fundamental_power_ratio": zeta_peak_ratio,
        "dist_top_fundamental_to_zeta": float(dist_to_zeta),
        "catcher_verdict": catch.verdict,
        "catcher_n_sharp": len(catch.sharp_features),
        "catcher_sharp_at_gamma": catch_sharp_gammas,
        "catcher_summary": summarize_novelty_for_report(catch),
    }


# --------------------------------------------------------------------------
# (B) Chebyshev prime race (Rubinstein-Sarnak log-density statistic)
# --------------------------------------------------------------------------

def race_experiment(q: int, nonres: int, res: int, x_max: int, x_min: float,
                    n_samples: int, n_perm: int = 400, seed: int = 0) -> dict:
    """One-sided log-density lead fraction of the quadratic non-residue
    class vs the residue class, calibrated against a residue-label
    permutation null.

    Statistic (Rubinstein-Sarnak style, NOT pointwise sign): the fraction
    of the LOGARITHMIC measure (uniform in u = ln x) on which
    pi(x; q, nonres) > pi(x; q, res). Chebyshev's bias => > 1/2.

    Null: randomly relabel each prime's residue class among the coprime
    classes (a permutation of residue labels), then recompute the lead
    fraction. Under the null the bias collapses to ~1/2.
    """
    primes = primes_up_to(x_max)
    rp = primes % q
    coprime_classes = [a for a in range(1, q) if math.gcd(a, q) == 1]
    coprime_mask = np.isin(rp, coprime_classes)
    p_co = primes[coprime_mask]
    r_co = rp[coprime_mask]

    u = np.linspace(math.log(x_min), math.log(x_max), n_samples)
    xs = np.exp(u)

    def lead_fraction(residues: np.ndarray) -> float:
        non = p_co[residues == nonres]
        rr = p_co[residues == res]
        cN = np.searchsorted(non, xs, side="right")
        cR = np.searchsorted(rr, xs, side="right")
        return float(np.mean((cN - cR) > 0))

    observed = lead_fraction(r_co)

    # Permutation null: shuffle the residue labels across the coprime primes.
    rng = np.random.default_rng(seed)
    null_fracs = np.empty(n_perm)
    for i in range(n_perm):
        null_fracs[i] = lead_fraction(rng.permutation(r_co))
    null_mean = float(np.mean(null_fracs))
    null_std = float(np.std(null_fracs))
    bias = observed - null_mean
    # One-sided p-value: how often the null lead fraction is >= observed.
    p_value = float((np.sum(null_fracs >= observed) + 1) / (n_perm + 1))
    z = float(bias / null_std) if null_std > 0 else float("inf")

    return {
        "q": int(q),
        "nonresidue_class": int(nonres),
        "residue_class": int(res),
        "x_max": int(x_max),
        "observed_lead_fraction": observed,
        "null_mean_lead_fraction": null_mean,
        "null_std": null_std,
        "bias": float(bias),
        "z_vs_label_null": z,
        "p_value_one_sided": p_value,
        "n_perm": int(n_perm),
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run(x_max: int, x_min: float, n_samples: int, gamma_lo: float,
        gamma_hi: float, n_freq: int, n_perm: int, seed: int = 0) -> dict:
    combs = {}
    for q in (4, 3):
        combs[str(q)] = comb_experiment(
            q, x_max, x_min, n_samples, gamma_lo, gamma_hi, n_freq, seed=seed)

    # Disjointness: smallest pairwise distance between the recovered top
    # fundamentals of the two combs (should be well-separated).
    top4 = combs["4"]["top_fundamental"]
    top3 = combs["3"]["top_fundamental"]
    comb_separation = abs(top4 - top3)

    races = {
        "4": race_experiment(4, 3, 1, x_max, x_min, n_samples,
                             n_perm=n_perm, seed=seed),
        "3": race_experiment(3, 2, 1, x_max, x_min, n_samples,
                             n_perm=n_perm, seed=seed),
    }

    return {
        "x_max": int(x_max),
        "x_min": float(x_min),
        "combs": combs,
        "comb_fundamental_separation": float(comb_separation),
        "races": races,
    }


def main() -> None:
    import sys
    try:  # the catcher summary contains the lambda-2 glyph
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x-max", type=float, default=1e7)
    ap.add_argument("--x-min", type=float, default=1000.0)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--gamma-lo", type=float, default=3.0)
    ap.add_argument("--gamma-hi", type=float, default=25.0)
    ap.add_argument("--n-freq", type=int, default=6000)
    ap.add_argument("--n-perm", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="fast X = 1e6")
    args = ap.parse_args()

    x_max = int(1e6) if args.quick else int(args.x_max)

    print("=" * 74)
    print("Dirichlet L-function zeros from twisted primes + Chebyshev race")
    print("=" * 74)
    print(f"  X = {x_max:,}   analysis window x in [{args.x_min:g}, {x_max:g}]")
    print()

    result = run(x_max, args.x_min, args.n_samples, args.gamma_lo,
                 args.gamma_hi, args.n_freq, args.n_perm, seed=args.seed)

    for q in ("4", "3"):
        c = result["combs"][q]
        print(f"  --- chi_{q} comb (L(s, chi_{q}) zeros) ---")
        print(f"  true zeros          : "
              f"{[round(g, 3) for g in c['true_zeros']]}")
        print(f"  recovered top peaks : {c['recovered_peaks_top8']}")
        print(f"  hits within Rayleigh: {c['hits_within_rayleigh']}"
              f"/{c['n_scored']}   mean|pct err| = "
              f"{c['mean_abs_pct_error']:.2f}%")
        print(f"  cascade correlation : "
              f"{c['explicit_formula_correlation']:.4f}   "
              f"(phase-rand surrogate: "
              f"{c['phase_randomized_correlation']:.4f})")
        print(f"  zeta-14.13 power/median = "
              f"{c['zeta_fundamental_power_ratio']:.2f}   "
              f"top-fundamental dist to 14.13 = "
              f"{c['dist_top_fundamental_to_zeta']:.2f}")
        print(f"  [catcher] {c['catcher_summary']}")
        print()

    print(f"  chi_4 vs chi_3 fundamental separation = "
          f"{result['comb_fundamental_separation']:.3f}")
    print()

    print("  --- Chebyshev prime race (Rubinstein-Sarnak log-density) ---")
    for q in ("4", "3"):
        r = result["races"][q]
        print(f"  q={q}: pi({r['nonresidue_class']}) vs "
              f"pi({r['residue_class']})  observed lead = "
              f"{r['observed_lead_fraction']:.4f}  "
              f"null = {r['null_mean_lead_fraction']:.4f}  "
              f"bias = {r['bias']:+.4f}  z = {r['z_vs_label_null']:.1f}  "
              f"p = {r['p_value_one_sided']:.4g}")
    print()

    out_path = Path(__file__).parent / (
        f"millennium_dirichlet_l_dsi_x{x_max}_results.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
