"""Perturbation / sensitivity test: would the Systrophe DSI lens DETECT an
off-line (Riemann-Hypothesis-violating) zero?

Companion to `millennium_primes_dsi_inverse.py`, which recovered the first
~26 zeta zeros from the primes as a log-periodic cascade. That experiment
confirms a known identity. This one asks the falsifiable question: if a
zero sat OFF the critical line, would the same machinery see it -- and
does it cry wolf on the actual (on-line) primes?

The signature of an off-line zero
---------------------------------
A zero rho = beta + i*gamma contributes -x^rho/rho to psi(x). In the
normalised fluctuation f(u) = (psi(e^u) - e^u + ...) / e^{u/2}, with
u = ln x, this is

    t(u; beta, gamma) = -2 e^{(beta - 1/2) u} [beta cos(gamma u)
                                               + gamma sin(gamma u)]
                        / (beta^2 + gamma^2)        (+ functional-eq
                                                      partner at 1-beta)

So an ON-line zero (beta = 1/2) is a *stationary* log-periodic mode of
constant amplitude in u, while an OFF-line zero (beta > 1/2) is a mode
whose amplitude GROWS as e^{(beta - 1/2) u}. The growth exponent
sigma = beta - 1/2 is the RH-violation signature, and it is directly
measurable: track the local amplitude A(u) of the mode at frequency
gamma across the ln x window and read off the slope of log A vs u.

What this script does
---------------------
1. FALSE-ALARM CHECK on the real primes: build f(u) from a sieve, and at
   each of several true zero frequencies measure the growth exponent
   sigma_hat. RH-consistency => sigma_hat ~ 0 (no growing mode). This is
   the test that must NOT cry wolf.

2. SENSITIVITY / RECOVERY on synthetic explicit-formula signals: build
   the on-line cascade of the first N zeros (control), then move ONE zero
   off the line to beta in {0.50, 0.55, 0.60, 0.70, 0.80} and check that
   sigma_hat recovers beta - 1/2, and that the mandated address-space
   novelty catcher (third-split, as in millennium_riemann_catcher) flips
   from `smooth` (on-line) to `novel_structure` (off-line).

Real primes cannot be perturbed (RH appears to hold), so part 2 is a
detector-sensitivity test on synthetic signals whose on-line periodogram
reproduces the real-prime peaks of experiment 1.

Usage
-----
    python examples/millennium_primes_offline_zero_test.py
    python examples/millennium_primes_offline_zero_test.py --x-max 1e7
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from systrophe.catchers.novelty_catcher import catch_novelty_per_quantity
from systrophe.catchers.growth_catcher import catch_growth

# Reuse the machinery from experiment 1 (same examples/ directory).
from millennium_primes_dsi_inverse import (
    prime_fluctuation,
    prime_power_steps,
    zeta_zeros,
)


# --------------------------------------------------------------------------
# Single-zero contribution to the normalised fluctuation f(u)
# --------------------------------------------------------------------------

def zero_mode(u: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Contribution of a zero at beta +- i*gamma to f(u).

    On-line (beta == 1/2): a single conjugate pair, constant amplitude.
    Off-line (beta != 1/2): the functional-equation quadruple
        {beta +- i gamma, (1-beta) +- i gamma},
    giving a growing e^{(beta-1/2)u} mode plus a decaying e^{(1/2-beta)u}
    partner.
    """
    c = np.cos(gamma * u)
    s = np.sin(gamma * u)
    if abs(beta - 0.5) < 1e-12:
        return -2.0 * (0.5 * c + gamma * s) / (0.25 + gamma ** 2)
    grow = -2.0 * np.exp((beta - 0.5) * u) * (beta * c + gamma * s) \
        / (beta ** 2 + gamma ** 2)
    b2 = 1.0 - beta
    dec = -2.0 * np.exp((b2 - 0.5) * u) * (b2 * c + gamma * s) \
        / (b2 ** 2 + gamma ** 2)
    return grow + dec


def synthetic_cascade(u: np.ndarray, betas: np.ndarray,
                      gammas: np.ndarray) -> np.ndarray:
    """Sum of zero modes -- the explicit-formula model of f(u)."""
    out = np.zeros_like(u)
    for b, g in zip(betas, gammas):
        out += zero_mode(u, float(b), float(g))
    return out


# --------------------------------------------------------------------------
# Local-amplitude tracking and growth-exponent estimate
# --------------------------------------------------------------------------

def windowed_amplitude(u: np.ndarray, f: np.ndarray, gamma: float,
                       width: float, n_centers: int = 40):
    """Local amplitude A(u_c) of the cos(gamma u) component in sliding
    windows of half-width `width/2`, via least-squares fit of
    [cos, sin, const].
    """
    centers = np.linspace(u.min() + width / 2, u.max() - width / 2, n_centers)
    amps = np.empty(n_centers)
    for i, uc in enumerate(centers):
        m = (u >= uc - width / 2) & (u <= uc + width / 2)
        uw, fw = u[m], f[m]
        if len(uw) < 6:
            amps[i] = np.nan
            continue
        M = np.column_stack([np.cos(gamma * uw), np.sin(gamma * uw),
                             np.ones_like(uw)])
        coef, *_ = np.linalg.lstsq(M, fw, rcond=None)
        amps[i] = math.hypot(coef[0], coef[1])
    return centers, amps


def growth_exponent(centers: np.ndarray, amps: np.ndarray) -> float:
    """Slope of log A vs u_c -- estimates sigma = beta - 1/2."""
    ok = np.isfinite(amps) & (amps > 0)
    if ok.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(centers[ok], np.log(amps[ok]), 1)
    return float(slope)


def catcher_third_split(amps: np.ndarray) -> str:
    """Mandated address-space catcher, third-split on the amplitude track.

    Parallels millennium_riemann_catcher: `smooth` => stationary
    (on-line / RH-consistent), `novel_structure` => the amplitude
    distribution shifts across the window (growing => off-line).
    """
    a = amps[np.isfinite(amps)]
    third = max(len(a) // 3, 1)
    per_q = {
        "amplitude": {
            "first_third":  a[:third],
            "middle_third": a[third:2 * third],
            "last_third":   a[2 * third:],
        }
    }
    return catch_novelty_per_quantity(per_q, n_bins=32)["aggregate_verdict"]


# --------------------------------------------------------------------------
# Experiment
# --------------------------------------------------------------------------

def run(x_max: int, x_min: float, n_samples: int, n_zeros: int,
        k_pert: int, betas: list[float], window: float) -> dict:
    u = np.linspace(math.log(x_min), math.log(x_max), n_samples)
    gammas = zeta_zeros(n_zeros)

    # ---- Part 1: false-alarm check on the REAL primes ----
    pos, cum = prime_power_steps(x_max)

    def psi(x):
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(pos, x, side="right")
        return np.where(idx > 0, cum[np.clip(idx - 1, 0, len(cum) - 1)], 0.0)

    f_real = prime_fluctuation(psi, u)
    real_checks = []
    for k in (1, 5, 10, min(15, n_zeros - 1)):
        g = float(gammas[k - 1])
        c, a = windowed_amplitude(u, f_real, g, window)
        gc = catch_growth(c, a, parameter_label="u")
        real_checks.append({
            "k": k,
            "gamma": g,
            "sigma_hat": gc.growth_exponent,
            "growth_z": round(gc.significance_z, 2),
            "old_catcher": catcher_third_split(a),
            "growth_catcher": gc.verdict,
        })

    # ---- Part 2: synthetic sensitivity / recovery sweep ----
    g_pert = float(gammas[k_pert - 1])
    sweep = []
    for beta in betas:
        betas_arr = np.full(n_zeros, 0.5)
        betas_arr[k_pert - 1] = beta
        f_syn = synthetic_cascade(u, betas_arr, gammas)
        c, a = windowed_amplitude(u, f_syn, g_pert, window)
        gc = catch_growth(c, a, parameter_label="u")
        sweep.append({
            "beta": float(beta),
            "sigma_true": float(beta - 0.5),
            "sigma_hat": gc.growth_exponent,
            "growth_z": round(gc.significance_z, 2),
            "old_catcher": catcher_third_split(a),
            "growth_catcher": gc.verdict,
        })

    return {
        "x_max": int(x_max),
        "x_min": float(x_min),
        "u_span": float(u.max() - u.min()),
        "n_zeros": int(n_zeros),
        "window_width_u": float(window),
        "perturbed_zero": {"k": int(k_pert), "gamma": g_pert},
        "real_prime_false_alarm_check": real_checks,
        "synthetic_offline_sweep": sweep,
    }


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x-max", type=float, default=1e7)
    ap.add_argument("--x-min", type=float, default=1000.0)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--n-zeros", type=int, default=20)
    ap.add_argument("--k-pert", type=int, default=10,
                    help="which zero (1-indexed) to push off-line")
    ap.add_argument("--window", type=float, default=4.0,
                    help="sliding-window width in u = ln x")
    args = ap.parse_args()
    x_max = int(args.x_max)

    print("=" * 72)
    print("Off-line-zero detection test (RH-violation sensitivity) via "
          "Systrophe DSI")
    print("=" * 72)
    print(f"  X = {x_max:,}   window in [{args.x_min:g}, {x_max:g}]")
    print()

    result = run(
        x_max=x_max, x_min=args.x_min, n_samples=args.n_samples,
        n_zeros=args.n_zeros, k_pert=args.k_pert,
        betas=[0.50, 0.55, 0.60, 0.70, 0.80], window=args.window,
    )

    print("  Part 1 -- false-alarm check on the REAL primes")
    print("  (RH-consistent => sigma_hat ~ 0, both catchers benign)")
    print("  " + "-" * 64)
    print(f"  {'k':>2}  {'gamma':>9}  {'sigma_hat':>10}  {'z':>6}  "
          f"{'old':>8}  {'growth':>11}")
    for r in result["real_prime_false_alarm_check"]:
        print(f"  {r['k']:>2}  {r['gamma']:>9.4f}  {r['sigma_hat']:>10.4f}  "
              f"{r['growth_z']:>6.1f}  {r['old_catcher']:>8}  "
              f"{r['growth_catcher']:>11}")
    print()

    pz = result["perturbed_zero"]
    print(f"  Part 2 -- synthetic sweep: zero k={pz['k']} "
          f"(gamma={pz['gamma']:.4f}) moved off-line")
    print("  (detector should recover sigma_hat ~ beta - 1/2)")
    print("  " + "-" * 64)
    print(f"  {'beta':>5}  {'sigma_true':>10}  {'sigma_hat':>10}  {'z':>6}  "
          f"{'old':>8}  {'growth':>11}")
    for r in result["synthetic_offline_sweep"]:
        print(f"  {r['beta']:>5.2f}  {r['sigma_true']:>10.3f}  "
              f"{r['sigma_hat']:>10.4f}  {r['growth_z']:>6.1f}  "
              f"{r['old_catcher']:>8}  {r['growth_catcher']:>11}")
    print()

    out_path = Path(__file__).parent / (
        f"millennium_primes_offline_zero_test_x{x_max}_results.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
