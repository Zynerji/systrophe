"""The error-exponent hypothesis: read the PNT error exponent theta =
sup_rho Re(rho) straight off the primes, as the growth exponent of the
prime-fluctuation envelope.

Background
----------
The prime number theorem error term obeys |psi(x) - x| = O(x^theta) with
theta = sup over non-trivial zeros of Re(rho). RH <=> theta = 1/2. In the
normalised variable

    f(u) = (psi(e^u) - e^u + corr) / e^{u/2},   u = ln x,

an off-line zero at beta > 1/2 makes f grow like e^{(beta-1/2)u}, while
under RH f grows only like log-log-log (Littlewood) -- i.e. exponent
EXACTLY 0. So the growth exponent of the *envelope* of f is

    sigma_env = theta - 1/2 = sup_rho Re(rho) - 1/2.

Hypothesis (companion to FINDINGS_PRIMES_OFFLINE_ZERO.md): the Systrophe
growth catcher applied to the sliding-window RMS envelope of f returns
sigma_env, and on the REAL primes it is statistically `stationary`
(sigma_env ~ 0) -- the primes certifying theta = 1/2 -- with the
certified detection floor shrinking like ~1/ln X.

Unlike the per-mode off-line test, this is a SINGLE GLOBAL statistic on
the whole envelope, so it has no neighbouring-zero leakage: the floor is
set only by the ln x span. The off-line synthetic injection calibrates
that the envelope catcher does respond to theta > 1/2.

Usage
-----
    python examples/millennium_primes_envelope_exponent.py
    python examples/millennium_primes_envelope_exponent.py --x-list 1e6,1e7,1e8
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from systrophe.growth_catcher import catch_growth

from millennium_primes_dsi_inverse import (
    prime_fluctuation,
    prime_power_steps,
    zeta_zeros,
)
from millennium_primes_offline_zero_test import synthetic_cascade


def rms_envelope(u: np.ndarray, f: np.ndarray, width: float,
                 n_centers: int = 60):
    """Sliding-window RMS envelope E(u_c) = sqrt(<f^2>) over windows."""
    centers = np.linspace(u.min() + width / 2, u.max() - width / 2, n_centers)
    env = np.empty(n_centers)
    for i, uc in enumerate(centers):
        m = (u >= uc - width / 2) & (u <= uc + width / 2)
        env[i] = math.sqrt(np.mean(f[m] ** 2)) if m.sum() >= 4 else np.nan
    return centers, env


def real_prime_envelope_exponent(x_max: int, x_min: float, n_samples: int,
                                 width: float) -> dict:
    u = np.linspace(math.log(x_min), math.log(x_max), n_samples)
    pos, cum = prime_power_steps(x_max)

    def psi(x):
        x = np.asarray(x, dtype=float)
        idx = np.searchsorted(pos, x, side="right")
        return np.where(idx > 0, cum[np.clip(idx - 1, 0, len(cum) - 1)], 0.0)

    f = prime_fluctuation(psi, u)
    centers, env = rms_envelope(u, f, width)
    gc = catch_growth(centers, env, parameter_label="u")
    span = float(u.max() - u.min())
    return {
        "x_max": int(x_max),
        "u_span": span,
        "sigma_env": gc.growth_exponent,
        "growth_z": gc.significance_z,
        "null_slope_std": gc.null_slope_std,
        "floor_times_span": gc.null_slope_std * span,
        "verdict": gc.verdict,
    }


def synthetic_calibration(x_max: int, x_min: float, n_samples: int,
                          width: float, n_zeros: int,
                          betas: list[float]) -> list[dict]:
    u = np.linspace(math.log(x_min), math.log(x_max), n_samples)
    gammas = zeta_zeros(n_zeros)
    out = []
    for beta in betas:
        betas_arr = np.full(n_zeros, 0.5)
        betas_arr[0] = beta  # push the dominant (largest-amplitude) zero
        f = synthetic_cascade(u, betas_arr, gammas)
        centers, env = rms_envelope(u, f, width)
        gc = catch_growth(centers, env, parameter_label="u")
        out.append({
            "theta": float(beta),
            "sigma_env_true": float(beta - 0.5),
            "sigma_env_hat": gc.growth_exponent,
            "growth_z": gc.significance_z,
            "verdict": gc.verdict,
        })
    return out


def main() -> None:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--x-list", type=str, default="1e6,1e7,1e8")
    ap.add_argument("--x-min", type=float, default=1000.0)
    ap.add_argument("--n-samples", type=int, default=20000)
    ap.add_argument("--width", type=float, default=2.0)
    ap.add_argument("--n-zeros", type=int, default=20)
    args = ap.parse_args()
    x_list = [int(float(s)) for s in args.x_list.split(",")]

    print("=" * 72)
    print("Prime-fluctuation envelope exponent  sigma_env = theta - 1/2"
          "  (theta = sup Re rho)")
    print("=" * 72)
    print()

    print("  Part 1 -- REAL primes: envelope growth exponent vs X")
    print("  (RH => sigma_env ~ 0 'stationary'; floor*span ~ const => "
          "floor ~ 1/ln X)")
    print("  " + "-" * 66)
    print(f"  {'X':>12}  {'u_span':>7}  {'sigma_env':>10}  {'z':>6}  "
          f"{'floor':>8}  {'floor*span':>10}  {'verdict':>11}")
    real = []
    for X in x_list:
        r = real_prime_envelope_exponent(X, args.x_min, args.n_samples,
                                         args.width)
        real.append(r)
        print(f"  {X:>12,}  {r['u_span']:>7.3f}  {r['sigma_env']:>10.4f}  "
              f"{r['growth_z']:>6.1f}  {r['null_slope_std']:>8.4f}  "
              f"{r['floor_times_span']:>10.4f}  {r['verdict']:>11}")
    print()

    X_cal = x_list[-1] if x_list[-1] <= int(1e7) else int(1e7)
    print(f"  Part 2 -- synthetic calibration at X={X_cal:,}: push the "
          f"dominant zero to theta")
    print("  (envelope catcher should recover sigma_env ~ theta - 1/2)")
    print("  " + "-" * 60)
    print(f"  {'theta':>6}  {'sigma_true':>10}  {'sigma_hat':>10}  "
          f"{'z':>6}  {'verdict':>11}")
    cal = synthetic_calibration(
        X_cal, args.x_min, args.n_samples, args.width, args.n_zeros,
        betas=[0.50, 0.55, 0.60, 0.70, 0.80])
    for c in cal:
        print(f"  {c['theta']:>6.2f}  {c['sigma_env_true']:>10.3f}  "
              f"{c['sigma_env_hat']:>10.4f}  {c['growth_z']:>6.1f}  "
              f"{c['verdict']:>11}")
    print()

    out_path = Path(__file__).parent / "millennium_primes_envelope_exponent_results.json"
    out_path.write_text(json.dumps({"real_primes": real,
                                    "synthetic_calibration": cal},
                                   indent=2), encoding="utf-8")
    print(f"  Wrote {out_path}")


if __name__ == "__main__":
    main()
