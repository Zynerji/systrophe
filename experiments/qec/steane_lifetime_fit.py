"""Fit exponential decays to the Steane round sweep + bare-baseline data
and extract T_1,L (logical lifetime) and T_1,phys (physical baseline
lifetime) on ibm_kingston.

Model: P(Z = 0)(n_rounds) - 0.5 = (P_0 - 0.5) * exp(-n_rounds / tau_n)
where tau_n is the lifetime measured in round-units. Multiply by the
per-round duration to get T_1 in seconds.

For each curve (logical, bare):
  - p_inf = 0.5 (decay-to-random asymptote)
  - p_0 = pre-decay value (likely ~ 1.0 for both, since at n_rounds = 0
    the encoded state is pure)
  - tau_n = decay constant in round-units

Extracts also the BREAK-EVEN crossover: at what n_rounds (if any) does
the logical curve cross above the bare curve? Linear extrapolation if
they never cross in the measured range.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def decay_model(n, p_0, tau_n):
    """P(Z = 0; n) - 0.5 = (p_0 - 0.5) * exp(-n / tau_n)."""
    return 0.5 + (p_0 - 0.5) * np.exp(-n / tau_n)


def main() -> None:
    path = Path(__file__).parent / "results" / "steane_round_sweep_analysis.json"
    data = json.loads(path.read_text())
    paired = data["paired_by_n_rounds"]

    rounds = sorted(int(k) for k in paired.keys())
    log_rates = np.array([paired[str(n)]["steane"]["logical_zero_rate"]
                            for n in rounds])
    bare_rates = np.array([paired[str(n)]["bare"]["physical_zero_rate"]
                            for n in rounds])
    n_arr = np.array(rounds, dtype=float)

    # Shot-noise sigma
    shots = 4096
    sigma_log = np.sqrt(log_rates * (1 - log_rates) / shots)
    sigma_bare = np.sqrt(bare_rates * (1 - bare_rates) / shots)

    # Fit each curve
    p0 = [0.95, 6.0]  # initial guess: p_0 ~ 0.95, tau_n ~ 6 rounds
    bounds = ([0.5, 0.1], [1.0, 100.0])

    popt_log, pcov_log = curve_fit(
        decay_model, n_arr, log_rates, p0=p0, sigma=sigma_log,
        absolute_sigma=True, bounds=bounds, maxfev=10000,
    )
    perr_log = np.sqrt(np.diag(pcov_log))
    p0_log, tau_log = popt_log
    p0_log_s, tau_log_s = perr_log

    popt_bare, pcov_bare = curve_fit(
        decay_model, n_arr, bare_rates, p0=p0, sigma=sigma_bare,
        absolute_sigma=True, bounds=bounds, maxfev=10000,
    )
    perr_bare = np.sqrt(np.diag(pcov_bare))
    p0_bare, tau_bare = popt_bare
    p0_bare_s, tau_bare_s = perr_bare

    print("=" * 70)
    print("Steane d=3 round-sweep exponential decay fit")
    print("=" * 70)
    print()
    print(f"Logical curve:")
    print(f"  P_0 = {p0_log:.4f} +- {p0_log_s:.4f}")
    print(f"  tau_n (round-units) = {tau_log:.2f} +- {tau_log_s:.2f}")
    print()
    print(f"Bare-baseline curve:")
    print(f"  P_0 = {p0_bare:.4f} +- {p0_bare_s:.4f}")
    print(f"  tau_n (round-units) = {tau_bare:.2f} +- {tau_bare_s:.2f}")
    print()

    # Per-round duration estimate from the submitted job:
    # n_rounds=1 has duration_dt = 45800. dt for Heron-r2 is 0.5 ns.
    submitted_path = Path(__file__).parent / "results" / "steane_round_sweep_submitted.json"
    submitted = json.loads(submitted_path.read_text())
    duration_per_round_dt = (
        submitted["metadata_per_circuit"][0]["duration_dt"]  # n=1 total
    )
    # dt for ibm_kingston Heron-r2 is 0.5 ns
    dt_seconds = 0.5e-9
    duration_per_round_s = duration_per_round_dt * dt_seconds

    # The n_rounds=1 ENTIRE circuit duration includes encoding+1round
    # So per-round duration is approximately (n_8 - n_1) / 7 rounds:
    duration_n8_dt = submitted["metadata_per_circuit"][6]["duration_dt"]
    duration_n1_dt = submitted["metadata_per_circuit"][0]["duration_dt"]
    duration_per_round_dt_est = (duration_n8_dt - duration_n1_dt) / 7
    duration_per_round_s_est = duration_per_round_dt_est * dt_seconds

    print(f"Per-round duration estimate: {duration_per_round_dt_est:.0f} dt "
          f"= {duration_per_round_s_est*1e6:.2f} us")

    T_logical_s = tau_log * duration_per_round_s_est
    T_bare_s = tau_bare * duration_per_round_s_est
    print()
    print(f"T_1,L  (logical-qubit lifetime)        = {T_logical_s*1e6:.2f} us")
    print(f"T_1,phys (bare-qubit baseline T_2-eff) = {T_bare_s*1e6:.2f} us")
    print()

    # Break-even check
    if tau_log > tau_bare:
        print("Logical tau > bare tau -- if curves cross, logical wins past some n.")
        # Find n where curves cross
        n_grid = np.linspace(0.1, 100, 10000)
        log_curve = decay_model(n_grid, p0_log, tau_log)
        bare_curve = decay_model(n_grid, p0_bare, tau_bare)
        crossings = np.where(np.diff(np.sign(log_curve - bare_curve)) > 0)[0]
        if len(crossings) > 0:
            print(f"Predicted break-even n_rounds: {n_grid[crossings[0]]:.2f}")
        else:
            print("No break-even crossing in n <= 100.")
    else:
        print("Logical tau < bare tau -- logical decays FASTER than bare.")
        print("d=3 Steane is sub-threshold; no break-even crossing exists.")

    out_path = Path(__file__).parent / "results" / "steane_lifetime_fit.json"
    out_path.write_text(json.dumps({
        "logical_fit": {
            "P_0":    {"value": float(p0_log), "sigma": float(p0_log_s)},
            "tau_n":  {"value": float(tau_log), "sigma": float(tau_log_s)},
            "T_1_seconds": float(T_logical_s),
        },
        "bare_fit": {
            "P_0":    {"value": float(p0_bare), "sigma": float(p0_bare_s)},
            "tau_n":  {"value": float(tau_bare), "sigma": float(tau_bare_s)},
            "T_1_seconds": float(T_bare_s),
        },
        "per_round_duration_dt": float(duration_per_round_dt_est),
        "per_round_duration_s":  float(duration_per_round_s_est),
        "rounds_tested": rounds,
        "logical_rates": log_rates.tolist(),
        "bare_rates":    bare_rates.tolist(),
    }, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
