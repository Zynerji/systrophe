"""Re-run the Millennium-problem catcher suite with the v2 detectors
(coherence + consensus + multi-run) to see how many additional
emergents fire.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

from systrophe.catchers.catcher_v2 import (
    scan_novelty_coherent,
    scan_novelty_consensus,
    scan_novelty_multirun,
)


# ============================================================
# Re-test: 3-SAT phase transition (was caught by derivative catcher
# at alpha_c = 4.270 in v1; can coherence catch it natively?)
# ============================================================

def sat_phase_data():
    """Generate the SAT phase-transition data as a 2D output:
    each parameter alpha returns (P_SAT, median_decisions/100, p90_runtime).
    """
    # Cached from earlier results:
    cached = json.loads(
        Path("examples/millennium_sat_phase_transition_results.json").read_text()
    )
    alpha_grid = np.array(cached["alpha_grid"])
    per_alpha = cached["per_alpha"]
    rows = []
    for r in per_alpha:
        rows.append([
            r["sat_fraction"],
            r["median_decisions"] / 100,
            r["p90_runtime_s"] * 100,
            r["median_runtime_s"] * 100,
            r["p90_decisions"] / 100,
        ])
    M = np.array(rows)

    def fn(alpha_val):
        idx = int(np.argmin(np.abs(alpha_grid - alpha_val)))
        return M[idx]
    return alpha_grid, fn


def recapture_sat():
    print("=" * 70)
    print("SAT phase transition: v2 catcher re-run")
    print("=" * 70)
    alpha_grid, fn = sat_phase_data()

    # Variant 1: coherence
    coh = scan_novelty_coherent(alpha_grid, fn, sharp_threshold=0.20,
                                  min_components=3)
    print(f"  coherence: verdict={coh.verdict}, "
          f"max_coh={coh.max_coherence:.3f} at alpha={coh.max_coherence_parameter:.3f}")
    print(f"  coherence sharps: {[(s['parameter_value'], s['coherence']) for s in coh.sharp_features]}")

    # Variant 2: consensus
    cons = scan_novelty_consensus(alpha_grid, fn, z_threshold=2.5,
                                    consensus_fraction=0.50)
    print(f"  consensus: verdict={cons.verdict}, "
          f"max_cons={cons.max_consensus:.3f} at alpha={cons.max_consensus_parameter:.3f}")
    print(f"  consensus sharps: {[(s['parameter_value'], s['consensus_fraction']) for s in cons.sharp_features]}")

    return {"coherence": coh.verdict, "consensus": cons.verdict,
             "best_coherence_alpha": coh.max_coherence_parameter,
             "best_consensus_alpha": cons.max_consensus_parameter}


# ============================================================
# Re-test: Goldbach 3-band comet
# ============================================================

def goldbach_data():
    cached = json.loads(
        Path("examples/millennium_goldbach_catcher_n1000_results.json").read_text()
    )
    return cached


def recapture_goldbach():
    print("=" * 70)
    print("Goldbach: v2 catcher re-run")
    print("=" * 70)
    # Recompute g(n) since we need full data
    import sys
    sys.path.insert(0, "examples")
    from millennium_goldbach_catcher import compute_goldbach_comet

    data = compute_goldbach_comet(n_max=500)
    evens = np.array(data["evens"])
    g_values = data["g_values"]

    # Build a vector output per n: [g(n), g(n) by mod 6 class indicator]
    def fn(n_val):
        idx = int(np.argmin(np.abs(evens - n_val)))
        n = int(evens[idx])
        g = g_values[idx]
        return np.array([g, g if n % 6 == 0 else 0,
                         g if n % 6 == 2 else 0,
                         g if n % 6 == 4 else 0])

    coh = scan_novelty_coherent(evens.astype(float), fn,
                                  sharp_threshold=0.20, min_components=3)
    print(f"  coherence: verdict={coh.verdict}, max_coh={coh.max_coherence:.3f}")

    cons = scan_novelty_consensus(evens.astype(float), fn,
                                    z_threshold=2.5, consensus_fraction=0.30)
    print(f"  consensus: verdict={cons.verdict}, max_cons={cons.max_consensus:.3f}")

    return {"coherence": coh.verdict, "consensus": cons.verdict}


# ============================================================
# Re-test: BSD per-curve z-score within rank
# ============================================================

def recapture_bsd():
    print("=" * 70)
    print("BSD: per-curve z-score within rank class")
    print("=" * 70)
    cached = json.loads(
        Path("examples/millennium_bsd_catcher_results.json").read_text()
    )

    # We previously got mean by rank: 0->-1.39, 1->-0.87, 2->-2.22
    # The original test used the per-rank distribution. With consensus,
    # we ask: for each curve, is its log-L 2.5sigma above/below the
    # whole-curve median?
    per_curve = cached["per_curve"]
    log_Ls = np.array([c["log_L_final"] for c in per_curve])
    ranks = np.array([c["expected_rank"] for c in per_curve])
    print(f"  Per-curve log_L (sorted): {sorted(log_Ls)}")

    # If we view log_L as a 1D output and treat each curve as a
    # parameter index, can consensus identify the rank-2 outlier?
    parameter_axis = np.arange(len(log_Ls)).astype(float)

    def fn(idx_float):
        i = int(round(idx_float))
        i = max(0, min(len(log_Ls) - 1, i))
        # Return log_L plus its rank indicator vector
        rank_onehot = np.zeros(3)
        rank_onehot[ranks[i]] = 1
        return np.array([log_Ls[i], log_Ls[i] * rank_onehot[0],
                         log_Ls[i] * rank_onehot[1], log_Ls[i] * rank_onehot[2]])

    cons = scan_novelty_consensus(parameter_axis, fn,
                                    z_threshold=2.0, consensus_fraction=0.20,
                                    baseline_quartiles=(0.30, 0.30))
    print(f"  consensus: verdict={cons.verdict}, max_cons={cons.max_consensus:.3f}")
    print(f"  consensus sharps: {cons.sharp_features}")

    return {"consensus": cons.verdict}


# ============================================================
# Re-test: Burgers' shock formation with multi-band consensus
# ============================================================

def recapture_burgers():
    print("=" * 70)
    print("Burgers' shock formation: per-band consensus")
    print("=" * 70)
    cached = json.loads(
        Path("examples/millennium_burgers_shock_catcher_results.json").read_text()
    )

    # For nu=0.005, take the |u_x|_max series across time
    nu_key = "nu_0.005"
    if nu_key not in cached:
        print(f"  {nu_key} not found, trying others")
        nu_key = list(cached.keys())[0]
    data = cached[nu_key]
    # The cached JSON has 'simulation' (sub-dict without ux_max_series) and 'detection'
    # Need to re-simulate to get the ux_max trajectory + multi-band reconstruction
    # Skip the multi-band part and just use the scalar series in 'detection'
    if "detection" in data and "max_ux_overall" in data["detection"]:
        # We don't have the trajectory cached; skip the multi-band test
        # and report the cached scalar result
        print(f"  Scalar trajectory not cached; multi-band test deferred.")
        return {"consensus": "skipped"}

    return {"consensus": "skipped"}


# ============================================================
# Multi-run on Riemann GUE null
# ============================================================

def recapture_riemann_multirun():
    print("=" * 70)
    print("Riemann: multi-run coincidence on GUE null")
    print("=" * 70)
    from mpmath import mp, zetazero
    mp.dps = 15
    # Get first 200 zeros (5 sub-runs of 40 each)
    print("  fetching 200 zeta zeros (~15s)...")
    all_zeros = np.array([float(zetazero(k).imag) for k in range(1, 201)])
    spacings = np.diff(all_zeros)
    t_c = (all_zeros[:-1] + all_zeros[1:]) / 2
    mean_local = 2 * math.pi / np.log(t_c / (2 * math.pi))
    s_norm = spacings / mean_local

    # Build 5 sub-runs, each a window of 40 spacings from different offsets
    parameter_axis = np.arange(40).astype(float)
    runs = []
    for offset in (0, 20, 40, 80, 120):
        sub = s_norm[offset:offset + 40]
        def make_fn(sub_data):
            def fn(idx_float):
                i = int(round(idx_float))
                i = max(0, min(len(sub_data) - 1, i))
                # Return a 4-component vector: the spacing + 3 lagged versions
                lagged = [sub_data[i]]
                for lag in (1, 2, 3):
                    j = max(0, i - lag)
                    lagged.append(sub_data[j])
                return np.array(lagged)
            return fn
        runs.append(make_fn(sub))

    multi = scan_novelty_multirun(parameter_axis, runs,
                                    per_run_z_threshold=2.0,
                                    coincidence_fraction=0.60,
                                    parameter_tolerance=1)
    print(f"  multi-run: verdict={multi.verdict}, "
          f"max_coincidence={multi.max_coincidence:.3f}, n_runs={multi.n_runs}")
    print(f"  sharps: {[s['parameter_value'] for s in multi.sharp_features][:5]}")

    return {"multirun": multi.verdict, "max_coinc": multi.max_coincidence}


def main():
    results = {}
    try:
        results["sat"] = recapture_sat()
    except Exception as e:
        results["sat"] = {"error": str(e)}
        print(f"  SAT error: {e}")

    try:
        results["goldbach"] = recapture_goldbach()
    except Exception as e:
        results["goldbach"] = {"error": str(e)}
        print(f"  Goldbach error: {e}")

    try:
        results["bsd"] = recapture_bsd()
    except Exception as e:
        results["bsd"] = {"error": str(e)}
        print(f"  BSD error: {e}")

    try:
        results["riemann"] = recapture_riemann_multirun()
    except Exception as e:
        results["riemann"] = {"error": str(e)}
        print(f"  Riemann error: {e}")

    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")

    out = Path(__file__).parent / "millennium_v2_recapture_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
