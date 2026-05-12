"""Millennium-problem exploration: P vs NP via the SAT phase transition.

The 3-SAT problem has a well-established phase transition at
    alpha_c ~ 4.267  (where alpha = m / n, m clauses, n variables).

For alpha < alpha_c, random 3-SAT instances are typically satisfiable
and easy to solve. For alpha > alpha_c, instances are typically
unsatisfiable. Around alpha_c, solver runtime peaks sharply (the
"easy-hard-easy" pattern) and is conjecturally related to the P vs NP
threshold for typical-case 3-SAT.

This script runs the Systrophe address-space novelty catcher on the
SAT/UNSAT verdict distribution as a function of alpha. The catcher
should:

  - report SMOOTH below alpha_c (all SAT, no novelty)
  - report SHARP TRANSITION at alpha ~ 4.267 (sat fraction collapses)
  - report SMOOTH above alpha_c (all UNSAT, no novelty)

If we recover alpha_c to within the catcher's grid spacing, the
catcher has independently rediscovered the SAT phase boundary from
address-space novelty alone -- a P vs NP-adjacent computational
phenomenon detected as catcher novelty.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from systrophe.novelty_catcher import (
    catch_novelty_in_named_arrays,
    real_array_to_address,
    scan_novelty,
)
from systrophe.derivative_catcher import catch_smooth_transition


def random_3sat(n_vars: int, n_clauses: int, rng: random.Random) -> list[tuple[int, int, int]]:
    """Generate a uniformly random 3-SAT instance.

    Each clause is a tuple of three signed literals in {-n..-1, 1..n}.
    """
    clauses = []
    for _ in range(n_clauses):
        vars_chosen = rng.sample(range(1, n_vars + 1), 3)
        clause = tuple(v if rng.random() < 0.5 else -v for v in vars_chosen)
        clauses.append(clause)
    return clauses


def dpll_sat(
    clauses: list[tuple[int, ...]], n_vars: int, time_budget_s: float = 1.0,
) -> tuple[bool | None, int, float]:
    """Naive DPLL with unit propagation. Returns (sat, n_decisions, runtime_s).

    `sat = None` means we ran out of time budget (treat as unknown).
    Suitable for n_vars <= 30 with a 1-second budget.
    """
    t0 = time.perf_counter()
    decisions = 0

    def simplify(cs: list[tuple[int, ...]], lit: int) -> list[tuple[int, ...]] | None:
        out = []
        for c in cs:
            if lit in c:
                continue  # clause satisfied
            new_c = tuple(x for x in c if x != -lit)
            if not new_c:
                return None  # empty clause -> unsat
            out.append(new_c)
        return out

    def solve(cs: list[tuple[int, ...]]) -> bool | None:
        nonlocal decisions
        if time.perf_counter() - t0 > time_budget_s:
            return None
        if not cs:
            return True
        # Unit propagation
        for c in cs:
            if len(c) == 1:
                cs2 = simplify(cs, c[0])
                if cs2 is None:
                    return False
                return solve(cs2)
        decisions += 1
        # Branch on first variable in first clause
        lit = cs[0][0]
        for sign in (1, -1):
            cs_branch = simplify(cs, sign * abs(lit))
            if cs_branch is None:
                continue
            r = solve(cs_branch)
            if r is True:
                return True
            if r is None:
                return None
        return False

    result = solve(clauses)
    return result, decisions, time.perf_counter() - t0


def measure_sat_fraction(
    n_vars: int, alpha: float, n_instances: int = 60,
    seed: int = 11,
) -> dict:
    """Measure SAT fraction and median runtime at given alpha."""
    rng = random.Random(seed + int(alpha * 1000))
    m = int(round(alpha * n_vars))
    sats = 0
    unsats = 0
    unknowns = 0
    runtimes = []
    decision_counts = []
    for _ in range(n_instances):
        instance = random_3sat(n_vars, m, rng)
        verdict, decisions, runtime = dpll_sat(instance, n_vars, time_budget_s=2.0)
        runtimes.append(runtime)
        decision_counts.append(decisions)
        if verdict is True:
            sats += 1
        elif verdict is False:
            unsats += 1
        else:
            unknowns += 1
    return {
        "alpha": alpha,
        "n_vars": n_vars,
        "m_clauses": m,
        "n_instances": n_instances,
        "sat_fraction": sats / n_instances,
        "unsat_fraction": unsats / n_instances,
        "unknown_fraction": unknowns / n_instances,
        "median_runtime_s": float(np.median(runtimes)),
        "median_decisions": float(np.median(decision_counts)),
        "p90_runtime_s": float(np.percentile(runtimes, 90)),
        "p90_decisions": float(np.percentile(decision_counts, 90)),
    }


def run_phase_transition(
    n_vars: int = 20,
    alpha_grid: np.ndarray | None = None,
    n_instances: int = 60,
) -> dict:
    if alpha_grid is None:
        alpha_grid = np.array([
            2.0, 2.5, 3.0, 3.3, 3.6, 3.9,
            4.0, 4.1, 4.2, 4.27, 4.35, 4.5,
            4.8, 5.1, 5.5, 6.0,
        ])

    print(f"Running 3-SAT phase transition at n={n_vars}, "
          f"{n_instances} instances per alpha-point...")
    results = []
    for alpha in alpha_grid:
        r = measure_sat_fraction(n_vars, float(alpha), n_instances=n_instances)
        results.append(r)
        print(f"  alpha={alpha:.3f}  P(SAT)={r['sat_fraction']:.3f}  "
              f"P(UNSAT)={r['unsat_fraction']:.3f}  "
              f"P(unknown)={r['unknown_fraction']:.3f}  "
              f"median_decisions={r['median_decisions']:.0f}  "
              f"p90_runtime={r['p90_runtime_s']:.3f}s")

    # Apply scan_novelty across alpha-axis using SAT fraction + median runtime
    def fn(alpha_val):
        idx = int(np.argmin(np.abs(alpha_grid - alpha_val)))
        r = results[idx]
        return np.array([r["sat_fraction"], np.log10(r["median_decisions"] + 1)])

    scan_adaptive = scan_novelty(alpha_grid, fn, n_bits=32, data_adaptive=True)
    scan_binned = scan_novelty(alpha_grid, fn, n_bits=32, data_adaptive=False)

    # Derivative catcher: scalar SAT fraction vs alpha
    sat_frac_lookup = {r["alpha"]: r["sat_fraction"] for r in results}
    def sat_frac_fn(alpha_val):
        idx = int(np.argmin(np.abs(alpha_grid - alpha_val)))
        return results[idx]["sat_fraction"]
    deriv_result = catch_smooth_transition(alpha_grid, sat_frac_fn, n_bits=32)

    def _sharps_to_json(sharp_features):
        return [
            {k: int(v) if isinstance(v, np.integer) else v for k, v in s.items()}
            for s in sharp_features
        ]

    return {
        "n_vars": n_vars,
        "alpha_grid": alpha_grid.tolist(),
        "per_alpha": results,
        "scan_novelty_data_adaptive": {
            "verdict": scan_adaptive.verdict,
            "sharp_features": _sharps_to_json(scan_adaptive.sharp_features),
        },
        "scan_novelty_per_output_binning": {
            "verdict": scan_binned.verdict,
            "sharp_features": _sharps_to_json(scan_binned.sharp_features),
        },
        "derivative_catcher": {
            "kind": deriv_result["kind"],
            "estimated_transition_centre": deriv_result["estimated_transition_centre"],
            "value_verdict": deriv_result["value_scan"].verdict,
            "derivative_verdict": deriv_result["derivative_scan"].verdict,
            "derivative_sharp_features": _sharps_to_json(
                deriv_result["derivative_scan"].sharp_features
            ),
        },
    }


def main() -> None:
    print("=" * 70)
    print("3-SAT phase transition via Systrophe address-space catcher")
    print("=" * 70)
    print()

    out = run_phase_transition(n_vars=20, n_instances=60)
    print()
    for mode in ("scan_novelty_data_adaptive", "scan_novelty_per_output_binning"):
        block = out[mode]
        print(f"[{mode}] verdict={block['verdict']}, "
              f"n_sharp={len(block['sharp_features'])}")
        for sf in block["sharp_features"]:
            a_val = sf["parameter_value"]
            print(f"   sharp at alpha ~ {a_val:.3f}, "
                  f"hamming_step={sf['hamming_step']}, "
                  f"median_step={sf['median_step']:.1f}")
    dc = out["derivative_catcher"]
    print(f"[derivative_catcher] kind={dc['kind']}, "
          f"value_verdict={dc['value_verdict']}, "
          f"derivative_verdict={dc['derivative_verdict']}")
    if dc["estimated_transition_centre"] is not None:
        print(f"   estimated transition centre: alpha ~ "
              f"{dc['estimated_transition_centre']:.3f}")
    for sf in dc["derivative_sharp_features"]:
        a_val = sf["parameter_value"]
        print(f"   derivative sharp at alpha ~ {a_val:.3f}, "
              f"hamming_step={sf['hamming_step']}, "
              f"median_step={sf['median_step']:.1f}")

    out_path = Path(__file__).parent / "millennium_sat_phase_transition_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print()
    print(f"Wrote {out_path}")
    print()
    print("Interpretation")
    print("==============")
    print("  Expected: sharp transition near alpha_c ~ 4.267 (P vs NP-adjacent")
    print("  computational phase boundary of typical-case 3-SAT). The catcher")
    print("  catching this transition validates address-space novelty as a")
    print("  probe of typical-case complexity transitions.")


if __name__ == "__main__":
    main()
