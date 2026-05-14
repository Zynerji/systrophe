"""Synthetic-noise A/B: naive single-boundary-flip MWPM vs Dijkstra-shortest-path.

Generates `n_shots` synthetic d=5 surface-code shots under a simple
independent X-error noise model on the data qubits, decodes each
shot with both algorithms, and reports the logical-zero rate of each.

This is not a hardware reproduction -- the Heron-r2 hardware data
from the Systrophe Heron-r2 program lives in
`Systrophe/experiments/results/surface_code_d5_dijkstra_mwpm_analysis.json`
and shows the +5-25 pp delta on real device noise. The synthetic
benchmark here verifies the algorithmic delta on a benchmark that
runs in seconds on a laptop and is reproducible without the IBM
Quantum stack.

Expected outcome: Dijkstra >= naive in logical-zero rate, with the
gap widening as the error rate p increases (more multi-qubit chains
the naive variant can't correct).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Callable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dijkstra_mwpm import (
    build_stabilizers,
    data_index,
    decode_with_dijkstra_mwpm,
    decode_with_naive_mwpm,
)


def gen_shot(d: int, p: float, n_rounds: int,
              rng: np.random.Generator) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
    """Independent X-error noise model.

    Each data qubit flips with prob p before the final measurement; the
    intermediate-round syndromes are noise-free (a 'good ancilla' model).
    Returns (final data bits, list of per-round Z-syndrome tuples).
    """
    _, Z_stabs = build_stabilizers(d)
    n_z = len(Z_stabs)
    n_data = d * d
    # Accumulate X errors over the rounds; we report the syndrome at each
    # round so the decoder can see the syndrome-difference history.
    err = np.zeros(n_data, dtype=int)
    syndromes = []
    for _ in range(n_rounds):
        new_err = (rng.random(n_data) < p / max(n_rounds + 1, 1)).astype(int)
        err = (err + new_err) % 2
        s = tuple(
            int(sum(err[q] for q in Z_stabs[i]) % 2) for i in range(n_z)
        )
        syndromes.append(s)
    # One more layer of errors before destructive measurement
    new_err = (rng.random(n_data) < p / max(n_rounds + 1, 1)).astype(int)
    err = (err + new_err) % 2
    data_bits = tuple(int(b) for b in err)
    return data_bits, syndromes


def run_arm(decoder: Callable, d: int, n_shots: int, p: float,
              n_rounds: int, rng: np.random.Generator) -> dict:
    n_zero = 0
    for _ in range(n_shots):
        data, syndromes = gen_shot(d, p, n_rounds, rng)
        log = decoder(data, syndromes, d)
        if log == 0:
            n_zero += 1
    return {"n_shots": n_shots, "n_zero": n_zero, "logical_zero_rate": n_zero / n_shots}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=5)
    parser.add_argument("--n_shots", type=int, default=400)
    parser.add_argument("--p_grid", type=float, nargs="*",
                          default=[0.005, 0.01, 0.02, 0.04, 0.08])
    parser.add_argument("--n_rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rng_seed = args.seed
    rows = []
    t_total = 0.0
    print(f"d={args.d}  n_shots={args.n_shots}  n_rounds={args.n_rounds}")
    print()
    print(f"{'p':>8s}  {'naive':>10s}  {'dijkstra':>10s}  {'delta_pp':>10s}")
    print("-" * 45)
    for p in args.p_grid:
        # Same RNG seed for both arms = same noise realisations
        rng_naive = np.random.default_rng(rng_seed)
        rng_dijkstra = np.random.default_rng(rng_seed)
        t0 = time.time()
        naive = run_arm(decode_with_naive_mwpm, args.d, args.n_shots, float(p),
                          args.n_rounds, rng_naive)
        dijkstra = run_arm(decode_with_dijkstra_mwpm, args.d, args.n_shots,
                              float(p), args.n_rounds, rng_dijkstra)
        elapsed = time.time() - t0
        t_total += elapsed
        delta_pp = 100.0 * (dijkstra["logical_zero_rate"] - naive["logical_zero_rate"])
        rows.append({
            "p": float(p),
            "naive_logical_zero_rate": naive["logical_zero_rate"],
            "dijkstra_logical_zero_rate": dijkstra["logical_zero_rate"],
            "delta_pp": delta_pp,
            "elapsed_seconds": elapsed,
        })
        print(f"{p:>8.4f}  {naive['logical_zero_rate']:>10.4f}  "
              f"{dijkstra['logical_zero_rate']:>10.4f}  "
              f"{delta_pp:>+10.2f}")
    print()
    print(f"Total elapsed: {t_total:.1f} s")
    print()
    print("On real Heron-r2 hardware (Systrophe v0.19.2), the same algorithmic")
    print("swap gave +5-25 percentage points across the d=5 round-sweep.")
    print("See paper/surface_code_multidistance_break_even.pdf in the parent")
    print("repo for the hardware reference numbers.")

    out_path = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).with_name(
        "compare_naive_vs_dijkstra_results.json"
    )
    out_path.write_text(json.dumps({
        "args": vars(args),
        "rows": rows,
        "total_elapsed_s": t_total,
    }, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
