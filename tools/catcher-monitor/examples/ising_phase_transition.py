"""Reproduction: 2D Ising T_c detection via the Systrophe catcher.

The HASH-QUINE prior work (`tHHmL/HASH-QUINE/`) reported that the
address-space catcher hits the 2D Ising critical temperature within
~1.4% on an L=32 lattice when fed a magnetisation-vs-temperature
sweep encoded in *address-space* (path 3 -- not value-encoded).

This example reproduces that result with the catcher-monitor public
API. We simulate 2D Ising on an L=24 lattice via Metropolis-Hastings,
sweep temperature, and feed the magnetisation array (per-site spin
configuration coarsened to per-row sums) to `find_phase_transition`.

Analytical 2D Ising T_c = 2 / log(1 + sqrt(2)) ~= 2.269.

The catcher should land within a few percent of that.

Note: a brute-force Metropolis sweep at every T point would take
minutes on a single thread. We use a coarse-but-honest sweep with
short equilibration; this is a calibration check, not a production
Ising solver.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from catcher_monitor import find_phase_transition


def metropolis_step(state: np.ndarray, T: float,
                       rng: np.random.Generator) -> None:
    """One in-place Metropolis sweep over an L x L Ising lattice."""
    L = state.shape[0]
    # Random update order
    for _ in range(L * L):
        i = int(rng.integers(0, L))
        j = int(rng.integers(0, L))
        s = state[i, j]
        # Sum of nearest neighbours (periodic boundaries)
        nb = (
            state[(i + 1) % L, j] + state[(i - 1) % L, j]
            + state[i, (j + 1) % L] + state[i, (j - 1) % L]
        )
        dE = 2.0 * s * nb
        if dE <= 0 or rng.random() < math.exp(-dE / max(T, 1e-9)):
            state[i, j] = -s


def measure_at_T(L: int, T: float, n_eq: int, n_measure: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Return a snapshot of the *coarse-grained* spin configuration at T.

    We average n_measure post-equilibration snapshots and coarse-grain
    to an L-element row signature so that successive temperatures can be
    distinguished in address-space. Hash-Quine's L=32 reference used
    a similar coarse-graining of the spin field, not the raw L*L
    configuration -- the latter is dominated by Monte Carlo noise.
    """
    state = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    for _ in range(n_eq):
        metropolis_step(state, T, rng)
    row_mags = np.zeros(L, dtype=float)
    abs_mag_sum = 0.0
    nn_corr_sum = 0.0   # nearest-neighbour correlation, sharpest near T_c
    for _ in range(n_measure):
        metropolis_step(state, T, rng)
        row_mags += state.sum(axis=1)
        abs_mag_sum += float(abs(state.sum()))
        # 4-NN correlation
        nn_corr_sum += float(
            (state * (np.roll(state, 1, 0) + np.roll(state, -1, 0)
                       + np.roll(state, 1, 1) + np.roll(state, -1, 1))
            ).sum()
        )
    # Concatenate row mags + global |M| + NN correlation -> richer signal
    out = np.concatenate([
        row_mags / float(n_measure),
        np.array([abs_mag_sum / float(n_measure),
                  nn_corr_sum / float(n_measure) / float(L * L * 4)]),
    ])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=24)
    parser.add_argument("--n_T", type=int, default=40)
    parser.add_argument("--T_min", type=float, default=1.5)
    parser.add_argument("--T_max", type=float, default=3.5)
    parser.add_argument("--n_eq", type=int, default=200)
    parser.add_argument("--n_measure", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    T_grid = np.linspace(args.T_min, args.T_max, args.n_T)
    T_c_truth = 2.0 / math.log(1.0 + math.sqrt(2.0))   # ~2.269

    print(f"L={args.L}  n_T={args.n_T}  T in [{args.T_min}, {args.T_max}]")
    print(f"Analytical T_c = {T_c_truth:.6f}")
    print()
    print("Simulating Ising magnetisation curve (may take ~30s)")

    t0 = time.time()
    rows_by_T = []
    for T in T_grid:
        rows = measure_at_T(args.L, float(T), args.n_eq, args.n_measure, rng)
        rows_by_T.append(rows)
        sys.stdout.write(f"  T={T:.3f}  |M|/N={float(abs(rows).sum())/args.L/args.L:.3f}\r")
        sys.stdout.flush()
    print()
    elapsed_sim = time.time() - t0
    print(f"Simulation: {elapsed_sim:.1f} s")

    # Feed |row magnetisation vector| as the address (L-dim) at each T.
    # This is the address-space encoding -- the Systrophe HASH-QUINE finding
    # is that address-space is what works here, not raw value encoding.
    rows_array = np.stack(rows_by_T)  # (n_T, L)

    def measurement_fn(T: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(T_grid - T)))
        return rows_array[idx]

    print("Running find_phase_transition...")
    t0 = time.time()
    res = find_phase_transition(T_grid, measurement_fn, n_bits=32)
    elapsed_cat = time.time() - t0
    print(f"Catcher: {elapsed_cat:.2f} s")
    print()

    print("=" * 60)
    print("Result:")
    print(f"  kind:           {res.kind}")
    print(f"  detected T_c:   {res.transition_at}")
    print(f"  truth T_c:      {T_c_truth:.4f}")
    if res.transition_at is not None:
        err = abs(res.transition_at - T_c_truth)
        err_pct = 100.0 * err / T_c_truth
        print(f"  absolute error: {err:.4f}")
        print(f"  relative error: {err_pct:.2f}%")
    print(f"  confidence:     {res.confidence:.3f}")
    print(f"  n_emergents:    {res.n_sharp_features_value + res.n_sharp_features_derivative}")

    out = {
        "args": vars(args),
        "T_c_truth": float(T_c_truth),
        "T_c_detected": float(res.transition_at) if res.transition_at is not None else None,
        "abs_error": (
            float(abs(res.transition_at - T_c_truth)) if res.transition_at is not None else None
        ),
        "kind": res.kind,
        "confidence": res.confidence,
        "elapsed_sim_s": elapsed_sim,
        "elapsed_catcher_s": elapsed_cat,
        "magnetisation_curve": rows_array.mean(axis=1).tolist(),
        "T_grid": T_grid.tolist(),
    }
    out_path = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).with_name(
        "ising_phase_transition_results.json"
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
