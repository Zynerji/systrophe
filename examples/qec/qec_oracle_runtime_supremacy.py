"""Demonstrate spectral-oracle runtime supremacy for parameter sweeps.

The DCTC spectral oracle predicts QEC decoder iteration count from
|lambda_2(channel)| in O(d^6) classical computation. Running a real
decoder costs O(iter_count * d^a) for some decoder-specific a.

For PARAMETER SWEEPS (sweeping p_phys, code distance, noise model)
the oracle is computed ONCE per channel; the decoder runs ONCE per
parameter point. For 100 parameter points and iter_count ~ 1000,
the runtime comparison heavily favors the oracle:

    oracle cost:  100 * O(d^6) = 100 * d^6
    decoder cost: 100 * 1000 * O(d^a) = 10^5 * d^a

For d=3, oracle = 7.3e4 vs decoder = 2.7e6 (~37x oracle speedup).
For d=5, oracle = 1.6e6 vs decoder = 1.25e7 (~8x speedup at low d^a).

This script benchmarks the actual speedup on a parameter sweep.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from systrophe.quantum_info.qec_codes import (
    channel_lambda_2,
    channel_lambda_2_fast,
    repetition_code_superoperator,
)
from systrophe.quantum_info.qec_decoders import iterative_channel_decoder


def benchmark_runtime_comparison(
    n_qubits: int = 3,
    p_grid: np.ndarray | None = None,
    n_decoder_runs: int = 1,
    tol: float = 1e-6,
) -> dict:
    """Compare oracle vs decoder total runtime across a parameter sweep."""
    if p_grid is None:
        p_grid = np.linspace(0.01, 0.2, 30)

    # Oracle: compute |lambda_2| at each p using fast power-iteration
    t0 = time.perf_counter()
    oracle_predictions = []
    for p in p_grid:
        S = repetition_code_superoperator(float(p), n=n_qubits)
        lam_2 = channel_lambda_2_fast(S)
        if lam_2 <= 1e-15:
            iters = 1.0
        elif lam_2 >= 1.0 - 1e-15:
            iters = float("inf")
        else:
            iters = -math.log(tol) / -math.log(lam_2)
        oracle_predictions.append(iters)
    oracle_runtime = time.perf_counter() - t0

    # Decoder: actually run the iterative decoder at each p
    t0 = time.perf_counter()
    decoder_iterations = []
    for p in p_grid:
        S = repetition_code_superoperator(float(p), n=n_qubits)
        r = iterative_channel_decoder(S, tol=tol, max_iter=5000)
        decoder_iterations.append(r.iter_count)
    decoder_runtime = time.perf_counter() - t0

    speedup = decoder_runtime / max(oracle_runtime, 1e-9)

    # Per-point comparison
    pred = np.array(oracle_predictions)
    meas = np.array(decoder_iterations)
    finite = np.isfinite(pred) & (meas > 0)
    if finite.sum() >= 3:
        log_pred = np.log10(pred[finite])
        log_meas = np.log10(meas[finite])
        pearson = float(np.corrcoef(log_pred, log_meas)[0, 1])
    else:
        pearson = float("nan")

    return {
        "n_qubits": n_qubits,
        "p_grid": p_grid.tolist(),
        "oracle_predictions": oracle_predictions,
        "decoder_iterations": decoder_iterations,
        "oracle_runtime_s": oracle_runtime,
        "decoder_runtime_s": decoder_runtime,
        "speedup": speedup,
        "pearson_r_log_log": pearson,
    }


def main() -> None:
    print("=" * 70)
    print("QEC Oracle Runtime Supremacy Benchmark")
    print("=" * 70)
    print()

    all_results = {}
    for n in (3, 4, 5):
        print(f"--- {n}-qubit repetition code ---")
        out = benchmark_runtime_comparison(n_qubits=n)
        print(f"  oracle runtime:  {out['oracle_runtime_s']:8.4f} s "
              f"({len(out['p_grid'])} parameter points)")
        print(f"  decoder runtime: {out['decoder_runtime_s']:8.4f} s")
        print(f"  speedup:         {out['speedup']:8.1f}x")
        print(f"  Pearson r:       {out['pearson_r_log_log']:8.4f}")
        print()
        all_results[f"n{n}"] = out

    out_path = Path(__file__).parent / "qec_oracle_runtime_supremacy_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
