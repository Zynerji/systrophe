"""D-CTC deep exploration --- Phase A: dim_CR x dim_CTC scaling.

Sweeps (dim_CR, dim_CTC) and tracks iteration count distribution,
fixed-point purity, and entropy. Looks for scaling exponents.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.d_ctc import (
    dctc_fixed_point,
    density_matrix_diagnostics,
    maximally_mixed_state,
)


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def vn_entropy(rho: np.ndarray) -> float:
    eigs = np.linalg.eigvalsh(rho).real
    eigs = np.clip(eigs, 1e-15, None)
    return float(-np.sum(eigs * np.log(eigs)))


def stress_one_config(
    dim_cr: int, dim_ctc: int, n_samples: int, seed: int, max_iter: int = 5000
) -> dict:
    rng = np.random.default_rng(seed)
    dim_total = dim_cr * dim_ctc
    iters, purities, entropies, residuals, converged = [], [], [], [], []
    for k in range(n_samples):
        U = haar_random_unitary(dim_total, rng)
        # Pure |0><0| on CR, random pure state on CTC
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                              rho_ctc_init=rho_init, tol=1e-10,
                              max_iter=max_iter)
        rho_fp = r["rho_ctc"]
        diag = density_matrix_diagnostics(rho_fp)
        iters.append(r["iterations"])
        purities.append(diag["purity"])
        entropies.append(vn_entropy(rho_fp))
        residuals.append(r["residual"])
        converged.append(bool(r["converged"]))
    return {
        "iters": np.array(iters),
        "purities": np.array(purities),
        "entropies": np.array(entropies),
        "residuals": np.array(residuals),
        "converged_fraction": float(np.mean(converged)),
    }


def main():
    print("=" * 70)
    print("D-CTC Phase A: dim_CR x dim_CTC scaling")
    print("=" * 70)
    print()

    dim_cr_list = [1, 2, 3, 4, 5, 6, 8]
    dim_ctc_list = [2, 3, 4, 5]
    n_per = 150

    print(f"Sweep: dim_CR in {dim_cr_list}, dim_CTC in {dim_ctc_list}")
    print(f"{n_per} samples per (dim_CR, dim_CTC) cell")
    print()

    results = {}
    t0 = time.time()
    for dim_cr in dim_cr_list:
        for dim_ctc in dim_ctc_list:
            seed = 11 + dim_cr * 100 + dim_ctc
            r = stress_one_config(dim_cr, dim_ctc, n_per, seed)
            key = f"{dim_cr}x{dim_ctc}"
            results[key] = {
                "dim_cr": dim_cr, "dim_ctc": dim_ctc,
                "iter_median": float(np.median(r["iters"])),
                "iter_mean": float(np.mean(r["iters"])),
                "iter_p95": float(np.percentile(r["iters"], 95)),
                "iter_p99": float(np.percentile(r["iters"], 99)),
                "iter_max": int(np.max(r["iters"])),
                "iter_min": int(np.min(r["iters"])),
                "iter_std": float(np.std(r["iters"])),
                "purity_mean": float(np.mean(r["purities"])),
                "purity_max": float(np.max(r["purities"])),
                "purity_min": float(np.min(r["purities"])),
                "purity_q99": float(np.percentile(r["purities"], 99)),
                "purity_floor": 1.0 / dim_ctc,
                "entropy_mean": float(np.mean(r["entropies"])),
                "entropy_max_theoretical": float(np.log(dim_ctc)),
                "converged_fraction": r["converged_fraction"],
            }
            print(f"  dim_CR={dim_cr:1d}, dim_CTC={dim_ctc:1d}: "
                  f"iter med={results[key]['iter_median']:6.1f} "
                  f"max={results[key]['iter_max']:5d}  "
                  f"purity mean={results[key]['purity_mean']:.3f} "
                  f"max={results[key]['purity_max']:.3f} "
                  f"(floor {1.0/dim_ctc:.3f})")
    print()
    print(f"Total compute: {time.time() - t0:.1f}s")
    print()

    # Scaling laws
    print("=" * 70)
    print("Scaling laws")
    print("=" * 70)
    print()

    # iter_median as a function of dim_CR at fixed dim_CTC
    print("Median iterations vs. dim_CR (at fixed dim_CTC):")
    print("  dim_CTC | dim_CR=1   2   3   4   5   6   8")
    for dim_ctc in dim_ctc_list:
        row = [results[f"{dc}x{dim_ctc}"]["iter_median"] for dc in dim_cr_list]
        print(f"     {dim_ctc:2d}    | " + " ".join(f"{r:5.0f}" for r in row))

    # Fit iter ~ dim_CR^alpha
    print()
    print("Power-law fit iter_median ~ dim_CR^alpha (per dim_CTC):")
    for dim_ctc in dim_ctc_list:
        # Skip dim_CR = 1 (trivial)
        xs = np.array(dim_cr_list[1:], dtype=float)
        ys = np.array([results[f"{int(dc)}x{dim_ctc}"]["iter_median"] for dc in xs])
        # log-log fit
        valid = ys > 0
        if valid.sum() >= 3:
            slope, intercept = np.polyfit(np.log(xs[valid]), np.log(ys[valid]), 1)
            print(f"  dim_CTC={dim_ctc}: alpha = {slope:.3f}, C = {np.exp(intercept):.2f}")

    print()
    print("Max purity vs. dim_CR (do highly-pure fixed points get rarer at large dim_CR?):")
    print("  dim_CTC | dim_CR=1     2     3     4     5     6     8")
    for dim_ctc in dim_ctc_list:
        row = [results[f"{dc}x{dim_ctc}"]["purity_max"] for dc in dim_cr_list]
        floor_str = f"(floor {1.0/dim_ctc:.3f})"
        print(f"     {dim_ctc:2d}    | " + " ".join(f"{r:5.3f}" for r in row) + f" {floor_str}")

    # Write results
    out_path = Path("examples") / "dctc_deep_phase_a_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
