"""Stress Test 3: D-CTC fixed-point on Haar-random unitaries.

Samples Haar-random unitaries U on the joint Hilbert space
(CR x CTC) = (dim_CR x 3). Runs `dctc_fixed_point` on each and
records:
  - iteration count to converge
  - fixed-point trace, Hermitian residual, positivity
  - fixed-point purity tr(rho^2)
  - fixed-point von Neumann entropy

Question: are there U with super-linear or super-fast convergence?
Are there U where convergence stalls (long iteration tail)?

Output
------
examples/stress_dctc_haar_results.json with statistics + outlier list.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import (
    dctc_fixed_point,
    density_matrix_diagnostics,
    maximally_mixed_state,
)


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random unitary via QR of complex Gaussian."""
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    # Phase-correct diagonal to make Q Haar-uniform
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def von_neumann_entropy(rho: np.ndarray) -> float:
    """Standard S(rho) = -tr(rho log rho)."""
    eigs = np.linalg.eigvalsh(rho).real
    eigs = np.clip(eigs, 1e-15, None)
    return float(-np.sum(eigs * np.log(eigs)))


def stress_run(dim_cr: int = 2, n_samples: int = 200, seed: int = 7) -> dict:
    """Sample n_samples Haar-random U and run dctc_fixed_point on each."""
    rng = np.random.default_rng(seed)
    dim_ctc = 3
    dim_total = dim_cr * dim_ctc

    iter_counts = []
    purities = []
    entropies = []
    converged_flags = []
    final_residuals = []

    for k in range(n_samples):
        U = haar_random_unitary(dim_total, rng)
        # Pure |0><0| for CR; random pure state for CTC init.
        # Both are non-maximally-mixed -> non-trivial fixed point.
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        # random pure CTC init
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        result = dctc_fixed_point(
            U, sigma_cr, dim_cr=dim_cr,
            rho_ctc_init=rho_init,
            tol=1e-10, max_iter=2000,
        )
        rho_fp = result["rho_ctc"]
        diag = density_matrix_diagnostics(rho_fp)
        iter_counts.append(result["iterations"])
        purities.append(diag["purity"])
        entropies.append(von_neumann_entropy(rho_fp))
        converged_flags.append(bool(result["converged"]))
        final_residuals.append(result["residual"])

    iter_counts = np.array(iter_counts, dtype=float)
    purities = np.array(purities)
    entropies = np.array(entropies)
    converged_flags = np.array(converged_flags)

    return {
        "dim_cr": dim_cr,
        "n_samples": n_samples,
        "iter_counts": iter_counts.tolist(),
        "purities": purities.tolist(),
        "entropies": entropies.tolist(),
        "converged_flags": converged_flags.tolist(),
        "final_residuals": final_residuals,
        "summary": {
            "fraction_converged": float(converged_flags.mean()),
            "iter_min": int(iter_counts.min()),
            "iter_median": float(np.median(iter_counts)),
            "iter_mean": float(iter_counts.mean()),
            "iter_max": int(iter_counts.max()),
            "iter_p95": float(np.percentile(iter_counts, 95)),
            "iter_p99": float(np.percentile(iter_counts, 99)),
            "purity_mean": float(purities.mean()),
            "purity_min": float(purities.min()),  # close to 1/dim_ctc = 0.333 if max-mixed
            "purity_max": float(purities.max()),  # 1 if pure
            "entropy_mean": float(entropies.mean()),
            "entropy_max": float(entropies.max()),
            "entropy_min": float(entropies.min()),
        },
    }


def find_outliers(result: dict) -> dict:
    """Report fastest-converging and slowest-converging samples."""
    iter_counts = np.array(result["iter_counts"])
    purities = np.array(result["purities"])
    entropies = np.array(result["entropies"])

    fast_idx = np.argsort(iter_counts)[:5].tolist()
    slow_idx = np.argsort(-iter_counts)[:5].tolist()
    pure_idx = np.argsort(-purities)[:5].tolist()

    def info(idx):
        return [{
            "sample": int(i),
            "iter": int(iter_counts[i]),
            "purity": float(purities[i]),
            "entropy": float(entropies[i]),
        } for i in idx]
    return {
        "fastest_converging": info(fast_idx),
        "slowest_converging": info(slow_idx),
        "purest_fixed_point": info(pure_idx),
    }


def main():
    print("=" * 60)
    print("D-CTC fixed-point on Haar-random unitaries")
    print("=" * 60)
    print()

    # Run for dim_cr = 2 and dim_cr = 4
    all_results = {}
    for dim_cr in (2, 4):
        n = 200
        print(f"Sampling {n} Haar-random unitaries on (dim_CR={dim_cr} x dim_CTC=3)...")
        result = stress_run(dim_cr=dim_cr, n_samples=n)
        s = result["summary"]
        print(f"  fraction converged    : {s['fraction_converged']:.3f}")
        print(f"  iteration count (min, median, mean, p95, p99, max):")
        print(f"    {s['iter_min']:4d}, {s['iter_median']:6.1f}, {s['iter_mean']:7.2f}, "
              f"{s['iter_p95']:5.1f}, {s['iter_p99']:5.1f}, {s['iter_max']:4d}")
        print(f"  purity:  mean={s['purity_mean']:.4f}, min={s['purity_min']:.4f}, max={s['purity_max']:.4f}")
        print(f"  entropy: mean={s['entropy_mean']:.4f}, min={s['entropy_min']:.4f}, max={s['entropy_max']:.4f}")
        out_idx = find_outliers(result)
        print(f"\n  Outliers (dim_cr={dim_cr}):")
        print(f"    fastest 5 (iter): {[o['iter'] for o in out_idx['fastest_converging']]}")
        print(f"    slowest 5 (iter): {[o['iter'] for o in out_idx['slowest_converging']]}")
        print(f"    most-pure entropies: {[round(o['entropy'], 3) for o in out_idx['purest_fixed_point']]}")
        all_results[dim_cr] = {**result, "outliers": out_idx}
        print()

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print()
    s2 = all_results[2]["summary"]
    s4 = all_results[4]["summary"]
    print(f"dim_CR=2: typical iter = {s2['iter_median']:.0f}, max = {s2['iter_max']}")
    print(f"dim_CR=4: typical iter = {s4['iter_median']:.0f}, max = {s4['iter_max']}")
    print()
    print(f"Maximally mixed state on CTC has purity = 1/3 = 0.333. Haar")
    print(f"average purity: dim_cr=2 -> {s2['purity_mean']:.4f}, "
          f"dim_cr=4 -> {s4['purity_mean']:.4f}.")
    print()
    print(f"Maximum entropy = log(3) = 1.0986. Haar mean entropy:")
    print(f"  dim_cr=2 -> {s2['entropy_mean']:.4f}")
    print(f"  dim_cr=4 -> {s4['entropy_mean']:.4f}")
    print()

    # Strip the per-sample arrays from JSON output to keep it readable
    for dim_cr in (2, 4):
        # keep summary + outliers
        del all_results[dim_cr]["iter_counts"]
        del all_results[dim_cr]["purities"]
        del all_results[dim_cr]["entropies"]
        del all_results[dim_cr]["converged_flags"]
        del all_results[dim_cr]["final_residuals"]

    payload = {str(k): v for k, v in all_results.items()}
    out_path = Path("examples") / "stress_dctc_haar_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
