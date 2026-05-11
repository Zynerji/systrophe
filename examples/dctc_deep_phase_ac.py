"""D-CTC Phase AC: noise robustness of the amplification effect.

If we add CPTP noise (depolarizing channel) to each iteration step,
at what noise level does the Clifford amplification break down?

This determines whether the amplification is observable in any
realistic implementation (with finite-noise CTC operations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.d_ctc import (
    apply_channel,
    density_matrix_diagnostics,
    dctc_fixed_point,
)
from systrophe.novelty_catcher import (
    catch_novelty_in_named_arrays,
    catch_novelty_per_quantity,
)


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def clifford_like_unitary(dim, rng):
    perm = rng.permutation(dim)
    P = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        P[i, perm[i]] = 1.0
    D = np.diag(rng.choice([1, -1, 1j, -1j], dim))
    return P @ D


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def noisy_dctc_iteration(
    U, sigma_cr, dim_cr, rho_init, noise_level, max_iter=2000, tol=1e-10,
):
    """D-CTC iteration with depolarizing noise after each step.

    Each iteration:
        rho_{n+1} = (1 - p) * E(sigma (x) rho_n) + p * I / d_ctc
    """
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    I_d = np.eye(dim_ctc, dtype=complex) / dim_ctc
    rho = rho_init.copy()
    for k in range(max_iter):
        rho_new = apply_channel(U, sigma_cr, rho, dim_cr)
        rho_new = (1 - noise_level) * rho_new + noise_level * I_d
        tr = np.trace(rho_new)
        rho_new = rho_new / tr if abs(tr) > 1e-30 else rho_new
        if np.linalg.norm(rho_new - rho) < tol:
            return rho_new, k + 1
        rho = rho_new
    return rho, max_iter


def main():
    print("=" * 70)
    print("Phase AC: noise robustness of Clifford-amplification")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    rng = np.random.default_rng(1313)

    psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
    psi = psi / np.linalg.norm(psi)
    rho_init = np.outer(psi, psi.conj())

    # Find a high-purity Clifford U
    sigma_cr_pure = np.zeros((dim_cr, dim_cr), dtype=complex)
    sigma_cr_pure[0, 0] = 1.0
    best_U = None
    best_pur = 0
    for _ in range(500):
        U = clifford_like_unitary(dim_total, rng)
        rho_fp, _ = noisy_dctc_iteration(U, sigma_cr_pure, dim_cr, rho_init, 0.0)
        p = density_matrix_diagnostics(rho_fp)["purity"]
        if p > best_pur:
            best_pur = p
            best_U = U
        if p > 0.95:
            break
    print(f"Reference U: purity (noise=0) = {best_pur:.4f}")
    print()

    # Test states (eps = 0.1, close states)
    eps = 0.1
    sigma_a = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_a[0, 0] = 1.0
    sigma_b = np.zeros((dim_cr, dim_cr), dtype=complex)
    sigma_b[0, 0] = 1 - eps; sigma_b[1, 1] = eps
    input_td = trace_distance(sigma_a, sigma_b)
    helstrom = 0.5 + input_td / 2

    print(f"Input states: sigma_a, sigma_b at eps={eps}, td_input = {input_td:.4f}, "
          f"Helstrom = {helstrom:.4f}")
    print()

    noise_vals = np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    print(f"{'noise':6s} {'rho_a_purity':14s} {'output_td':12s} {'CTC_success':14s} {'speedup':9s}")
    results = []
    for noise in noise_vals:
        rho_a, _ = noisy_dctc_iteration(best_U, sigma_a, dim_cr, rho_init, float(noise))
        rho_b, _ = noisy_dctc_iteration(best_U, sigma_b, dim_cr, rho_init, float(noise))
        pur_a = density_matrix_diagnostics(rho_a)["purity"]
        out_td = trace_distance(rho_a, rho_b)
        ctc_succ = 0.5 + out_td / 2
        speedup = (ctc_succ - 0.5) / (helstrom - 0.5)
        results.append({
            "noise": float(noise),
            "rho_a_purity": float(pur_a),
            "output_td": float(out_td),
            "ctc_success": float(ctc_succ),
            "speedup": float(speedup),
        })
        print(f"  {noise:5.3f}  {pur_a:12.4f}    {out_td:9.4f}    "
              f"{ctc_succ:12.4f}    {speedup:7.2f}")

    print()
    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    breakdown_noise = None
    for r in results:
        if r["speedup"] < 1.5:
            breakdown_noise = r["noise"]
            break
    if breakdown_noise is not None:
        print(f"Amplification (speedup > 1.5) survives up to noise = {breakdown_noise:.4f}")
        print(f"For depolarizing noise above this level, D-CTC advantage vanishes.")
    else:
        print(f"Amplification (speedup > 1.5) survives across all tested noise levels.")
    print()

    # Native novelty catcher: per observable, split the noise sweep into
    # low-noise vs high-noise regimes. Sharp Hamming step within a
    # quantity flags a regime change in that observable.
    purities = np.array([r["rho_a_purity"] for r in results])
    out_tds  = np.array([r["output_td"]    for r in results])
    succs    = np.array([r["ctc_success"]  for r in results])
    speedups = np.array([r["speedup"]      for r in results])
    half = len(results) // 2
    novelty = catch_novelty_per_quantity({
        "purity":   {"low_noise": purities[:half], "high_noise": purities[half:]},
        "output_td":{"low_noise": out_tds[:half],  "high_noise": out_tds[half:]},
        "ctc_success":{"low_noise": succs[:half], "high_noise": succs[half:]},
        "speedup": {"low_noise": speedups[:half], "high_noise": speedups[half:]},
    })
    print()
    print(f"Novelty catcher aggregate='{novelty['aggregate_verdict']}', "
          f"novel quantities={novelty['novel_quantities']}")

    out = Path("examples") / "dctc_deep_phase_ac_results.json"
    with open(out, "w") as f:
        json.dump({
            "config": {"dim_cr": dim_cr, "dim_ctc": dim_ctc, "eps_input": eps,
                       "helstrom_success": helstrom},
            "noise_sweep": results,
            "reference_purity_noise0": float(best_pur),
            "breakdown_noise_level": float(breakdown_noise) if breakdown_noise is not None else None,
            "novelty_catcher": novelty,
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
