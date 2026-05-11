"""E1 deep dive: limit cycle structure of D-CTC iteration.

Questions:
  1. How often do limit cycles vs true convergence occur?
  2. Distribution of cycle lengths (2, 3, 4, ...)?
  3. Spectral predictor: do channels with eigenvalues near
     -1 (period-2), exp(2*pi*i/3) (period-3), etc. produce
     corresponding cycles?
  4. Do cycles affect AW amplification?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.d_ctc import (
    apply_channel,
    channel_superoperator,
    clifford_like_unitary,
    dctc_fixed_point,
    density_matrix_diagnostics,
)


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def detect_cycle_length(U, sigma_cr, dim_cr, rho_init, max_steps: int = 100,
                         tol: float = 1e-8) -> int:
    """Detect cycle length by checking if rho_n matches rho_{n-k} for some k.

    Returns 0 if iteration converges (rho_{n+1} = rho_n), else the
    detected cycle length.
    """
    rho_history = [rho_init]
    rho = rho_init.copy()
    for step in range(max_steps):
        rho_new = apply_channel(U, sigma_cr, rho, dim_cr)
        # Check if rho_new matches any previous rho within tol
        for k in range(1, min(step + 1, 20)):
            past = rho_history[-k]
            d = float(np.linalg.norm(rho_new - past, "fro"))
            if d < tol:
                return k  # cycle length k
        rho_history.append(rho_new)
        rho = rho_new
    return -1  # no cycle detected within max_steps


def main():
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_cr[0, 0] = 1.0
    rng = np.random.default_rng(7777)

    print("=" * 70)
    print("E1 deep dive: D-CTC iteration cycle structure")
    print("=" * 70)
    print()

    # ---- Part 1: cycle-length histogram across ensembles ----
    n = 1000
    psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
    psi /= np.linalg.norm(psi)
    rho_init = np.outer(psi, psi.conj())

    haar_cycle_lengths = []
    cliff_cycle_lengths = []
    for _ in range(n):
        for kind in ("haar", "cliff"):
            U = haar_random_unitary(dim_total, rng) if kind == "haar" else clifford_like_unitary(dim_total, rng)
            cycle = detect_cycle_length(U, sigma_cr, dim_cr, rho_init)
            (haar_cycle_lengths if kind == "haar" else cliff_cycle_lengths).append(cycle)

    print("Cycle-length distribution (Haar, n=1000):")
    for k in (1, 2, 3, 4, 5, -1):
        count = sum(1 for c in haar_cycle_lengths if c == k)
        label = "converged" if k == 1 else ("no cycle" if k == -1 else f"period-{k}")
        print(f"  {label:14s}: {count:4d} ({count*100/n:.1f}%)")
    print()
    print("Cycle-length distribution (Clifford-like, n=1000):")
    for k in (1, 2, 3, 4, 5, -1):
        count = sum(1 for c in cliff_cycle_lengths if c == k)
        label = "converged" if k == 1 else ("no cycle" if k == -1 else f"period-{k}")
        print(f"  {label:14s}: {count:4d} ({count*100/n:.1f}%)")
    print()

    # ---- Part 2: spectral prediction ----
    print("=" * 70)
    print("Spectral predictor: do eigenvalues near -1 cause period-2?")
    print("=" * 70)
    print()
    print("Channels with |arg(lambda_2)| near pi (negative real eigenvalues):")

    n_test = 500
    arg_lam2_period2 = []
    arg_lam2_converged = []
    arg_lam2_period3 = []
    abs_lam2_period2 = []
    abs_lam2_converged = []
    rng2 = np.random.default_rng(99999)
    for _ in range(n_test):
        U = clifford_like_unitary(dim_total, rng2)
        cycle = detect_cycle_length(U, sigma_cr, dim_cr, rho_init)
        M = channel_superoperator(U, sigma_cr, dim_cr)
        eigs = np.linalg.eigvals(M)
        eigs_abs = np.abs(eigs)
        order = np.argsort(-eigs_abs)
        lam2 = eigs[order[1]]
        lam2_abs = float(np.abs(lam2))
        lam2_arg = float(np.angle(lam2))
        if cycle == 1:
            arg_lam2_converged.append(lam2_arg)
            abs_lam2_converged.append(lam2_abs)
        elif cycle == 2:
            arg_lam2_period2.append(lam2_arg)
            abs_lam2_period2.append(lam2_abs)
        elif cycle == 3:
            arg_lam2_period3.append(lam2_arg)

    print(f"  Period-1 (converged):    n={len(arg_lam2_converged)}, mean |arg(lam2)| = {np.mean(np.abs(arg_lam2_converged)):.3f}")
    print(f"  Period-2:                n={len(arg_lam2_period2)}, mean |arg(lam2)| = {np.mean(np.abs(arg_lam2_period2)):.3f}")
    print(f"  Period-3:                n={len(arg_lam2_period3)}, mean |arg(lam2)| = {np.mean(np.abs(arg_lam2_period3)) if arg_lam2_period3 else 0:.3f}")
    print(f"  For period-2, |arg(lam2)| should cluster near pi = {np.pi:.3f}")
    print(f"  For period-3, should cluster near 2*pi/3 = {2*np.pi/3:.3f}")
    print()
    print(f"  Period-2 mean |lambda_2|:    {np.mean(abs_lam2_period2):.4f}")
    print(f"  Period-1 mean |lambda_2|:    {np.mean(abs_lam2_converged):.4f}")
    print()

    # ---- Part 3: cycle vs amplification ----
    print("=" * 70)
    print("Do cycles affect AW amplification?")
    print("=" * 70)
    print()
    sigma_a = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_a[0, 0] = 1.0
    sigma_b = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_b[0, 0] = 0.9; sigma_b[1, 1] = 0.1
    converged_amps = []
    period2_amps = []
    rng3 = np.random.default_rng(20000)
    n_a = 500
    for _ in range(n_a):
        U = clifford_like_unitary(dim_total, rng3)
        cycle = detect_cycle_length(U, sigma_cr, dim_cr, rho_init)
        r_a = dctc_fixed_point(U, sigma_a, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                tol=1e-10, max_iter=2000)["rho_ctc"]
        r_b = dctc_fixed_point(U, sigma_b, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                tol=1e-10, max_iter=2000)["rho_ctc"]
        diff = r_a - r_b
        eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
        amp = 0.5 * float(np.sum(np.abs(eigs)))
        if cycle == 1:
            converged_amps.append(amp)
        elif cycle == 2:
            period2_amps.append(amp)
    print(f"  Period-1: n={len(converged_amps)}, mean amp = {np.mean(converged_amps):.4f}, max = {np.max(converged_amps) if converged_amps else 0:.4f}")
    print(f"  Period-2: n={len(period2_amps)}, mean amp = {np.mean(period2_amps):.4f}, max = {np.max(period2_amps) if period2_amps else 0:.4f}")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    p2_haar = sum(1 for c in haar_cycle_lengths if c == 2)
    p2_cliff = sum(1 for c in cliff_cycle_lengths if c == 2)
    print(f"Period-2 cycle fraction:")
    print(f"  Haar: {p2_haar/n*100:.1f}%")
    print(f"  Clifford: {p2_cliff/n*100:.1f}%")
    if p2_cliff > p2_haar * 3:
        print(f"  Clifford produces period-2 cycles at ~{p2_cliff/max(p2_haar,1):.1f}x Haar rate.")
    print()

    out = Path("examples") / "dctc_deep_E1_cycles_results.json"
    with open(out, "w") as f:
        json.dump({
            "haar_cycle_distribution": {str(k): int(sum(1 for c in haar_cycle_lengths if c == k))
                                          for k in (1, 2, 3, 4, 5, -1)},
            "clifford_cycle_distribution": {str(k): int(sum(1 for c in cliff_cycle_lengths if c == k))
                                                for k in (1, 2, 3, 4, 5, -1)},
            "period2_mean_arg_lam2": float(np.mean(np.abs(arg_lam2_period2)) if arg_lam2_period2 else 0),
            "period1_mean_arg_lam2": float(np.mean(np.abs(arg_lam2_converged)) if arg_lam2_converged else 0),
            "period2_mean_amp": float(np.mean(period2_amps) if period2_amps else 0),
            "period1_mean_amp": float(np.mean(converged_amps) if converged_amps else 0),
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
