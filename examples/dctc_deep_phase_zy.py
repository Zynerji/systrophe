"""D-CTC Phase Z+Y: trajectory + Bell preservation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.d_ctc import (
    apply_channel,
    clifford_like_unitary,
    dctc_fixed_point,
    density_matrix_diagnostics,
)


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def vn_entropy(rho):
    eigs = np.linalg.eigvalsh(rho).real
    eigs = np.clip(eigs, 1e-15, None)
    return float(-np.sum(eigs * np.log(eigs)))


def main():
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_cr[0, 0] = 1.0
    rng = np.random.default_rng(33)

    # Phase Z: trajectory analysis
    print("=" * 70)
    print("Phase Z: D-CTC iteration trajectory")
    print("=" * 70)
    print()

    # Pick a high-purity Clifford U
    best_U = None
    best_pur = 0
    psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
    psi /= np.linalg.norm(psi)
    rho_init = np.outer(psi, psi.conj())
    for _ in range(300):
        U = clifford_like_unitary(dim_total, rng)
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                              tol=1e-10, max_iter=3000)
        p = density_matrix_diagnostics(r["rho_ctc"])["purity"]
        if p > best_pur:
            best_pur = p
            best_U = U
        if p > 0.95:
            break
    print(f"Reference U: purity = {best_pur:.4f}")
    print()

    # Trajectory: track purity, entropy, trace distance to fp at each step
    rho_n = rho_init.copy()
    rho_fp_target = dctc_fixed_point(best_U, sigma_cr, dim_cr=dim_cr,
                                        rho_ctc_init=rho_init,
                                        tol=1e-10, max_iter=3000)["rho_ctc"]
    trajectory = []
    for n in range(50):
        diag = density_matrix_diagnostics(rho_n)
        entr = vn_entropy(rho_n)
        dist_to_fp = float(np.linalg.norm(rho_n - rho_fp_target, "fro"))
        trajectory.append({
            "n": n, "purity": float(diag["purity"]), "entropy": entr,
            "dist_to_fp": dist_to_fp,
        })
        rho_n = apply_channel(best_U, sigma_cr, rho_n, dim_cr)

    print(f"{'n':4s} {'purity':9s} {'entropy':9s} {'dist_to_fp':12s}")
    for t in trajectory[:10]:
        print(f"  {t['n']:3d}  {t['purity']:7.4f}   {t['entropy']:7.4f}   {t['dist_to_fp']:9.6f}")
    print("  ...")
    for t in trajectory[-3:]:
        print(f"  {t['n']:3d}  {t['purity']:7.4f}   {t['entropy']:7.4f}   {t['dist_to_fp']:9.6f}")
    print()

    # Detect convergence rate: log dist_to_fp vs n -> slope
    dists = np.array([t["dist_to_fp"] for t in trajectory])
    valid = dists > 1e-12
    if valid.sum() >= 5:
        n_arr = np.arange(len(dists))[valid]
        log_d = np.log(dists[valid])
        slope, intercept = np.polyfit(n_arr, log_d, 1)
        print(f"  Convergence: log(dist) ~ {intercept:.3f} + {slope:.4f} * n")
        print(f"  Effective |lambda_2| = exp(slope) = {np.exp(slope):.4f}")
    print()

    # Phase Y: Bell-state preservation
    print("=" * 70)
    print("Phase Y: Bell-state entanglement preservation under U")
    print("=" * 70)
    print()

    # Construct Bell-like state on CR x CR' (auxiliary qubit)
    # |bell> = (|00> + |11>)/sqrt(2)
    # But we need state on (CR x CTC) -- use bell state between CR and one CTC subspace
    # |bell> = (|0,0> + |1,1>)/sqrt(2) where second register is dim_ctc dimensional
    bell = np.zeros(dim_cr * dim_ctc, dtype=complex)
    bell[0] = 1 / np.sqrt(2)
    bell[dim_ctc + 1] = 1 / np.sqrt(2)  # |1, 1>
    bell_rho = np.outer(bell, bell.conj())
    purity_bell = float(np.real(np.trace(bell_rho @ bell_rho)))
    print(f"Initial Bell state on (CR x CTC subspace):")
    print(f"  purity: {purity_bell:.4f}")
    # Reduced state on CR
    reduced_cr_initial = bell_rho.reshape(dim_cr, dim_ctc, dim_cr, dim_ctc).trace(axis1=1, axis2=3)
    entropy_cr_initial = vn_entropy(reduced_cr_initial)
    print(f"  S(rho_CR) initial: {entropy_cr_initial:.4f}  (log 2 = {np.log(2):.4f} max)")
    print()

    # Apply U
    bell_rho_evolved = best_U @ bell_rho @ best_U.conj().T
    reduced_cr_evolved = bell_rho_evolved.reshape(dim_cr, dim_ctc, dim_cr, dim_ctc).trace(axis1=1, axis2=3)
    entropy_cr_evolved = vn_entropy(reduced_cr_evolved)
    print(f"After applying Clifford U:")
    print(f"  S(rho_CR) after: {entropy_cr_evolved:.4f}")
    print(f"  Change: {entropy_cr_evolved - entropy_cr_initial:+.4f}")
    if abs(entropy_cr_evolved - entropy_cr_initial) < 0.01:
        print("  Entanglement preserved.")
    elif entropy_cr_evolved < entropy_cr_initial - 0.1:
        print("  Disentangled (purified CR subsystem).")
    else:
        print("  Slightly modified.")
    print()

    # Compare to random Haar
    n_haar_trials = 30
    entropy_changes_haar = []
    for _ in range(n_haar_trials):
        U_h = haar_random_unitary(dim_total, rng)
        bell_evolved_h = U_h @ bell_rho @ U_h.conj().T
        reduced_h = bell_evolved_h.reshape(dim_cr, dim_ctc, dim_cr, dim_ctc).trace(axis1=1, axis2=3)
        entr_h = vn_entropy(reduced_h)
        entropy_changes_haar.append(entr_h - entropy_cr_initial)
    entropy_changes_haar = np.array(entropy_changes_haar)
    print(f"Haar (n={n_haar_trials}):")
    print(f"  S change: mean = {entropy_changes_haar.mean():+.4f}, "
          f"std = {entropy_changes_haar.std():.4f}")
    print()

    out = Path("examples") / "dctc_deep_phase_zy_results.json"
    with open(out, "w") as f:
        json.dump({
            "trajectory": trajectory,
            "convergence_lambda2_effective": float(np.exp(slope)) if valid.sum() >= 5 else None,
            "bell": {
                "initial_entropy_cr": entropy_cr_initial,
                "after_clifford_entropy_cr": entropy_cr_evolved,
                "haar_entropy_change_mean": float(entropy_changes_haar.mean()),
                "haar_entropy_change_std": float(entropy_changes_haar.std()),
            },
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
