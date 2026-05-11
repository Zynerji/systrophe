"""D-CTC Phase AB + AL.

AB: perturbation sensitivity. Perturb high-purity Clifford-like U by
    eps * Haar; how does purity decay with eps?

AL: explicit Aaronson-Watrous-style state distinguisher.
    Two related-but-distinct mixed states; can the D-CTC procedure
    classify them in polynomial time?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.d_ctc import dctc_fixed_point, density_matrix_diagnostics


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


def perturb_unitary(U: np.ndarray, eps: float, rng) -> np.ndarray:
    """Mix U with a Haar perturbation: U' = polar((1-eps) U + eps V) where
    V is Haar-random."""
    dim = U.shape[0]
    V = haar_random_unitary(dim, rng)
    M = (1 - eps) * U + eps * V
    # Polar decomposition: project back to unitary group
    U_new, _, Vh = np.linalg.svd(M)
    return U_new @ Vh


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def get_fixed_point(U, sigma_cr, dim_cr, rho_init):
    r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                          rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
    return r["rho_ctc"]


def main():
    print("=" * 70)
    print("Phase AB: perturbation sensitivity of high-purity Clifford U")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
    sigma_cr[0, 0] = 1.0

    # Find a high-purity Clifford U first
    print("Finding a high-purity Clifford U as the reference...")
    rng = np.random.default_rng(11111)
    psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
    psi = psi / np.linalg.norm(psi)
    rho_init = np.outer(psi, psi.conj())
    U_clifford = None
    best_purity = 0
    for trial in range(500):
        U = clifford_like_unitary(dim_total, rng)
        rho_fp = get_fixed_point(U, sigma_cr, dim_cr, rho_init)
        p = density_matrix_diagnostics(rho_fp)["purity"]
        if p > best_purity:
            best_purity = p
            U_clifford = U
        if p > 0.95:
            break
    print(f"  Reference U: purity = {best_purity:.4f}")
    print()

    # Sweep perturbation level
    print(f"Perturbation sweep:")
    print(f"  {'eps':6s} {'purity_mean':12s} {'purity_std':11s} {'purity_max':11s} {'purity_min':11s}")
    eps_vals = np.array([0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    rng_p = np.random.default_rng(99999)
    n_per_eps = 50
    ab_results = []
    for eps in eps_vals:
        purs = []
        for _ in range(n_per_eps):
            U_pert = perturb_unitary(U_clifford, float(eps), rng_p)
            rho_fp = get_fixed_point(U_pert, sigma_cr, dim_cr, rho_init)
            purs.append(density_matrix_diagnostics(rho_fp)["purity"])
        purs = np.array(purs)
        ab_results.append({
            "eps": float(eps),
            "mean": float(purs.mean()),
            "std": float(purs.std()),
            "max": float(purs.max()),
            "min": float(purs.min()),
        })
        print(f"  {eps:5.2f}  {purs.mean():10.4f}  {purs.std():10.4f}  "
              f"{purs.max():10.4f}  {purs.min():10.4f}")
    print()

    # ============================================================
    # Phase AL: A-W algorithm: state distinguisher under amplification
    # ============================================================
    print("=" * 70)
    print("Phase AL: Aaronson-Watrous state-distinguisher implementation")
    print("=" * 70)
    print()

    # Setup: two close mixed states. Encode binary classification problem:
    # sigma_a = |0><0|;   sigma_b = (1-eps)|0><0| + eps|1><1|.
    # Helstrom success probability ~ 0.5 + eps/4.
    # Question: with D-CTC + Clifford U, can we succeed at much higher prob?

    print("Setup: two close states sigma_a, sigma_b on CR = qubit")
    print("       sigma_a = |0><0|")
    print("       sigma_b = (1-eps)|0><0| + eps|1><1|")
    print()

    # Sweep eps (input separation)
    epsvals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    print(f"{'eps':6s} {'helstrom':10s} {'CTC_clifford':14s} {'CTC_haar':10s} {'speedup':9s}")
    al_results = []
    rng_AL = np.random.default_rng(202)
    for eps in epsvals:
        sigma_a = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_a[0, 0] = 1.0
        sigma_b = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_b[0, 0] = 1 - eps; sigma_b[1, 1] = eps

        input_td = trace_distance(sigma_a, sigma_b)
        helstrom = 0.5 + input_td / 2.0

        # Best clifford channel for amplification
        best_amp_cliff = 0.0
        for _ in range(50):
            U = clifford_like_unitary(dim_total, rng_AL)
            rho_a = get_fixed_point(U, sigma_a, dim_cr, rho_init)
            rho_b = get_fixed_point(U, sigma_b, dim_cr, rho_init)
            output_td = trace_distance(rho_a, rho_b)
            best_amp_cliff = max(best_amp_cliff, output_td)

        # Best haar
        best_amp_haar = 0.0
        for _ in range(50):
            U = haar_random_unitary(dim_total, rng_AL)
            rho_a = get_fixed_point(U, sigma_a, dim_cr, rho_init)
            rho_b = get_fixed_point(U, sigma_b, dim_cr, rho_init)
            output_td = trace_distance(rho_a, rho_b)
            best_amp_haar = max(best_amp_haar, output_td)

        ctc_succ_cliff = 0.5 + best_amp_cliff / 2.0
        ctc_succ_haar  = 0.5 + best_amp_haar  / 2.0
        speedup_cliff  = (ctc_succ_cliff - 0.5) / (helstrom - 0.5)
        al_results.append({
            "eps": float(eps),
            "input_trace_distance": float(input_td),
            "helstrom_success": float(helstrom),
            "ctc_success_clifford": float(ctc_succ_cliff),
            "ctc_success_haar": float(ctc_succ_haar),
            "speedup_clifford": float(speedup_cliff),
        })
        print(f"  {eps:.2f}    {helstrom:8.4f}    {ctc_succ_cliff:12.4f}    "
              f"{ctc_succ_haar:8.4f}  {speedup_cliff:7.2f}")

    print()
    print("speedup = (ctc - 0.5) / (helstrom - 0.5)")
    print("  > 1 means D-CTC amplifies the success rate beyond Helstrom.")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    # Find slope of speedup vs input distance
    speedups = np.array([r["speedup_clifford"] for r in al_results])
    if speedups.max() > 1.5:
        print(f"Speedup factor > 1.5 observed at some input separations.")
        print("Clifford D-CTC channels EMPIRICALLY amplify state distinguishability")
        print("beyond Helstrom in polynomial time.")
    else:
        print("Speedup factor near 1 -- D-CTC matches Helstrom but does not exceed.")

    print()
    out = Path("examples") / "dctc_deep_phase_abal_results.json"
    with open(out, "w") as f:
        json.dump({
            "phase_AB": ab_results,
            "phase_AL": al_results,
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
