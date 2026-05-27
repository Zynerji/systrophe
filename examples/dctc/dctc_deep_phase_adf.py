"""D-CTC Phase AD + AF.

AD: Anderson mixing acceleration of the iteration.
AF: Holevo classical capacity of the D-CTC channel.

Hypothesis (AF): high-purity Clifford D-CTC channels should have
specific (high or low) Holevo capacity reflecting their structured
information-preservation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import (
    apply_channel,
    dctc_fixed_point,
    density_matrix_diagnostics,
)
from systrophe.catchers.novelty_catcher import catch_novelty_in_named_arrays


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


def von_neumann_entropy(rho):
    eigs = np.linalg.eigvalsh(rho).real
    eigs = np.clip(eigs, 1e-15, None)
    return float(-np.sum(eigs * np.log(eigs)))


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def picard_iteration(U, sigma_cr, dim_cr, rho_init, max_iter=2000, tol=1e-10):
    """Standard Picard iteration."""
    rho = rho_init.copy()
    for k in range(max_iter):
        rho_new = apply_channel(U, sigma_cr, rho, dim_cr)
        tr = np.trace(rho_new)
        rho_new = rho_new / tr if abs(tr) > 1e-30 else rho_new
        if np.linalg.norm(rho_new - rho) < tol:
            return rho_new, k + 1
        rho = rho_new
    return rho, max_iter


def anderson_iteration(U, sigma_cr, dim_cr, rho_init, history=3,
                          max_iter=2000, tol=1e-10):
    """Anderson-style acceleration.

    Combine the last `history` iterates to minimise the residual.
    """
    dim_ctc = rho_init.shape[0]
    rho = rho_init.copy()
    rhos = [rho]
    Fs = []
    for k in range(max_iter):
        rho_new = apply_channel(U, sigma_cr, rho, dim_cr)
        tr = np.trace(rho_new)
        rho_new = rho_new / tr if abs(tr) > 1e-30 else rho_new
        F = rho_new - rho
        Fs.append(F)
        if np.linalg.norm(F) < tol:
            return rho_new, k + 1
        # Anderson combination of last `history` F's
        m = min(history, len(Fs))
        if m >= 2:
            # Solve min_alpha ||sum alpha_i F_{-i}||^2 subject to sum alpha = 1
            FF = np.array([F.reshape(-1) for F in Fs[-m:]])
            G = FF @ FF.conj().T
            ones = np.ones(m)
            try:
                lam = np.linalg.solve(G + 1e-12 * np.eye(m), ones)
                alpha = lam / np.sum(lam)
                accel = np.zeros_like(rho)
                for i, a in enumerate(alpha):
                    accel = accel + a * rhos[-m + i]
                rho = accel
            except np.linalg.LinAlgError:
                rho = rho_new
        else:
            rho = rho_new
        rhos.append(rho)
    return rho, max_iter


def main():
    print("=" * 70)
    print("Phase AD + AF: Anderson mixing + Holevo capacity")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
    sigma_cr[0, 0] = 1.0
    rng = np.random.default_rng(4747)

    # ============================================================
    # Phase AD: Anderson acceleration
    # ============================================================
    print("Phase AD: Anderson mixing vs standard iteration")
    print()
    n_AD = 100
    psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
    psi = psi / np.linalg.norm(psi)
    rho_init = np.outer(psi, psi.conj())

    iters_standard = []
    iters_anderson = []
    for _ in range(n_AD):
        U = haar_random_unitary(dim_total, rng)
        _, k1 = picard_iteration(U, sigma_cr, dim_cr, rho_init, max_iter=2000)
        _, k2 = anderson_iteration(U, sigma_cr, dim_cr, rho_init, history=4, max_iter=2000)
        iters_standard.append(k1)
        iters_anderson.append(k2)
    iters_standard = np.array(iters_standard)
    iters_anderson = np.array(iters_anderson)
    print(f"  n={n_AD}")
    print(f"  Standard:  mean iter = {iters_standard.mean():.1f}, median = {np.median(iters_standard):.1f}")
    print(f"  Anderson:  mean iter = {iters_anderson.mean():.1f}, median = {np.median(iters_anderson):.1f}")
    if iters_standard.mean() > 0:
        accel = iters_standard.mean() / max(iters_anderson.mean(), 1)
        print(f"  Acceleration factor: {accel:.2f}x")
    print()

    # ============================================================
    # Phase AF: Holevo classical capacity (estimate)
    # ============================================================
    print("Phase AF: Holevo classical capacity")
    print()
    print("  Estimating chi(E) = S(E(p_i rho_i)) - sum p_i S(E(rho_i))")
    print("  with rho_i = |i><i| (computational basis) and uniform p_i.")
    print()

    n_AF = 200
    holevos_haar = []
    holevos_cliff = []
    purities_haar = []
    purities_cliff = []
    rng_AF = np.random.default_rng(848)
    for _ in range(n_AF):
        for kind in ("haar", "clifford"):
            if kind == "haar":
                U = haar_random_unitary(dim_total, rng_AF)
            else:
                U = clifford_like_unitary(dim_total, rng_AF)

            # Compute E(|i><i|) for each computational basis state on CTC
            outputs = []
            for i in range(dim_ctc):
                rho_in = np.zeros((dim_ctc, dim_ctc), dtype=complex)
                rho_in[i, i] = 1.0
                rho_out = apply_channel(U, sigma_cr, rho_in, dim_cr)
                rho_out = 0.5 * (rho_out + rho_out.conj().T)  # symmetrise
                outputs.append(rho_out)

            # Mixture and average entropy
            avg_rho = sum(outputs) / dim_ctc
            S_avg = von_neumann_entropy(avg_rho)
            avg_S = sum(von_neumann_entropy(rho) for rho in outputs) / dim_ctc
            holevo = S_avg - avg_S

            # Also: get the fixed-point purity for context
            r_fp = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                      rho_ctc_init=rho_init, tol=1e-10, max_iter=2000)
            pur = density_matrix_diagnostics(r_fp["rho_ctc"])["purity"]

            if kind == "haar":
                holevos_haar.append(holevo)
                purities_haar.append(pur)
            else:
                holevos_cliff.append(holevo)
                purities_cliff.append(pur)

    holevos_haar = np.array(holevos_haar)
    holevos_cliff = np.array(holevos_cliff)
    purities_haar = np.array(purities_haar)
    purities_cliff = np.array(purities_cliff)

    print(f"  Haar:     Holevo mean = {holevos_haar.mean():.4f}, max = {holevos_haar.max():.4f}")
    print(f"  Clifford: Holevo mean = {holevos_cliff.mean():.4f}, max = {holevos_cliff.max():.4f}")
    print(f"  Max possible Holevo (= log dim_CTC) = {np.log(dim_ctc):.4f}")
    print()
    print(f"  Correlation(Haar purity vs Holevo):     {float(np.corrcoef(purities_haar, holevos_haar)[0,1]):+.4f}")
    print(f"  Correlation(Clifford purity vs Holevo): {float(np.corrcoef(purities_cliff, holevos_cliff)[0,1]):+.4f}")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    if holevos_cliff.mean() > 1.5 * holevos_haar.mean():
        print(f"Clifford channels have HIGHER mean Holevo capacity ({holevos_cliff.mean():.4f}")
        print(f"vs {holevos_haar.mean():.4f}). They preserve more classical")
        print("information through the D-CTC iteration.")
    elif holevos_cliff.mean() < 0.5 * holevos_haar.mean():
        print(f"Clifford channels have LOWER mean Holevo capacity.")
    else:
        print(f"Holevo capacity is comparable across Clifford and Haar.")

    # Native novelty catcher: AD iteration-count distributions +
    # AF Holevo / purity distributions for Haar and Clifford.
    novelty = catch_novelty_in_named_arrays({
        "AD_iters_standard": iters_standard,
        "AD_iters_anderson": iters_anderson,
        "AF_holevos_haar":   holevos_haar,
        "AF_holevos_cliff":  holevos_cliff,
        "AF_purities_haar":  purities_haar,
        "AF_purities_cliff": purities_cliff,
    })
    print()
    print(f"Novelty catcher: verdict='{novelty['verdict']}', "
          f"n_sharp={len(novelty['sharp_features'])}")

    out = Path("examples") / "dctc_deep_phase_adf_results.json"
    with open(out, "w") as f:
        json.dump({
            "phase_AD": {
                "n": n_AD,
                "standard_mean_iter": float(iters_standard.mean()),
                "anderson_mean_iter": float(iters_anderson.mean()),
                "acceleration_factor": float(iters_standard.mean() / max(iters_anderson.mean(), 1)),
            },
            "phase_AF": {
                "n": n_AF,
                "haar_holevo_mean": float(holevos_haar.mean()),
                "haar_holevo_max": float(holevos_haar.max()),
                "clifford_holevo_mean": float(holevos_cliff.mean()),
                "clifford_holevo_max": float(holevos_cliff.max()),
                "max_possible_holevo": float(np.log(dim_ctc)),
                "haar_pearson_pur_holevo": float(np.corrcoef(purities_haar, holevos_haar)[0,1]),
                "clifford_pearson_pur_holevo": float(np.corrcoef(purities_cliff, holevos_cliff)[0,1]),
            },
            "novelty_catcher": novelty,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
