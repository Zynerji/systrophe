"""D-CTC Phase K + N + P + AI batch.

K: 3-Kraus channels (dim_CR=3). Triple-overlap vs purity.
N: rho_init independence -- does the iteration always converge to same fixed point?
P: Eigenvector IPR (localization) of E's principal eigenvector vs purity.
AI: Z_3 sigma_CR -- does structured sigma_CR change purity statistics?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.d_ctc import (
    channel_superoperator,
    dctc_fixed_point,
    density_matrix_diagnostics,
)
from systrophe.floquet_mobius import z3_cycle_shift


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


def kraus_operators(U, sigma_cr, dim_cr):
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    diag_sigma = np.diag(sigma_cr).real
    K_list = []
    U_tensor = U.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
    for a in range(dim_cr):
        if diag_sigma[a] < 1e-12:
            continue
        sqrt_pa = np.sqrt(diag_sigma[a])
        for b in range(dim_cr):
            K_ab = sqrt_pa * U_tensor[b, :, a, :]
            K_list.append(K_ab)
    return K_list


def participation_ratio(v: np.ndarray) -> float:
    """Inverse participation ratio of a normalised vector.

    IPR = sum |v_i|^4. IPR=1/n means uniform (delocalised); IPR=1 means
    fully localised on one basis vector.
    """
    v_norm = v / np.linalg.norm(v)
    return float(np.sum(np.abs(v_norm) ** 4))


def main():
    print("=" * 70)
    print("D-CTC Phase K + N + P + AI batch")
    print("=" * 70)
    print()

    rng = np.random.default_rng(8888)
    all_results = {}

    # ============================================================
    # Phase K: 3-Kraus channels (dim_CR=3)
    # ============================================================
    print("Phase K: 3-Kraus channels (dim_CR=3, dim_CTC=3)")
    print()
    dim_cr, dim_ctc = 3, 3
    dim_total = dim_cr * dim_ctc
    n_K = 1500
    purities_K = []
    triple_overlaps = []
    t0 = time.time()
    for _ in range(n_K):
        U = haar_random_unitary(dim_total, rng)
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                              rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
        purity = density_matrix_diagnostics(r["rho_ctc"])["purity"]
        purities_K.append(purity)

        # Triple overlap: best simultaneous-eigenvector candidates
        K_list = kraus_operators(U, sigma_cr, dim_cr)
        if len(K_list) >= 3:
            ev = [np.linalg.eig(K)[1] for K in K_list[:3]]
            # For each combination of eigenvector index across 3 Kraus, compute
            # geometric mean of pairwise overlaps
            best_triple = 0.0
            for i in range(dim_ctc):
                for j in range(dim_ctc):
                    for k in range(dim_ctc):
                        v0 = ev[0][:, i] / np.linalg.norm(ev[0][:, i])
                        v1 = ev[1][:, j] / np.linalg.norm(ev[1][:, j])
                        v2 = ev[2][:, k] / np.linalg.norm(ev[2][:, k])
                        ov = (abs(np.vdot(v0, v1)) * abs(np.vdot(v1, v2))
                                * abs(np.vdot(v2, v0))) ** (1/3)
                        best_triple = max(best_triple, float(ov))
            triple_overlaps.append(best_triple)
        else:
            triple_overlaps.append(np.nan)
    elapsed = time.time() - t0
    purities_K = np.array(purities_K)
    triple_overlaps = np.array(triple_overlaps)
    print(f"  n={n_K}, time={elapsed:.1f}s")
    print(f"  max purity:    {purities_K.max():.4f}")
    print(f"  P(>0.7):       {float(np.mean(purities_K > 0.7)):.4f}")
    print(f"  P(>0.9):       {float(np.mean(purities_K > 0.9)):.4f}")
    valid = ~np.isnan(triple_overlaps)
    pearson_triple = float(np.corrcoef(purities_K[valid], triple_overlaps[valid])[0, 1])
    print(f"  Pearson(purity, triple_overlap): {pearson_triple:+.4f}")
    print()
    all_results["phase_K"] = {
        "n": n_K, "max_purity": float(purities_K.max()),
        "P_gt_07": float(np.mean(purities_K > 0.7)),
        "P_gt_09": float(np.mean(purities_K > 0.9)),
        "pearson_purity_vs_triple_overlap": pearson_triple,
    }

    # ============================================================
    # Phase N: rho_init independence
    # ============================================================
    print("Phase N: rho_init independence (does fixed point depend on init?)")
    print()
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    n_U = 100
    n_init = 10
    rng_N = np.random.default_rng(123456)
    max_pairwise_distances = []
    for _ in range(n_U):
        U = haar_random_unitary(dim_total, rng_N)
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        fixed_points = []
        for _ in range(n_init):
            psi = rng_N.standard_normal(dim_ctc) + 1j * rng_N.standard_normal(dim_ctc)
            psi = psi / np.linalg.norm(psi)
            rho_init = np.outer(psi, psi.conj())
            r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                  rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
            fixed_points.append(r["rho_ctc"])
        # Compute max pairwise distance among fixed points
        max_d = 0.0
        for i in range(len(fixed_points)):
            for j in range(i + 1, len(fixed_points)):
                d = float(np.linalg.norm(fixed_points[i] - fixed_points[j], "fro"))
                max_d = max(max_d, d)
        max_pairwise_distances.append(max_d)
    max_pairwise_distances = np.array(max_pairwise_distances)
    print(f"  Over {n_U} different U, {n_init} different rho_init each:")
    print(f"  Max pairwise distance of fixed points:")
    print(f"    mean: {max_pairwise_distances.mean():.6f}")
    print(f"    max:  {max_pairwise_distances.max():.6f}")
    print(f"    median: {np.median(max_pairwise_distances):.6f}")
    print(f"  P(max_distance < 1e-6): {float(np.mean(max_pairwise_distances < 1e-6)):.4f}")
    print(f"    -> {int(100 * np.mean(max_pairwise_distances < 1e-6))}% of channels have unique fixed point")
    print()
    all_results["phase_N"] = {
        "n_U": n_U, "n_init": n_init,
        "mean_max_distance": float(max_pairwise_distances.mean()),
        "max_max_distance": float(max_pairwise_distances.max()),
        "fraction_unique": float(np.mean(max_pairwise_distances < 1e-6)),
    }

    # ============================================================
    # Phase P: principal eigenvector IPR
    # ============================================================
    print("Phase P: principal eigenvector IPR (Anderson-localization analog)")
    print()
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    n_P = 1500
    rng_P = np.random.default_rng(54321)
    purities_P = []
    iprs = []
    for _ in range(n_P):
        U = haar_random_unitary(dim_total, rng_P)
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        psi = rng_P.standard_normal(dim_ctc) + 1j * rng_P.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                              rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
        rho_fp = r["rho_ctc"]
        purity = density_matrix_diagnostics(rho_fp)["purity"]
        purities_P.append(purity)
        # IPR of principal eigenvector of E
        E_mat = channel_superoperator(U, sigma_cr, dim_cr)
        eigvals, eigvecs = np.linalg.eig(E_mat)
        order = np.argsort(-np.abs(eigvals))
        principal_v = eigvecs[:, order[0]]
        iprs.append(participation_ratio(principal_v))
    purities_P = np.array(purities_P)
    iprs = np.array(iprs)
    pearson_ipr = float(np.corrcoef(purities_P, iprs)[0, 1])
    print(f"  n={n_P}")
    print(f"  IPR stats: mean={iprs.mean():.4f}, min={iprs.min():.4f}, max={iprs.max():.4f}")
    print(f"  Uniform delocalisation: 1/{dim_ctc**2} = {1/dim_ctc**2:.4f}")
    print(f"  Pearson(purity, IPR): {pearson_ipr:+.4f}")
    print()
    all_results["phase_P"] = {
        "n": n_P,
        "ipr_mean": float(iprs.mean()),
        "ipr_max": float(iprs.max()),
        "pearson_purity_vs_ipr": pearson_ipr,
    }

    # ============================================================
    # Phase AI: Z_3 cycle-shift as sigma_CR (only sensible for dim_CR=3)
    # ============================================================
    print("Phase AI: structured sigma_CR (dim_CR=3 with various sigma_CR)")
    print()
    dim_cr, dim_ctc = 3, 3
    dim_total = dim_cr * dim_ctc
    n_AI = 500
    rng_AI = np.random.default_rng(98765)

    cases = {
        "pure_|0>": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=complex),
        "uniform_max_mixed": np.eye(3, dtype=complex) / 3,
        "Z3_eigvec_branch0": None,
        "Z3_eigvec_branch1": None,
    }
    # Build Z_3 eigenvectors
    S = z3_cycle_shift()
    eigvals, eigvecs = np.linalg.eig(S)
    for i, ev in enumerate(eigvals):
        bname = f"Z3_eigvec_branch{i}"
        if bname in cases:
            cases[bname] = np.outer(eigvecs[:, i], eigvecs[:, i].conj())

    for case_name, sigma_cr in cases.items():
        if sigma_cr is None:
            continue
        sigma_cr = sigma_cr.astype(complex)
        # Ensure Hermitian and unit trace
        sigma_cr = 0.5 * (sigma_cr + sigma_cr.conj().T)
        sigma_cr = sigma_cr / np.trace(sigma_cr)
        prs = []
        for _ in range(n_AI):
            U = haar_random_unitary(dim_total, rng_AI)
            psi = rng_AI.standard_normal(dim_ctc) + 1j * rng_AI.standard_normal(dim_ctc)
            psi = psi / np.linalg.norm(psi)
            rho_init = np.outer(psi, psi.conj())
            r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                  rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
            prs.append(density_matrix_diagnostics(r["rho_ctc"])["purity"])
        prs = np.array(prs)
        print(f"  sigma_CR = {case_name:25s}: mean purity={prs.mean():.4f}, "
              f"max={prs.max():.4f}, P(>0.7)={float(np.mean(prs > 0.7)):.4f}")
    print()

    all_results["phase_AI"] = "see stdout"  # already printed

    # ============================================================
    # Summary
    # ============================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"K (3-Kraus dim_CR=3): max purity {all_results['phase_K']['max_purity']:.3f}, "
          f"P(>0.9) = {all_results['phase_K']['P_gt_09']:.4f}, "
          f"triple-overlap r = {all_results['phase_K']['pearson_purity_vs_triple_overlap']:+.3f}")
    print(f"N (rho_init independence): {all_results['phase_N']['fraction_unique']*100:.0f}% of channels")
    print(f"   have unique fixed point (max pairwise dist < 1e-6)")
    print(f"P (IPR correlation): Pearson r = {all_results['phase_P']['pearson_purity_vs_ipr']:+.3f}")
    print()

    out = Path("examples") / "dctc_deep_phase_knpai_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
