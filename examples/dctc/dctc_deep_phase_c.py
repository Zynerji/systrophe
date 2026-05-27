"""D-CTC deep exploration --- Phase C: structure of high-purity unitaries.

Generates 1000 Haar-random unitaries at (dim_CR=2, dim_CTC=3),
extracts the top-20 by fixed-point purity, and analyses what makes
them special.

Diagnostics per unitary:
  - lambda_2(E)              -- spectral gap
  - lambda_1(E)              -- principal eigenvalue (should be 1)
  - dominant eigenvector rank (rank of E's principal eigenmode as a
                                density matrix)
  - CR-CTC Schmidt entropy of U (treating U as a tensor on CR x CTC)
  - condition number of U

The question: do near-pure fixed points (purity > 0.9) share a
characteristic structural signature?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import dctc_fixed_point, density_matrix_diagnostics


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def channel_superoperator(
    U: np.ndarray, sigma_cr: np.ndarray, dim_cr: int, dim_ctc: int
) -> np.ndarray:
    d2 = dim_ctc * dim_ctc
    M = np.zeros((d2, d2), dtype=complex)
    for k in range(d2):
        i, j = divmod(k, dim_ctc)
        rho_basis = np.zeros((dim_ctc, dim_ctc), dtype=complex)
        rho_basis[i, j] = 1.0
        joint = np.kron(sigma_cr, rho_basis)
        out = U @ joint @ U.conj().T
        out_resh = out.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
        result = np.einsum("aiaj->ij", out_resh)
        M[:, k] = result.reshape(d2)
    return M


def schmidt_entropy_of_U(U: np.ndarray, dim_cr: int, dim_ctc: int) -> float:
    """Schmidt entropy of U treated as a vector on (CR x CTC) x (CR x CTC).

    Larger entropy => U entangles CR and CTC more strongly.
    """
    # U: (dim_cr * dim_ctc) x (dim_cr * dim_ctc)
    # Reshape U as a 4-index tensor: out_cr, out_ctc, in_cr, in_ctc
    d2 = dim_cr * dim_ctc
    U_tensor = U.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
    # Treat as a bipartite operator (out=CR_in_CR, out_ctc_in_ctc):
    # bipartition into (CR_out, CR_in) vs (CTC_out, CTC_in)
    U_bi = U_tensor.transpose((0, 2, 1, 3)).reshape(
        (dim_cr * dim_cr, dim_ctc * dim_ctc))
    svals = np.linalg.svd(U_bi, compute_uv=False)
    svals = svals[svals > 1e-12]
    pvals = svals ** 2 / np.sum(svals ** 2)
    return float(-np.sum(pvals * np.log(pvals)))


def density_matrix_rank(rho: np.ndarray, eps: float = 1e-6) -> int:
    eigs = np.linalg.eigvalsh(rho).real
    return int(np.sum(eigs > eps))


def main():
    print("=" * 70)
    print("D-CTC Phase C: structure of high-purity unitaries")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    n_samples = 1000
    print(f"Sampling {n_samples} Haar-random U at dim_CR={dim_cr}, dim_CTC={dim_ctc}")

    rng = np.random.default_rng(2026)
    dim_total = dim_cr * dim_ctc

    records = []
    t0 = time.time()
    for k in range(n_samples):
        U = haar_random_unitary(dim_total, rng)
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())

        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                              rho_ctc_init=rho_init, tol=1e-10,
                              max_iter=5000)
        rho_fp = r["rho_ctc"]
        diag = density_matrix_diagnostics(rho_fp)
        # Spectral analysis
        E_mat = channel_superoperator(U, sigma_cr, dim_cr, dim_ctc)
        eigs_E = np.linalg.eigvals(E_mat)
        eigs_E_abs = np.abs(eigs_E)
        order = np.argsort(-eigs_E_abs)
        l1 = eigs_E_abs[order[0]]
        l2 = eigs_E_abs[order[1]] if len(eigs_E_abs) > 1 else 0.0
        l3 = eigs_E_abs[order[2]] if len(eigs_E_abs) > 2 else 0.0

        sch = schmidt_entropy_of_U(U, dim_cr, dim_ctc)
        rank_fp = density_matrix_rank(rho_fp)

        records.append({
            "idx": k,
            "iter": r["iterations"],
            "purity": diag["purity"],
            "entropy": float(-np.sum(np.linalg.eigvalsh(rho_fp).real
                                       * np.log(np.clip(np.linalg.eigvalsh(rho_fp).real, 1e-15, None)))),
            "lambda_1": float(l1),
            "lambda_2": float(l2),
            "lambda_3": float(l3),
            "schmidt_entropy_U": sch,
            "rank_fixed_point": rank_fp,
            "converged": bool(r["converged"]),
        })
    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.1f}s")
    print()

    records_sorted = sorted(records, key=lambda r: -r["purity"])
    top20 = records_sorted[:20]
    bottom20 = records_sorted[-20:]

    print("Top-20 highest-purity fixed points:")
    print(f"  {'idx':5s} {'iter':5s} {'purity':7s} {'lambda_2':9s} {'schmidt_U':10s} {'rank_fp':7s}")
    for r in top20:
        print(f"  {r['idx']:5d} {r['iter']:5d} {r['purity']:7.4f} "
              f"{r['lambda_2']:9.4f} {r['schmidt_entropy_U']:10.4f} {r['rank_fixed_point']:7d}")
    print()

    print("Bottom-20 (most-mixed fixed points):")
    print(f"  {'idx':5s} {'iter':5s} {'purity':7s} {'lambda_2':9s} {'schmidt_U':10s} {'rank_fp':7s}")
    for r in bottom20:
        print(f"  {r['idx']:5d} {r['iter']:5d} {r['purity']:7.4f} "
              f"{r['lambda_2']:9.4f} {r['schmidt_entropy_U']:10.4f} {r['rank_fixed_point']:7d}")
    print()

    # Aggregate statistics
    top_l2 = np.array([r["lambda_2"] for r in top20])
    bot_l2 = np.array([r["lambda_2"] for r in bottom20])
    top_sch = np.array([r["schmidt_entropy_U"] for r in top20])
    bot_sch = np.array([r["schmidt_entropy_U"] for r in bottom20])
    top_iter = np.array([r["iter"] for r in top20])
    bot_iter = np.array([r["iter"] for r in bottom20])

    print("Aggregate comparison:")
    print(f"  top-20  : mean |lambda_2| = {top_l2.mean():.4f},  Schmidt = {top_sch.mean():.4f},  iter = {top_iter.mean():.1f}")
    print(f"  bottom-20: mean |lambda_2| = {bot_l2.mean():.4f},  Schmidt = {bot_sch.mean():.4f},  iter = {bot_iter.mean():.1f}")
    print()

    # Distributions of all features
    all_purity = np.array([r["purity"] for r in records])
    all_l2 = np.array([r["lambda_2"] for r in records])
    all_sch = np.array([r["schmidt_entropy_U"] for r in records])
    all_iter = np.array([r["iter"] for r in records])
    all_rank = np.array([r["rank_fixed_point"] for r in records])

    print("Distribution stats:")
    print(f"  purity:           mean={all_purity.mean():.3f},  std={all_purity.std():.3f},  max={all_purity.max():.3f}")
    print(f"  |lambda_2|:       mean={all_l2.mean():.3f},  std={all_l2.std():.3f},  max={all_l2.max():.3f}")
    print(f"  schmidt_U:        mean={all_sch.mean():.3f},  std={all_sch.std():.3f}")
    print(f"  rank fixed-point: mean={all_rank.mean():.3f},  max-rank fraction={float(np.mean(all_rank == dim_ctc)):.3f}")
    print()

    # Correlations
    pearson_pur_l2  = float(np.corrcoef(all_purity, all_l2)[0, 1])
    pearson_pur_sch = float(np.corrcoef(all_purity, all_sch)[0, 1])
    pearson_pur_iter= float(np.corrcoef(all_purity, all_iter)[0, 1])
    pearson_l2_sch  = float(np.corrcoef(all_l2, all_sch)[0, 1])

    print("Pearson correlations:")
    print(f"  purity vs |lambda_2|:    {pearson_pur_l2:+.4f}")
    print(f"  purity vs Schmidt(U):    {pearson_pur_sch:+.4f}")
    print(f"  purity vs iter count:    {pearson_pur_iter:+.4f}")
    print(f"  |lambda_2| vs Schmidt(U):{pearson_l2_sch:+.4f}")
    print()

    # How many fixed points are nearly rank-1?
    near_pure_count = int(np.sum(all_purity > 0.9))
    pure_state_count = int(np.sum(all_purity > 0.99))
    near_max_mixed = int(np.sum(np.abs(all_purity - 1.0/dim_ctc) < 0.05))
    print(f"Counts (of {n_samples}):")
    print(f"  purity > 0.99 (nearly pure):  {pure_state_count}")
    print(f"  purity > 0.9 (highly pure):   {near_pure_count}")
    print(f"  purity close to 1/{dim_ctc} (max-mixed): {near_max_mixed}")
    print()

    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print()
    if pearson_pur_sch < -0.3:
        print(f"Schmidt entropy of U is *negatively* correlated with purity")
        print(f"(r = {pearson_pur_sch:+.3f}). High-purity D-CTC fixed points come from")
        print("LESS-entangling U's --- the channel preserves coherence because U")
        print("does not strongly couple CR and CTC.")
    elif pearson_pur_sch > 0.3:
        print(f"Schmidt entropy of U is positively correlated with purity (r = {pearson_pur_sch:+.3f}) --- ")
        print("higher entanglement leads to purer fixed points. Counterintuitive.")
    else:
        print(f"Schmidt entropy of U weakly correlates with purity (r = {pearson_pur_sch:+.3f}).")
    print()

    out_path = Path("examples") / "dctc_deep_phase_c_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"dim_cr": dim_cr, "dim_ctc": dim_ctc, "n_samples": n_samples},
            "top_20": top20,
            "bottom_20": bottom20,
            "correlations": {
                "purity_vs_lambda2": pearson_pur_l2,
                "purity_vs_schmidt_U": pearson_pur_sch,
                "purity_vs_iter": pearson_pur_iter,
                "lambda2_vs_schmidt_U": pearson_l2_sch,
            },
            "counts": {
                "purity_gt_0.99": pure_state_count,
                "purity_gt_0.9": near_pure_count,
                "purity_near_maxmixed": near_max_mixed,
            },
            "distribution_stats": {
                "purity": {"mean": float(all_purity.mean()), "std": float(all_purity.std()),
                           "max": float(all_purity.max())},
                "lambda_2": {"mean": float(all_l2.mean()), "std": float(all_l2.std()),
                              "max": float(all_l2.max())},
                "schmidt_U": {"mean": float(all_sch.mean()), "std": float(all_sch.std())},
            },
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
