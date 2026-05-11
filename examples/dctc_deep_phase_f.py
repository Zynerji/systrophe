"""D-CTC Phase F: refined Kraus-structure diagnostics.

Phase E showed that small sigma_min([K_0, K_1]) is *necessary but not
sufficient* for high-purity fixed point. Some samples have small
commutator AND max-mixed fixed point.

Phase F refines the diagnostic with three additional structural
measures:

1. JADE-style joint-diagonalization residual: measures how close the
   Kraus operators are to simultaneous diagonalization in a single
   eigenbasis.

2. Common-eigenvector consistency: when a near-common eigenvector
   exists, is it actually the principal eigenvector of E?

3. Spectral fingerprint: ratio |lambda_1 - lambda_2| / |lambda_2|
   measures the dominance of the principal eigenvalue.

The expectation: high purity correlates with (a) small JADE residual
AND (b) the common eigenvector matching the principal eigenmode.
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


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def kraus_operators(U: np.ndarray, sigma_cr: np.ndarray, dim_cr: int) -> list[np.ndarray]:
    """Kraus operators for pure sigma_CR = |0><0|."""
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


def jade_residual(K_list: list[np.ndarray], max_iter: int = 100) -> float:
    """JADE-style joint-diagonalisation residual.

    Find a unitary V such that V^dag K_a V has small off-diagonal
    elements for all a. Returns the total off-diagonal Frobenius norm
    after the best simultaneous diagonalisation.

    Lower residual => Kraus operators are more nearly simultaneously
    diagonalisable => system has near-common eigenvector(s).
    """
    n = len(K_list)
    if n < 1:
        return 0.0
    dim = K_list[0].shape[0]
    # Initialise V = identity; iteratively diagonalise each K
    V = np.eye(dim, dtype=complex)
    for _ in range(max_iter):
        # Build the average of K_a^dag K_a and diagonalise that
        H = np.zeros((dim, dim), dtype=complex)
        for K in K_list:
            H = H + (K @ K.conj().T) / n
        # Eigendecompose H and apply
        eigvals, U_step = np.linalg.eigh(H)
        V = V @ U_step
        # Apply rotation to all K
        K_list = [U_step.conj().T @ K @ U_step for K in K_list]
        # Stopping criterion: off-diagonal norm
        off_diag = 0.0
        for K in K_list:
            off_diag += float(np.sum(np.abs(K - np.diag(np.diag(K))) ** 2))
        if off_diag < 1e-12:
            break
    # Final residual
    off_diag = 0.0
    for K in K_list:
        off_diag += float(np.sum(np.abs(K - np.diag(np.diag(K))) ** 2))
    return float(np.sqrt(off_diag))


def common_eigenvector_alignment(K_list: list[np.ndarray]) -> dict:
    """Find a candidate common eigenvector via the joint kernel of
    all pairwise commutators, and measure how 'eigenvector-like' it
    is to each Kraus operator.

    Returns:
      sigma_min_joint: smallest singular value of stacked commutators
      alignment_score: max(|K_a |psi>|^2 / (||K_a|| ||psi||)^2) - 1
                       (closer to 0 means psi is closer to a common eig.)
    """
    n = len(K_list)
    if n < 2:
        return {"sigma_min_joint": float("inf"), "alignment_score": 0.0}
    blocks = []
    for i in range(n):
        for j in range(i + 1, n):
            blocks.append(K_list[i] @ K_list[j] - K_list[j] @ K_list[i])
    M = np.vstack(blocks)
    U_svd, S, Vh = np.linalg.svd(M, full_matrices=False)
    sigma_min = float(S[-1])
    # Candidate common eigenvector: right-singular vector of smallest sv
    psi = Vh.conj().T[:, -1]
    # Measure: ||K_a psi - lambda_a psi|| for each K_a
    residuals = []
    for K in K_list:
        Kp = K @ psi
        # Project out psi component
        proj = np.vdot(psi, Kp) * psi
        residual_norm = float(np.linalg.norm(Kp - proj))
        residuals.append(residual_norm)
    alignment = float(max(residuals))
    return {"sigma_min_joint": sigma_min, "alignment_score": alignment,
            "candidate_psi_norm": float(np.linalg.norm(psi))}


def spectral_fingerprint(U: np.ndarray, sigma_cr: np.ndarray, dim_cr: int) -> dict:
    """Spectral structure of E: principal eigenvalue, gap, and how
    rank-1 the principal eigenvector is when reshaped as density matrix.
    """
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    M = channel_superoperator(U, sigma_cr, dim_cr)
    eigvals_c, eigvecs_c = np.linalg.eig(M)
    # Sort by descending |eigvalue|
    order = np.argsort(-np.abs(eigvals_c))
    l1 = complex(eigvals_c[order[0]])
    l2 = complex(eigvals_c[order[1]]) if len(eigvals_c) > 1 else 0
    principal_vec = eigvecs_c[:, order[0]]
    # Reshape as (dim_ctc, dim_ctc); this is the principal eigenmode rho_principal
    rho_pr = principal_vec.reshape((dim_ctc, dim_ctc))
    # The fixed-point density matrix is rho_pr / Tr[rho_pr] (real part)
    rho_pr_normalised = rho_pr / np.trace(rho_pr)
    # Make Hermitian (numerical noise can introduce skew)
    rho_pr_normalised = 0.5 * (rho_pr_normalised + rho_pr_normalised.conj().T)
    eig_rho = np.sort(np.linalg.eigvalsh(rho_pr_normalised).real)[::-1]
    # Concentration on dominant eigenvalue
    concentration = float(eig_rho[0])  # closer to 1 = pure
    # Spectral gap
    gap = abs(l1) - abs(l2)
    return {
        "lambda_1_abs": float(abs(l1)),
        "lambda_2_abs": float(abs(l2)),
        "gap": float(gap),
        "rho_principal_top_eig": concentration,
        "rho_principal_rank_eigs": eig_rho.tolist(),
    }


def main():
    print("=" * 70)
    print("D-CTC Phase F: refined Kraus-structure diagnostics")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    n_samples = 1500
    print(f"Generating {n_samples} samples at dim_CR={dim_cr}, dim_CTC={dim_ctc}")

    rng = np.random.default_rng(456)
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
        purity = density_matrix_diagnostics(rho_fp)["purity"]

        K_list = kraus_operators(U, sigma_cr, dim_cr)
        align = common_eigenvector_alignment(K_list)
        jade_res = jade_residual([K.copy() for K in K_list])  # copy because jade mutates
        spec = spectral_fingerprint(U, sigma_cr, dim_cr)
        records.append({
            "idx": int(k),
            "iter": int(r["iterations"]),
            "purity": float(purity),
            "sigma_min_joint": float(align["sigma_min_joint"]),
            "alignment_score": float(align["alignment_score"]),
            "jade_residual": float(jade_res),
            "lambda_2": float(spec["lambda_2_abs"]),
            "rho_principal_top_eig": float(spec["rho_principal_top_eig"]),
        })
    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.1f}s\n")

    # Convert to numpy
    purities = np.array([r["purity"] for r in records])
    sigma_min_joints = np.array([r["sigma_min_joint"] for r in records])
    alignments = np.array([r["alignment_score"] for r in records])
    jade_resids = np.array([r["jade_residual"] for r in records])
    lambda_2s = np.array([r["lambda_2"] for r in records])
    rho_prs = np.array([r["rho_principal_top_eig"] for r in records])

    print("Distribution stats:")
    print(f"  sigma_min_joint   : mean={sigma_min_joints.mean():.4f}, "
          f"min={sigma_min_joints.min():.4f}, max={sigma_min_joints.max():.4f}")
    print(f"  alignment_score   : mean={alignments.mean():.4f}, "
          f"min={alignments.min():.4f}, max={alignments.max():.4f}")
    print(f"  jade_residual     : mean={jade_resids.mean():.4f}, "
          f"min={jade_resids.min():.4f}, max={jade_resids.max():.4f}")
    print(f"  rho_principal_top : mean={rho_prs.mean():.4f}, "
          f"min={rho_prs.min():.4f}, max={rho_prs.max():.4f}")
    print()

    # Correlations
    pearson_sigma = float(np.corrcoef(purities, -np.log(np.maximum(sigma_min_joints, 1e-12)))[0, 1])
    pearson_align = float(np.corrcoef(purities, -np.log(np.maximum(alignments, 1e-12)))[0, 1])
    pearson_jade = float(np.corrcoef(purities, -np.log(np.maximum(jade_resids, 1e-12)))[0, 1])
    pearson_rho = float(np.corrcoef(purities, rho_prs)[0, 1])

    print("Pearson correlations with purity:")
    print(f"  -log(sigma_min_joint)            : {pearson_sigma:+.4f}")
    print(f"  -log(alignment_score)            : {pearson_align:+.4f}")
    print(f"  -log(jade_residual)              : {pearson_jade:+.4f}")
    print(f"  rho_principal_top_eig (direct)   : {pearson_rho:+.4f}")
    print()

    # Sort by alignment score (smallest = best alignment = should be purest)
    align_sorted_idx = np.argsort(alignments)[:20]
    print("Top-20 best-aligned samples (smallest alignment_score):")
    print(f"  {'idx':5s} {'purity':7s} {'align':9s} {'jade':9s} {'rho_pr':9s}")
    for i in align_sorted_idx[:10]:
        r = records[i]
        print(f"  {r['idx']:5d} {r['purity']:7.4f} {r['alignment_score']:9.5f} "
              f"{r['jade_residual']:9.4f} {r['rho_principal_top_eig']:9.4f}")
    print()

    # How does alignment_score predict purity?
    print("Conditional probability:")
    for thresh in (0.05, 0.1, 0.2, 0.5):
        mask_low_align = alignments < thresh
        n_low = mask_low_align.sum()
        if n_low == 0:
            continue
        pure_fraction = float(np.mean(purities[mask_low_align] > 0.9))
        print(f"  P(purity > 0.9 | alignment < {thresh}): "
              f"{pure_fraction:.3f}  (n = {n_low})")
    print()

    # Critical: does alignment_score CAUSE high purity?
    # Run a stricter test - all near-zero alignments should be pure.
    near_zero_align = alignments < 0.01
    print(f"Samples with alignment_score < 0.01: {near_zero_align.sum()}")
    if near_zero_align.sum() > 0:
        purs = purities[near_zero_align]
        print(f"  their purities: min={purs.min():.3f}, max={purs.max():.3f}, "
              f"mean={purs.mean():.3f}")
        print(f"  fraction with purity > 0.9: {float(np.mean(purs > 0.9)):.3f}")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    if abs(pearson_align) > 0.7:
        sign = "POSITIVE" if pearson_align > 0 else "NEGATIVE"
        print(f"STRONG {sign} correlation (r = {pearson_align:+.3f}) between purity")
        print(f"and -log(alignment_score). Hypothesis: near-common-eigenvector")
        print(f"alignment IS the structural signature of high-purity D-CTC.")
    elif abs(pearson_align) > 0.4:
        sign = "positive" if pearson_align > 0 else "negative"
        print(f"Moderate {sign} correlation (r = {pearson_align:+.3f}) for alignment.")
    else:
        print(f"Weak alignment correlation (r = {pearson_align:+.3f}).")

    print()
    if pearson_rho > 0.99:
        print(f"r(purity, rho_principal_top_eig) = {pearson_rho:.4f}")
        print("The principal eigenvalue of the reshaped fixed-point density")
        print("matrix is essentially equal to the purity itself (definitional).")
    print()

    out_path = Path("examples") / "dctc_deep_phase_f_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"dim_cr": dim_cr, "dim_ctc": dim_ctc, "n_samples": n_samples},
            "correlations": {
                "pearson_purity_vs_neg_log_sigma_min_joint": pearson_sigma,
                "pearson_purity_vs_neg_log_alignment": pearson_align,
                "pearson_purity_vs_neg_log_jade_residual": pearson_jade,
                "pearson_purity_vs_rho_principal_top": pearson_rho,
            },
            "stats": {
                "alignment_score": {
                    "mean": float(alignments.mean()),
                    "min": float(alignments.min()),
                    "max": float(alignments.max()),
                },
                "jade_residual": {
                    "mean": float(jade_resids.mean()),
                    "min": float(jade_resids.min()),
                    "max": float(jade_resids.max()),
                },
            },
            "top10_aligned": [records[int(i)] for i in align_sorted_idx[:10]],
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
