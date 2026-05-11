"""D-CTC Phase G: explicit joint-Kraus-eigenvalue check.

For each Kraus-pair (K_0, K_1), find the candidate that simultaneously
minimises |K_0 psi - lambda_0 psi|^2 + |K_1 psi - lambda_1 psi|^2 over
choices of psi, lambda_0, lambda_1.

This is the "best joint eigenvector" of the pair. We then check:
(a) does this candidate match the principal eigenvector of E?
(b) does the combined eigenvalue (|lambda_0|^2 + |lambda_1|^2) equal 1?

A perfect match means the pure fixed-point hypothesis is structurally
correct.

Also: fits the purity distribution tail to characterise its shape.
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


def find_best_joint_eigenvector(K_list: list[np.ndarray]) -> dict:
    """Find best simultaneous-eigenvector candidate via eigenvalue search.

    For each Kraus operator K_a, find eigenvectors v_a^{(i)} with
    eigenvalues lambda_a^{(i)}. Then for each pair of eigenvectors
    (v_0^{(i)}, v_1^{(j)}), check overlap |<v_0^{(i)} | v_1^{(j)}>|.
    The best overlap pair gives the candidate common eigenvector.

    Returns:
      best_overlap: |<v_0|v_1>|, the alignment quality
      psi: the candidate common eigenvector (chosen as average)
      lambda_0, lambda_1: respective eigenvalues
      eigenvalue_norm_sq: |lambda_0|^2 + |lambda_1|^2 (1 for CPTP fixed point)
    """
    n = len(K_list)
    if n < 2:
        return {"best_overlap": 1.0, "psi": None, "eigenvalue_norm_sq": 1.0}
    # Eigenvectors of K_0 and K_1
    eigvals_0, eigvecs_0 = np.linalg.eig(K_list[0])
    eigvals_1, eigvecs_1 = np.linalg.eig(K_list[1])
    dim = eigvecs_0.shape[0]
    best_overlap = 0.0
    best = None
    for i in range(dim):
        for j in range(dim):
            v0 = eigvecs_0[:, i] / np.linalg.norm(eigvecs_0[:, i])
            v1 = eigvecs_1[:, j] / np.linalg.norm(eigvecs_1[:, j])
            overlap = float(abs(np.vdot(v0, v1)))
            if overlap > best_overlap:
                best_overlap = overlap
                # Average direction (Bloch-like)
                psi = (v0 * np.exp(-1j * np.angle(np.vdot(v0, v1))) + v1)
                psi = psi / np.linalg.norm(psi)
                best = {"overlap": overlap, "psi": psi,
                        "lambda_0": complex(eigvals_0[i]),
                        "lambda_1": complex(eigvals_1[j])}
    if best is None:
        return {"best_overlap": 0.0}
    lam_sq = abs(best["lambda_0"]) ** 2 + abs(best["lambda_1"]) ** 2
    return {
        "best_overlap": best_overlap,
        "psi": best["psi"],
        "lambda_0_abs": float(abs(best["lambda_0"])),
        "lambda_1_abs": float(abs(best["lambda_1"])),
        "eigenvalue_norm_sq": float(lam_sq),
    }


def fixed_point_match_score(
    psi: np.ndarray, U: np.ndarray, sigma_cr: np.ndarray, dim_cr: int
) -> float:
    """How close is |psi><psi| to the actual fixed point of E?

    Compute the actual fixed point and measure fidelity F = <psi|rho|psi>.
    """
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    rho_init = np.outer(psi, psi.conj())
    r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                          rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
    rho_fp = r["rho_ctc"]
    fidelity = float(np.real(np.vdot(psi, rho_fp @ psi)))
    return fidelity


def main():
    print("=" * 70)
    print("D-CTC Phase G: explicit joint-Kraus-eigenvector check")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    n_samples = 1500
    print(f"Generating {n_samples} samples at dim_CR={dim_cr}, dim_CTC={dim_ctc}")

    rng = np.random.default_rng(789)
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
        joint = find_best_joint_eigenvector(K_list)
        # Fidelity of candidate joint-eigenvector to actual fixed point
        if joint.get("psi") is not None:
            fid = float(np.real(np.vdot(joint["psi"], rho_fp @ joint["psi"])))
        else:
            fid = float("nan")
        records.append({
            "idx": int(k),
            "iter": int(r["iterations"]),
            "purity": float(purity),
            "best_overlap": float(joint["best_overlap"]),
            "lambda_0_abs": float(joint.get("lambda_0_abs", 0.0)),
            "lambda_1_abs": float(joint.get("lambda_1_abs", 0.0)),
            "eigenvalue_norm_sq": float(joint.get("eigenvalue_norm_sq", 0.0)),
            "fidelity_to_fp": fid,
        })
    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.1f}s\n")

    purities = np.array([r["purity"] for r in records])
    overlaps = np.array([r["best_overlap"] for r in records])
    eig_norm_sqs = np.array([r["eigenvalue_norm_sq"] for r in records])
    fidelities = np.array([r["fidelity_to_fp"] for r in records])

    print("Distribution stats:")
    print(f"  best_overlap (|<v_0|v_1>|): mean={overlaps.mean():.4f}, "
          f"min={overlaps.min():.4f}, max={overlaps.max():.4f}")
    print(f"  |lambda_0|^2 + |lambda_1|^2: mean={eig_norm_sqs.mean():.4f}, "
          f"min={eig_norm_sqs.min():.4f}, max={eig_norm_sqs.max():.4f}")
    print(f"  fidelity (joint psi vs fp): mean={fidelities.mean():.4f}, "
          f"min={fidelities.min():.4f}, max={fidelities.max():.4f}")
    print()

    # Correlations
    pearson_overlap = float(np.corrcoef(purities, overlaps)[0, 1])
    pearson_eigsq   = float(np.corrcoef(purities, eig_norm_sqs)[0, 1])
    pearson_fid     = float(np.corrcoef(purities, fidelities)[0, 1])

    print("Pearson correlations with purity:")
    print(f"  best_overlap:               {pearson_overlap:+.4f}")
    print(f"  |lambda_0|^2 + |lambda_1|^2: {pearson_eigsq:+.4f}")
    print(f"  fidelity (joint psi vs fp): {pearson_fid:+.4f}")
    print()

    # Top-10 by best_overlap
    top_overlap_idx = np.argsort(-overlaps)[:10]
    print("Top-10 by best_overlap |<v_0|v_1>|:")
    print(f"  {'idx':5s} {'overlap':8s} {'purity':7s} {'|l_0|':6s} {'|l_1|':6s} {'l_norm_sq':10s} {'fid':6s}")
    for i in top_overlap_idx:
        r = records[i]
        print(f"  {r['idx']:5d} {r['best_overlap']:8.4f} {r['purity']:7.4f} "
              f"{r['lambda_0_abs']:6.3f} {r['lambda_1_abs']:6.3f} "
              f"{r['eigenvalue_norm_sq']:10.4f} {r['fidelity_to_fp']:6.3f}")
    print()

    # Top-10 by purity
    top_purity_idx = np.argsort(-purities)[:10]
    print("Top-10 by purity (and their joint-eigenvector data):")
    print(f"  {'idx':5s} {'purity':7s} {'overlap':8s} {'|l_0|':6s} {'|l_1|':6s} {'l_norm_sq':10s} {'fid':6s}")
    for i in top_purity_idx:
        r = records[i]
        print(f"  {r['idx']:5d} {r['purity']:7.4f} {r['best_overlap']:8.4f} "
              f"{r['lambda_0_abs']:6.3f} {r['lambda_1_abs']:6.3f} "
              f"{r['eigenvalue_norm_sq']:10.4f} {r['fidelity_to_fp']:6.3f}")
    print()

    # Conditional purity given near-CPTP-compatible eigenvalue sum
    print("Conditional purity given |lambda_0|^2 + |lambda_1|^2 ~ 1:")
    for tol in (0.01, 0.05, 0.1, 0.2):
        mask = np.abs(eig_norm_sqs - 1.0) < tol
        n = mask.sum()
        if n > 0:
            mean_pur = float(purities[mask].mean())
            max_pur  = float(purities[mask].max())
            frac_pur = float(np.mean(purities[mask] > 0.9))
            print(f"  ||l_0|^2+|l_1|^2 - 1| < {tol}: n={n}, mean purity={mean_pur:.4f}, "
                  f"max={max_pur:.4f}, P(>0.9)={frac_pur:.3f}")
    print()

    # Conditional purity given both criteria (high overlap AND near-unit eigenvalue sum)
    print("Conditional purity given BOTH high overlap AND near-unit eigenvalue:")
    for over_t, eig_t in [(0.99, 0.05), (0.98, 0.1), (0.95, 0.1)]:
        mask = (overlaps > over_t) & (np.abs(eig_norm_sqs - 1.0) < eig_t)
        n = mask.sum()
        if n > 0:
            mean_pur = float(purities[mask].mean())
            frac_pur = float(np.mean(purities[mask] > 0.9))
            print(f"  overlap > {over_t}, |eig_sq - 1| < {eig_t}: n={n}, mean purity={mean_pur:.4f}, "
                  f"P(>0.9)={frac_pur:.3f}")
    print()

    # Purity tail distribution
    print("Purity tail (P(purity > p)):")
    for p in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
        f = float(np.mean(purities > p))
        print(f"  p > {p}: {f:.4f}  (n = {int(np.sum(purities > p))})")
    print()

    # Fit tail to power-law: log P(purity > p) vs -log(1-p)?
    # Or just histogram
    print("Purity histogram (0.33 to 1.0):")
    bins = np.linspace(0.33, 1.0, 14)
    hist, edges = np.histogram(purities, bins=bins)
    for i, count in enumerate(hist):
        bar = "#" * (int(count * 50 / hist.max()) if hist.max() > 0 else 0)
        print(f"  [{edges[i]:.3f}, {edges[i+1]:.3f}): {count:4d}  {bar}")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    if pearson_overlap > 0.5:
        print(f"STRONG correlation (r = {pearson_overlap:+.3f}) between purity and")
        print(f"|<v_0(K_0) | v_1(K_1)>|. Hypothesis: high-purity D-CTC fixed points")
        print(f"correspond to high-overlap eigenvectors of the Kraus operators.")
    elif pearson_overlap > 0.3:
        print(f"Moderate correlation (r = {pearson_overlap:+.3f}). Some structure")
        print(f"recovered, but not a complete predictor.")
    else:
        print(f"Weak correlation (r = {pearson_overlap:+.3f}). The overlap of best")
        print(f"individual eigenvectors does not directly predict purity.")
    print()

    # Per quantity: bin by purity tail. Real novelty = a Kraus-eigenvector
    # diagnostic that flips regime exactly at the high-purity tail.
    p_low = np.where(purities < np.quantile(purities, 0.33))[0]
    p_mid = np.where((purities >= np.quantile(purities, 0.33)) &
                      (purities < np.quantile(purities, 0.67)))[0]
    p_hi  = np.where(purities >= np.quantile(purities, 0.67))[0]
    novelty = catch_novelty_per_quantity({
        "overlap":       {"p_low": overlaps[p_low], "p_mid": overlaps[p_mid],
                          "p_hi":  overlaps[p_hi]},
        "eig_norm_sq":   {"p_low": eig_norm_sqs[p_low],
                          "p_mid": eig_norm_sqs[p_mid],
                          "p_hi":  eig_norm_sqs[p_hi]},
        "fidelity":      {"p_low": fidelities[p_low],
                          "p_mid": fidelities[p_mid],
                          "p_hi":  fidelities[p_hi]},
    })
    print(f"Novelty catcher aggregate='{novelty['aggregate_verdict']}', "
          f"novel quantities={novelty['novel_quantities']}")

    out_path = Path("examples") / "dctc_deep_phase_g_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"dim_cr": dim_cr, "dim_ctc": dim_ctc, "n_samples": n_samples},
            "correlations": {
                "pearson_purity_vs_best_overlap": pearson_overlap,
                "pearson_purity_vs_eigenvalue_norm_sq": pearson_eigsq,
                "pearson_purity_vs_fidelity": pearson_fid,
            },
            "novelty_catcher": novelty,
            "stats": {
                "best_overlap": {"mean": float(overlaps.mean()),
                                   "min": float(overlaps.min()),
                                   "max": float(overlaps.max())},
                "eigenvalue_norm_sq": {"mean": float(eig_norm_sqs.mean()),
                                          "min": float(eig_norm_sqs.min()),
                                          "max": float(eig_norm_sqs.max())},
            },
            "purity_tail": {f"P(purity > {p})": float(np.mean(purities > p))
                            for p in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)},
            "top10_overlap": [records[int(i)] for i in top_overlap_idx],
            "top10_purity": [records[int(i)] for i in top_purity_idx],
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
