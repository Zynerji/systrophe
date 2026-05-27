"""D-CTC deep exploration --- Phase E: Kraus commutator hypothesis.

A CPTP channel E(rho) = Sum_a K_a rho K_a^dag has a *pure* fixed point
|psi><psi| if and only if |psi> is a *simultaneous* eigenvector of
every Kraus operator K_a. (Proof: E(|psi><psi|) = Sum_a K_a |psi><psi| K_a^dag;
this is rank-1 iff each K_a |psi> is proportional to |psi>.)

For dim_CR = 2 we have two Kraus operators K_0, K_1. They share an
eigenvector iff their commutator [K_0, K_1] is rank-deficient (i.e.,
has a non-trivial null vector). For Haar-random K_a, generic
commutators are full-rank; near-rank-deficient ones are rare and
correspond to near-pure fixed points.

Testable hypothesis: high-purity fixed points correlate with small
sigma_min([K_0, K_1]).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import dctc_fixed_point, density_matrix_diagnostics
from systrophe.catchers.novelty_catcher import (
    catch_novelty_in_named_arrays,
    catch_novelty_per_quantity,
)


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def kraus_operators(U: np.ndarray, sigma_cr: np.ndarray, dim_cr: int) -> list[np.ndarray]:
    """Kraus decomposition of the D-CTC channel.

    For E(rho) = Tr_CR[U(sigma_CR (x) rho)U^dag], the Kraus operators are

        K_{ab} = sqrt(p_a) <b|_CR U |a>_CR

    where sigma_CR = Sum p_a |a><a| (spectral decomposition).
    """
    p, vec = np.linalg.eigh(sigma_cr)
    dim_ctc = U.shape[0] // dim_cr
    kraus = []
    for a in range(dim_cr):
        if p[a] < 1e-12:
            continue
        # K_{ab} = sqrt(p_a) <b|_CR U |a>_CR
        # arrange as a (dim_cr, dim_ctc, dim_cr, dim_ctc) -> pick a in input, all b in output
        ket_a = vec[:, a]  # |a>_CR
        for b in range(dim_cr):
            ket_b = np.zeros(dim_cr, dtype=complex)
            ket_b[b] = 1.0
            # Build the operator on CTC
            K = np.sqrt(p[a]) * np.einsum("BC,B,c->Cc",
                U.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
                 .transpose((0, 2, 1, 3)).reshape((dim_cr * dim_cr, dim_ctc, dim_ctc))[b * dim_cr + a],
                ket_b, ket_a) if False else None
            # Simpler: K_{ab} acts on |psi>_CTC by applying U to |a>_CR (x) |psi>,
            # then projecting on <b|_CR.
            # K_{ab}[i, j] = sum_{X, Y} <b, i| U |a, j> ket_a[Y] ket_b[X] ... we use the
            # spectral form: assume sigma_cr is diagonal in computational basis (we'll
            # input it that way), so p[a] is the eigenvalue, ket_a is basis vector |a>.
            pass
    # Simple implementation: assume sigma_cr is diagonal (in standard basis), so we
    # have one Kraus operator per nonzero diagonal entry per output CR-basis state.
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


def commutator_min_singular(K_list: list[np.ndarray]) -> dict:
    """Smallest singular value of pairwise commutators [K_i, K_j].

    Returns:
      sigma_min_overall: smallest sigma_min across all pairs
      sigma_min_per_pair: list of (i, j, sigma_min) tuples
    """
    n = len(K_list)
    pair_sigmas = []
    for i in range(n):
        for j in range(i + 1, n):
            C = K_list[i] @ K_list[j] - K_list[j] @ K_list[i]
            svals = np.linalg.svd(C, compute_uv=False)
            pair_sigmas.append((i, j, float(svals[-1])))
    if not pair_sigmas:
        return {"sigma_min_overall": float("inf"), "pairs": []}
    sigma_min = min(s for _, _, s in pair_sigmas)
    return {"sigma_min_overall": sigma_min, "pairs": pair_sigmas}


def common_eigenvector_check(K_list: list[np.ndarray], eps: float = 1e-6) -> dict:
    """Test if Kraus operators share a common eigenvector.

    Two operators A, B share an eigenvector iff [A, B] has a null vector
    in the kernel of [A, B]. For multiple operators, we need a vector
    in the common kernel of all pairwise commutators.

    Returns the smallest min-eigenvalue across pairwise commutators
    and the candidate common eigenvector when one exists.
    """
    n = len(K_list)
    if n < 2:
        return {"shared": True, "score": 0.0}
    # Build the column-stack of all pairwise commutators
    blocks = []
    for i in range(n):
        for j in range(i + 1, n):
            blocks.append(K_list[i] @ K_list[j] - K_list[j] @ K_list[i])
    # Stack vertically
    M = np.vstack(blocks)
    svals = np.linalg.svd(M, compute_uv=False)
    return {
        "shared": float(svals[-1]) < eps,
        "score_smallest_singular_value": float(svals[-1]),
        "n_singular_values": int(len(svals)),
    }


def main():
    print("=" * 70)
    print("D-CTC Phase E: Kraus commutator hypothesis")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    n_samples = 2000
    print(f"Generating {n_samples} samples at dim_CR={dim_cr}, dim_CTC={dim_ctc}")

    rng = np.random.default_rng(123)
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
        if not K_list:
            continue
        cmm = commutator_min_singular(K_list)
        shared = common_eigenvector_check(K_list)
        records.append({
            "idx": int(k),
            "iter": int(r["iterations"]),
            "purity": float(purity),
            "sigma_min_commutator": float(cmm["sigma_min_overall"]),
            "shared_singular_value": float(shared["score_smallest_singular_value"]),
            "n_kraus": int(len(K_list)),
        })

    elapsed = time.time() - t0
    print(f"Sampling done in {elapsed:.1f}s\n")

    purities = np.array([r["purity"] for r in records])
    sigma_mins = np.array([r["sigma_min_commutator"] for r in records])
    shared_svs = np.array([r["shared_singular_value"] for r in records])

    print(f"sigma_min([K_0, K_1]) distribution:")
    print(f"  min:    {sigma_mins.min():.4f}")
    print(f"  median: {np.median(sigma_mins):.4f}")
    print(f"  mean:   {sigma_mins.mean():.4f}")
    print(f"  max:    {sigma_mins.max():.4f}")
    print()

    # Correlation: purity vs -log(sigma_min)
    # Use -log because small sigma_min -> high purity, log-scale for spread
    valid = sigma_mins > 1e-12
    pearson_raw = float(np.corrcoef(purities[valid], sigma_mins[valid])[0, 1])
    pearson_log = float(np.corrcoef(purities[valid], -np.log(sigma_mins[valid]))[0, 1])
    print(f"Pearson(purity, sigma_min):       {pearson_raw:+.4f}")
    print(f"Pearson(purity, -log sigma_min):  {pearson_log:+.4f}")
    print()

    # Same for shared_singular_value (all-pairs)
    pearson_shared = float(np.corrcoef(purities[valid], -np.log(shared_svs[valid]))[0, 1])
    print(f"Pearson(purity, -log shared_singular):  {pearson_shared:+.4f}")
    print()

    # Top-20 high-purity vs top-20 small-sigma-min: do they overlap?
    purity_top20 = set(np.argsort(-purities)[:20])
    sigma_bot20  = set(np.argsort(sigma_mins)[:20])
    overlap = purity_top20 & sigma_bot20
    print(f"Top-20 high-purity vs top-20 smallest commutator: {len(overlap)} overlap")
    print(f"  indices: {sorted(overlap)}")
    print()

    # Show the top-10 high-purity with their commutator structure
    purity_top10_idx = np.argsort(-purities)[:10]
    print("Top-10 high-purity samples (commutator structure):")
    print(f"  {'idx':5s} {'purity':7s} {'sigma_min_cmm':14s} {'shared_sv':10s}")
    for i in purity_top10_idx:
        r = records[i]
        print(f"  {r['idx']:5d} {r['purity']:7.4f} {r['sigma_min_commutator']:14.6f} "
              f"{r['shared_singular_value']:10.6f}")
    print()

    # Show low-purity bottom-10 for comparison
    purity_bot10_idx = np.argsort(purities)[:10]
    print("Bottom-10 (most-mixed) samples:")
    print(f"  {'idx':5s} {'purity':7s} {'sigma_min_cmm':14s} {'shared_sv':10s}")
    for i in purity_bot10_idx:
        r = records[i]
        print(f"  {r['idx']:5d} {r['purity']:7.4f} {r['sigma_min_commutator']:14.6f} "
              f"{r['shared_singular_value']:10.6f}")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    if abs(pearson_log) > 0.5:
        if pearson_log > 0:
            print(f"STRONG positive correlation (r = {pearson_log:+.3f}) between purity")
            print("and -log(sigma_min). Small commutator -> high purity.")
            print()
            print("HYPOTHESIS VALIDATED: high-purity D-CTC fixed points emerge from")
            print("Kraus operators with near-degenerate commutators, i.e., near-")
            print("simultaneous eigenvector structure.")
        else:
            print(f"STRONG negative correlation (r = {pearson_log:+.3f}). Hypothesis flipped.")
    elif abs(pearson_log) > 0.2:
        print(f"Moderate correlation (r = {pearson_log:+.3f}). Partial validation.")
    else:
        print(f"Weak correlation (r = {pearson_log:+.3f}). Hypothesis NOT validated.")
        print()
        print("The high-purity property is NOT explained by commutator rank-deficiency")
        print("alone. Some other structural feature is at play.")
    print()

    # Per quantity: split the Haar ensemble by purity quantile and check
    # if low/mid/high purity sub-samples have structurally distinct
    # commutator-singular-value distributions. Real novelty = a sigma_min
    # regime change correlated with the purity tail.
    p_low = np.where(purities < np.quantile(purities, 0.33))[0]
    p_mid = np.where((purities >= np.quantile(purities, 0.33)) &
                      (purities < np.quantile(purities, 0.67)))[0]
    p_hi  = np.where(purities >= np.quantile(purities, 0.67))[0]
    novelty = catch_novelty_per_quantity({
        "sigma_min":   {"p_low": sigma_mins[p_low], "p_mid": sigma_mins[p_mid],
                        "p_hi":  sigma_mins[p_hi]},
        "shared_sv":   {"p_low": shared_svs[p_low], "p_mid": shared_svs[p_mid],
                        "p_hi":  shared_svs[p_hi]},
        "purity":      {"p_low": purities[p_low],   "p_mid": purities[p_mid],
                        "p_hi":  purities[p_hi]},
    })
    print(f"Novelty catcher aggregate='{novelty['aggregate_verdict']}', "
          f"novel quantities={novelty['novel_quantities']}")

    out_path = Path("examples") / "dctc_deep_phase_e_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "config": {"dim_cr": dim_cr, "dim_ctc": dim_ctc, "n_samples": n_samples},
            "correlations": {
                "pearson_purity_vs_sigma_min": pearson_raw,
                "pearson_purity_vs_neg_log_sigma_min": pearson_log,
                "pearson_purity_vs_neg_log_shared_sv": pearson_shared,
            },
            "sigma_min_stats": {
                "min": float(sigma_mins.min()),
                "median": float(np.median(sigma_mins)),
                "mean": float(sigma_mins.mean()),
                "max": float(sigma_mins.max()),
            },
            "top10_high_purity": [records[int(i)] for i in purity_top10_idx],
            "bottom10_low_purity": [records[int(i)] for i in purity_bot10_idx],
            "overlap_top20": [int(x) for x in overlap],
            "novelty_catcher": novelty,
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
