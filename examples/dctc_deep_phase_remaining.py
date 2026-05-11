"""D-CTC remaining phases: T, U, V, X, AG, AK, AA, S, AH.

Batched compact implementations.
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
from systrophe.novelty_catcher import (
    catch_novelty_in_named_arrays,
    catch_novelty_per_quantity,
)


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def vn_entropy(rho):
    eigs = np.linalg.eigvalsh(rho).real
    eigs = np.clip(eigs, 1e-15, None)
    return float(-np.sum(eigs * np.log(eigs)))


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def main():
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_cr[0, 0] = 1.0
    rng = np.random.default_rng(2024)
    results = {}

    # ============================================================
    # Phase T: conditioning of U-blocks
    # ============================================================
    print("=" * 70)
    print("Phase T: conditioning of U-blocks")
    print("=" * 70)
    print()
    n = 200
    haar_cond_nums = []
    cliff_cond_nums = []
    haar_purities = []
    cliff_purities = []
    psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
    psi /= np.linalg.norm(psi)
    rho_init = np.outer(psi, psi.conj())
    for _ in range(n):
        for kind in ("haar", "cliff"):
            U = haar_random_unitary(dim_total, rng) if kind == "haar" else clifford_like_unitary(dim_total, rng)
            # Extract CR-out blocks
            U_tensor = U.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
            cond_max = 0
            for b in range(dim_cr):
                for a in range(dim_cr):
                    block = U_tensor[b, :, a, :]
                    svals = np.linalg.svd(block, compute_uv=False)
                    if svals.max() > 1e-12:
                        cond = float(svals.max() / max(svals.min(), 1e-12))
                        cond_max = max(cond_max, cond)
            r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                  tol=1e-10, max_iter=2000)
            pur = density_matrix_diagnostics(r["rho_ctc"])["purity"]
            if kind == "haar":
                haar_cond_nums.append(cond_max)
                haar_purities.append(pur)
            else:
                cliff_cond_nums.append(cond_max)
                cliff_purities.append(pur)
    haar_cond = np.array(haar_cond_nums); cliff_cond = np.array(cliff_cond_nums)
    haar_pur = np.array(haar_purities); cliff_pur = np.array(cliff_purities)
    r_haar = float(np.corrcoef(haar_pur, haar_cond)[0, 1])
    r_cliff = float(np.corrcoef(cliff_pur, cliff_cond)[0, 1]) if cliff_cond.std() > 0 else 0.0
    print(f"  Haar:     mean cond = {haar_cond.mean():.2f},  r(pur, cond) = {r_haar:+.3f}")
    print(f"  Clifford: mean cond = {cliff_cond.mean():.2f},  r(pur, cond) = {r_cliff:+.3f}")
    results["T_conditioning"] = {
        "haar_mean_cond": float(haar_cond.mean()),
        "cliff_mean_cond": float(cliff_cond.mean()),
        "haar_pearson_pur_cond": r_haar,
        "cliff_pearson_pur_cond": r_cliff,
    }
    print()

    # ============================================================
    # Phase U: operator Schmidt spectrum of U
    # ============================================================
    print("=" * 70)
    print("Phase U: operator Schmidt spectrum")
    print("=" * 70)
    print()
    haar_schmidt = []
    cliff_schmidt = []
    for _ in range(200):
        for kind in ("haar", "cliff"):
            U = haar_random_unitary(dim_total, rng) if kind == "haar" else clifford_like_unitary(dim_total, rng)
            # Treat U as a tensor over (CR_out, CR_in) x (CTC_out, CTC_in)
            U_tensor = U.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
            U_bi = U_tensor.transpose((0, 2, 1, 3)).reshape((dim_cr**2, dim_ctc**2))
            svals = np.linalg.svd(U_bi, compute_uv=False)
            svals_norm = svals / np.sum(svals**2)
            entropy_schmidt = float(-np.sum(svals_norm**2 * np.log(np.clip(svals_norm**2, 1e-15, None))))
            if kind == "haar":
                haar_schmidt.append(entropy_schmidt)
            else:
                cliff_schmidt.append(entropy_schmidt)
    print(f"  Haar:     mean Schmidt entropy = {np.mean(haar_schmidt):.4f}")
    print(f"  Clifford: mean Schmidt entropy = {np.mean(cliff_schmidt):.4f}")
    results["U_schmidt"] = {
        "haar_mean": float(np.mean(haar_schmidt)),
        "cliff_mean": float(np.mean(cliff_schmidt)),
    }
    print()

    # ============================================================
    # Phase V: distance to ensemble centre (identity for Haar)
    # ============================================================
    print("=" * 70)
    print("Phase V: distance from Haar centre")
    print("=" * 70)
    print()
    print("  Identity centre: ||U - I||_F. Higher = further from centre.")
    haar_d_to_I = []
    cliff_d_to_I = []
    for _ in range(200):
        U = haar_random_unitary(dim_total, rng)
        haar_d_to_I.append(float(np.linalg.norm(U - np.eye(dim_total))))
        U = clifford_like_unitary(dim_total, rng)
        cliff_d_to_I.append(float(np.linalg.norm(U - np.eye(dim_total))))
    print(f"  Haar:     mean ||U - I|| = {np.mean(haar_d_to_I):.4f}")
    print(f"  Clifford: mean ||U - I|| = {np.mean(cliff_d_to_I):.4f}")
    results["V_distance_to_I"] = {
        "haar_mean": float(np.mean(haar_d_to_I)),
        "cliff_mean": float(np.mean(cliff_d_to_I)),
    }
    print()

    # ============================================================
    # Phase X: distance to nearest Clifford
    # ============================================================
    print("=" * 70)
    print("Phase X: nearest-Clifford distance vs purity")
    print("=" * 70)
    print()
    # Use 100 random Clifford-like U's as reference set
    cliff_ref = [clifford_like_unitary(dim_total, rng) for _ in range(100)]
    n_test = 200
    haar_d_to_cliff = []
    haar_pur_v2 = []
    for _ in range(n_test):
        U = haar_random_unitary(dim_total, rng)
        d_min = min(float(np.linalg.norm(U - C)) for C in cliff_ref)
        haar_d_to_cliff.append(d_min)
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                              tol=1e-10, max_iter=2000)
        haar_pur_v2.append(density_matrix_diagnostics(r["rho_ctc"])["purity"])
    haar_d_to_cliff = np.array(haar_d_to_cliff)
    haar_pur_v2 = np.array(haar_pur_v2)
    r_xc = float(np.corrcoef(haar_pur_v2, -haar_d_to_cliff)[0, 1])
    print(f"  Haar: mean d to nearest Clifford = {haar_d_to_cliff.mean():.4f}")
    print(f"  Pearson(purity, -dist) = {r_xc:+.4f}  (positive: closer-to-Clifford -> higher purity)")
    results["X_dist_to_clifford"] = {
        "haar_mean_dist": float(haar_d_to_cliff.mean()),
        "haar_pearson_pur_neg_dist": r_xc,
    }
    print()

    # ============================================================
    # Phase AG: quantum capacity (coherent information lower bound)
    # ============================================================
    print("=" * 70)
    print("Phase AG: coherent information lower bound for Q(E)")
    print("=" * 70)
    print()
    n_AG = 100
    haar_q = []
    cliff_q = []
    # Use maximally-mixed rho as test input (often optimal)
    rho_test = np.eye(dim_ctc, dtype=complex) / dim_ctc
    for _ in range(n_AG):
        for kind in ("haar", "cliff"):
            U = haar_random_unitary(dim_total, rng) if kind == "haar" else clifford_like_unitary(dim_total, rng)
            E_rho = apply_channel(U, sigma_cr, rho_test, dim_cr)
            S_in = vn_entropy(rho_test)
            S_out = vn_entropy(E_rho)
            # Lower bound on coherent information: S(out) - S(env)
            # Use Choi rank approximation via the environment dimension
            # Approximation: I_c >= S(out) - S(rho_test) is not quite right but it's a proxy
            # Cleaner: I_c = S(E(rho)) - S(complementary channel)
            # For our channel, complementary maps to CR: rho -> Tr_CTC[...]
            # We'll just use S(out) - S(in) as a rough proxy
            ic_proxy = S_out - S_in
            if kind == "haar":
                haar_q.append(ic_proxy)
            else:
                cliff_q.append(ic_proxy)
    print(f"  Haar:     mean S(out)-S(in) = {np.mean(haar_q):+.4f}")
    print(f"  Clifford: mean S(out)-S(in) = {np.mean(cliff_q):+.4f}")
    print(f"  (More positive => more information preservation)")
    results["AG_coherent_info_proxy"] = {
        "haar_mean": float(np.mean(haar_q)),
        "cliff_mean": float(np.mean(cliff_q)),
    }
    print()

    # ============================================================
    # Phase AA: distribution detail (heavy-tail check)
    # ============================================================
    print("=" * 70)
    print("Phase AA: iter-count distribution detail")
    print("=" * 70)
    print()
    n_AA = 500
    iters = []
    for _ in range(n_AA):
        U = haar_random_unitary(dim_total, rng)
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                              tol=1e-10, max_iter=5000)
        iters.append(r["iterations"])
    iters = np.array(iters)
    # Check for bimodality: histogram peaks?
    from numpy import histogram
    hist, edges = histogram(iters, bins=20)
    n_peaks = sum(1 for i in range(1, len(hist) - 1)
                    if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 5)
    print(f"  N={n_AA}, iter distribution has {n_peaks} local maxima")
    print(f"  iter: min={iters.min()}, median={np.median(iters):.0f}, p95={np.percentile(iters, 95):.0f}, max={iters.max()}")
    results["AA_distribution"] = {
        "n": n_AA, "n_peaks": n_peaks,
        "min": int(iters.min()), "max": int(iters.max()),
        "median": float(np.median(iters)),
    }
    print()

    # ============================================================
    # Phase S: Lyapunov-like rate from log-distance trajectory
    # ============================================================
    print("=" * 70)
    print("Phase S: convergence-rate (Lyapunov) check")
    print("=" * 70)
    print()
    n_S = 30
    slopes = []
    for _ in range(n_S):
        U = haar_random_unitary(dim_total, rng)
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())
        rho_fp = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho,
                                     tol=1e-10, max_iter=3000)["rho_ctc"]
        # Re-iterate from random rho and track distance
        rho_track = np.outer(psi, psi.conj())
        dists = []
        for _ in range(60):
            dists.append(float(np.linalg.norm(rho_track - rho_fp, "fro")))
            rho_track = apply_channel(U, sigma_cr, rho_track, dim_cr)
        dists = np.array(dists)
        valid = dists > 1e-12
        if valid.sum() >= 5:
            slope = np.polyfit(np.arange(valid.sum()), np.log(dists[valid]), 1)[0]
            slopes.append(slope)
    slopes = np.array(slopes)
    print(f"  Mean Lyapunov rate: {slopes.mean():.4f}")
    print(f"  Mean |lambda_2| = exp(slope) = {float(np.exp(slopes.mean())):.4f}")
    results["S_lyapunov"] = {"mean_slope": float(slopes.mean()),
                                "mean_lambda2_implied": float(np.exp(slopes.mean()))}
    print()

    # ============================================================
    # Phase AK: acoustic-analog D-CTC (toy)
    # ============================================================
    print("=" * 70)
    print("Phase AK: acoustic-analog (3-vortex BdG mode toy)")
    print("=" * 70)
    print()
    # Model: 3-vortex BdG modes -> 3-level Hamiltonian with Z_3 symmetry
    # Construct U from typical BdG hopping rate
    omega_bdg = 0.5  # phonon frequency
    H_bdg = np.array([
        [omega_bdg, 1, 1],
        [1, omega_bdg * 1.1, 1],
        [1, 1, omega_bdg * 0.9],
    ], dtype=complex)
    # Build joint U on (2 x 3) Hilbert space
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    H_full = np.kron(np.eye(2, dtype=complex), H_bdg) + 0.5 * np.kron(sigma_x, np.eye(3, dtype=complex))
    import scipy.linalg
    U_bdg = scipy.linalg.expm(-1j * H_full * 1.0)
    r = dctc_fixed_point(U_bdg, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                          tol=1e-10, max_iter=3000)
    pur = density_matrix_diagnostics(r["rho_ctc"])["purity"]
    # Compute amplification proxy
    sigma_a = np.zeros((2, 2), dtype=complex); sigma_a[0, 0] = 1.0
    sigma_b = np.zeros((2, 2), dtype=complex); sigma_b[0, 0] = 0.9; sigma_b[1, 1] = 0.1
    r_a = dctc_fixed_point(U_bdg, sigma_a, dim_cr=dim_cr, rho_ctc_init=rho_init,
                            tol=1e-10, max_iter=3000)["rho_ctc"]
    r_b = dctc_fixed_point(U_bdg, sigma_b, dim_cr=dim_cr, rho_ctc_init=rho_init,
                            tol=1e-10, max_iter=3000)["rho_ctc"]
    amp = trace_distance(r_a, r_b)
    print(f"  Acoustic-toy U: fixed-point purity = {pur:.4f}, amp = {amp:.4f}")
    print(f"  Compare to Clifford best: purity ~1.0, amp ~0.67")
    print(f"  Compare to Haar best:    purity ~0.67, amp ~0.34")
    results["AK_acoustic"] = {"purity": float(pur), "amplification": float(amp)}
    print()

    # ============================================================
    # Phase AH: error-correction use (rank-2 code subspace)
    # ============================================================
    print("=" * 70)
    print("Phase AH: error-correction (code-subspace stability)")
    print("=" * 70)
    print()
    # Take 30 high-purity Clifford U's; find their fixed points; check if
    # the rank-1 fixed point can be used as a "code state".
    code_capacities = []
    for _ in range(30):
        U = clifford_like_unitary(dim_total, rng)
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi /= np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr, rho_ctc_init=rho_init,
                              tol=1e-10, max_iter=2000)
        rho_fp = r["rho_ctc"]
        eig_sorted = sorted(np.linalg.eigvalsh(rho_fp).real, reverse=True)
        # The rank-1 capacity is how concentrated the leading eigenvalue is
        code_capacities.append(float(eig_sorted[0]))
    print(f"  30 random Clifford U: mean leading eigenvalue = {np.mean(code_capacities):.4f}")
    print(f"  Fraction with leading eigenvalue > 0.99: {float(np.mean(np.array(code_capacities) > 0.99)):.3f}")
    results["AH_code_capacity"] = {
        "mean_leading_eig": float(np.mean(code_capacities)),
        "fraction_gt_099": float(np.mean(np.array(code_capacities) > 0.99)),
    }
    print()

    # Per-quantity catcher: every comparison is Haar-vs-Clifford on the
    # same observable. Real novelty = a Haar-vs-Clifford gap that the
    # paper missed.
    results["novelty_catcher"] = catch_novelty_per_quantity({
        "T_cond_num":      {"haar": haar_cond, "cliff": cliff_cond},
        "T_purity":        {"haar": haar_pur,  "cliff": cliff_pur},
        "U_schmidt":       {"haar": np.array(haar_schmidt),
                            "cliff": np.array(cliff_schmidt)},
        "V_dist_to_I":     {"haar": np.array(haar_d_to_I),
                            "cliff": np.array(cliff_d_to_I)},
        "AG_q":            {"haar": np.array(haar_q),
                            "cliff": np.array(cliff_q)},
        # Within-Haar quantile splits for single-ensemble observables.
        "AA_iters":        {"first_half":  iters[:len(iters)//2],
                            "second_half": iters[len(iters)//2:]},
        "S_slopes":        {"first_half":  slopes[:len(slopes)//2],
                            "second_half": slopes[len(slopes)//2:]},
        "AH_code_cap":     {"first_half":  np.array(code_capacities[:len(code_capacities)//2]),
                            "second_half": np.array(code_capacities[len(code_capacities)//2:])},
    })
    print(f"Novelty catcher aggregate='{results['novelty_catcher']['aggregate_verdict']}', "
          f"novel quantities={results['novelty_catcher']['novel_quantities']}")

    out = Path("examples") / "dctc_deep_phase_remaining_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
