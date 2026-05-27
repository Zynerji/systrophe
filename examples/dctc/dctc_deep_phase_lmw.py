"""D-CTC Phase L + M + W batch.

Phase L: extreme-tail statistics at dim_CR=2, dim_CTC=2 with 5000 Haar samples.
Phase M: sigma_CR mixedness sweep.
Phase W: distance to nearest separable U (Kronecker-separable).
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


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def distance_to_separable(U: np.ndarray, dim_cr: int, dim_ctc: int) -> float:
    """Min Frobenius distance from U to a Kronecker-separable U_CR (x) U_CTC.

    Approximated by SVD of U treated as a (dim_cr*dim_ctc, dim_cr*dim_ctc)
    operator viewed as a (dim_cr^2 x dim_ctc^2) tensor.
    """
    U_tensor = U.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
    U_bi = U_tensor.transpose((0, 2, 1, 3)).reshape(
        (dim_cr * dim_cr, dim_ctc * dim_ctc))
    svals = np.linalg.svd(U_bi, compute_uv=False)
    # Frobenius norm of U is sum of squared svals = dim (for unitary)
    # Distance to best rank-1 approximation = sqrt(sum of svals[1:]^2)
    return float(np.sqrt(np.sum(svals[1:] ** 2)))


def main():
    print("=" * 70)
    print("D-CTC Phase L + M + W batch")
    print("=" * 70)
    print()

    rng = np.random.default_rng(101)
    all_results = {}

    # ============================================================
    # Phase L: extreme tail at dim_CR=2, dim_CTC=2
    # ============================================================
    print("Phase L: extreme-tail statistics at dim_CR=2, dim_CTC=2")
    print()
    n_L = 5000
    dim_cr, dim_ctc = 2, 2
    dim_total = dim_cr * dim_ctc
    purities = []
    iters = []
    t0 = time.time()
    for k in range(n_L):
        U = haar_random_unitary(dim_total, rng)
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                              rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
        purities.append(density_matrix_diagnostics(r["rho_ctc"])["purity"])
        iters.append(r["iterations"])
    purities = np.array(purities)
    iters = np.array(iters)
    print(f"  n={n_L}, time={time.time()-t0:.1f}s")
    print(f"  max purity:    {purities.max():.6f}")
    print(f"  P(>0.9):       {float(np.mean(purities > 0.9)):.4f}")
    print(f"  P(>0.95):      {float(np.mean(purities > 0.95)):.4f}")
    print(f"  P(>0.99):      {float(np.mean(purities > 0.99)):.4f}")
    print(f"  P(>0.999):     {float(np.mean(purities > 0.999)):.4f}")
    print()

    # Power-law tail fit: 1 - P(p) vs p for the upper tail
    # P(purity > p) ~ (1 - p)^alpha?
    p_vals = np.array([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98])
    cdf_above = np.array([float(np.mean(purities > p)) for p in p_vals])
    valid = cdf_above > 0
    if valid.sum() >= 3:
        slope, intercept = np.polyfit(np.log(1 - p_vals[valid]),
                                         np.log(cdf_above[valid]), 1)
        print(f"  Power-law fit: P(>p) ~ (1-p)^{slope:.3f}")
    print()

    all_results["phase_L"] = {
        "dim_cr": dim_cr, "dim_ctc": dim_ctc, "n": n_L,
        "max_purity": float(purities.max()),
        "P_gt_09": float(np.mean(purities > 0.9)),
        "P_gt_095": float(np.mean(purities > 0.95)),
        "P_gt_099": float(np.mean(purities > 0.99)),
        "tail_powerlaw_alpha": float(slope) if valid.sum() >= 3 else None,
    }

    # ============================================================
    # Phase M: sigma_CR mixedness sweep
    # ============================================================
    print("Phase M: sigma_CR mixedness sweep (dim_CR=2, dim_CTC=3)")
    print()
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    epsilons = np.linspace(0.0, 0.5, 11)
    n_per = 500
    print(f"  {'eps':5s} {'mean':6s} {'median':7s} {'max':6s} {'P(>0.7)':9s} {'P(>0.9)':9s}")
    m_results = []
    for eps in epsilons:
        rng_m = np.random.default_rng(2222 + int(eps * 1000))
        sigma_cr = np.array([[1 - eps, 0], [0, eps]], dtype=complex)
        prs = []
        for _ in range(n_per):
            U = haar_random_unitary(dim_total, rng_m)
            psi = rng_m.standard_normal(dim_ctc) + 1j * rng_m.standard_normal(dim_ctc)
            psi = psi / np.linalg.norm(psi)
            rho_init = np.outer(psi, psi.conj())
            r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                  rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
            prs.append(density_matrix_diagnostics(r["rho_ctc"])["purity"])
        prs = np.array(prs)
        m_results.append({
            "eps": float(eps),
            "mean": float(prs.mean()),
            "median": float(np.median(prs)),
            "max": float(prs.max()),
            "P_gt_07": float(np.mean(prs > 0.7)),
            "P_gt_09": float(np.mean(prs > 0.9)),
        })
        print(f"  {eps:.2f}  {prs.mean():.3f}  {np.median(prs):.3f}  {prs.max():.3f}  "
              f"{float(np.mean(prs > 0.7)):.4f}    {float(np.mean(prs > 0.9)):.4f}")
    print()
    all_results["phase_M"] = m_results

    # ============================================================
    # Phase W: distance to separable U vs purity
    # ============================================================
    print("Phase W: distance to nearest separable U")
    print()
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    n_W = 1500
    rng_w = np.random.default_rng(3333)
    w_records = []
    for _ in range(n_W):
        U = haar_random_unitary(dim_total, rng_w)
        sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex)
        sigma_cr[0, 0] = 1.0
        psi = rng_w.standard_normal(dim_ctc) + 1j * rng_w.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                              rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
        purity = density_matrix_diagnostics(r["rho_ctc"])["purity"]
        sep_dist = distance_to_separable(U, dim_cr, dim_ctc)
        w_records.append({"purity": float(purity),
                            "sep_dist": float(sep_dist)})

    purities_w = np.array([r["purity"] for r in w_records])
    sep_dists = np.array([r["sep_dist"] for r in w_records])
    pearson = float(np.corrcoef(purities_w, sep_dists)[0, 1])
    pearson_neg = float(np.corrcoef(purities_w, -sep_dists)[0, 1])
    print(f"  Pearson(purity, sep_dist):     {pearson:+.4f}")
    print(f"  Pearson(purity, -sep_dist):    {pearson_neg:+.4f}")
    print(f"  sep_dist stats: mean={sep_dists.mean():.4f}, min={sep_dists.min():.4f}, "
          f"max={sep_dists.max():.4f}")
    print()

    # Top-10 by smallest sep_dist
    print("Top-10 by smallest distance-to-separable:")
    sorted_idx = np.argsort(sep_dists)[:10]
    print(f"  {'sep_dist':10s} {'purity':7s}")
    for i in sorted_idx:
        print(f"  {sep_dists[i]:10.4f} {purities_w[i]:7.4f}")
    print()

    all_results["phase_W"] = {
        "n": n_W,
        "pearson_purity_vs_sep_dist": pearson,
        "sep_dist_stats": {"mean": float(sep_dists.mean()),
                            "min": float(sep_dists.min()),
                            "max": float(sep_dists.max())},
    }

    print("=" * 70)
    print("Summary of L + M + W")
    print("=" * 70)
    print()
    print(f"L: max purity at d=2x2 = {all_results['phase_L']['max_purity']:.4f}")
    print(f"   (n=5000; purity > 0.99: {all_results['phase_L']['P_gt_099']*100:.2f}%)")
    print(f"M: P(>0.9) peaks at eps={[r['P_gt_09'] for r in m_results].index(max(r['P_gt_09'] for r in m_results)) * 0.05:.2f}")
    print(f"W: r(purity, sep_dist) = {pearson:+.4f}")
    print()

    # Per-quantity catcher: compare same observable across sub-experiments.
    # L is dim_CR=2, dim_CTC=2 (5000 Haar); W is dim_CR=2, dim_CTC=3 (1500).
    # Comparing L_purities vs W_purities = same-quantity-different-condition.
    # M is a sigma_CR-mixedness sweep; compare the per-eps mean-purity
    # distribution against itself split into low/high eps regimes.
    m_means = np.array([r["mean"] for r in m_results])
    m_p_gt_09 = np.array([r["P_gt_09"] for r in m_results])
    half_m = len(m_results) // 2
    half_W = len(sep_dists) // 2
    all_results["novelty_catcher"] = catch_novelty_per_quantity({
        "purity":      {"L_dim2x2": purities, "W_dim2x3": purities_w},
        "M_mean_purity_vs_eps": {"low_eps": m_means[:half_m],
                                  "high_eps": m_means[half_m:]},
        "M_P_gt_09_vs_eps":     {"low_eps": m_p_gt_09[:half_m],
                                  "high_eps": m_p_gt_09[half_m:]},
        "sep_dist":    {"first_half":  sep_dists[:half_W],
                        "second_half": sep_dists[half_W:]},
    })
    print(f"Novelty catcher aggregate='{all_results['novelty_catcher']['aggregate_verdict']}', "
          f"novel quantities={all_results['novelty_catcher']['novel_quantities']}")

    out = Path("examples") / "dctc_deep_phase_lmw_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
