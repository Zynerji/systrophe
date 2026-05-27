"""D-CTC Phase O+Q+R: spectral statistics + ergodicity.

O: full eigenvalue distribution of channel superoperator E
Q: spectral form factor K(tau) for level statistics
R: ergodicity -- do different rho_init converge to same fixed point?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import (
    channel_superoperator,
    clifford_like_unitary,
    dctc_fixed_point,
    density_matrix_diagnostics,
)
from systrophe.catchers.novelty_catcher import catch_novelty_in_named_arrays


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def main():
    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    sigma_cr = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_cr[0, 0] = 1.0
    rng = np.random.default_rng(2024)

    print("=" * 70)
    print("Phase O: full eigenvalue distribution of E")
    print("=" * 70)
    print()

    # Collect all eigenvalues from many channels
    n_haar = 500
    n_cliff = 500
    all_eigs_haar = []
    all_eigs_cliff = []
    for _ in range(n_haar):
        U = haar_random_unitary(dim_total, rng)
        M = channel_superoperator(U, sigma_cr, dim_cr)
        eigs = np.abs(np.linalg.eigvals(M))
        all_eigs_haar.extend(eigs.tolist())
    for _ in range(n_cliff):
        U = clifford_like_unitary(dim_total, rng)
        M = channel_superoperator(U, sigma_cr, dim_cr)
        eigs = np.abs(np.linalg.eigvals(M))
        all_eigs_cliff.extend(eigs.tolist())

    all_eigs_haar = np.array(all_eigs_haar)
    all_eigs_cliff = np.array(all_eigs_cliff)

    print(f"Haar: {n_haar} channels, {len(all_eigs_haar)} total eigenvalues")
    print(f"  mean: {all_eigs_haar.mean():.4f}, max: {all_eigs_haar.max():.4f}, min: {all_eigs_haar.min():.4f}")
    print(f"  fraction > 0.9: {float(np.mean(all_eigs_haar > 0.9)):.4f}")
    print(f"Clifford: {n_cliff} channels, {len(all_eigs_cliff)} total eigenvalues")
    print(f"  mean: {all_eigs_cliff.mean():.4f}, max: {all_eigs_cliff.max():.4f}, min: {all_eigs_cliff.min():.4f}")
    print(f"  fraction > 0.9: {float(np.mean(all_eigs_cliff > 0.9)):.4f}")
    print()

    # Histogram
    print("Histogram of |lambda| values for Haar:")
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(all_eigs_haar, bins=bins)
    for i in range(len(hist)):
        bar = "#" * (int(hist[i] * 60 / hist.max()) if hist.max() > 0 else 0)
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:4d}  {bar}")
    print()

    print("Histogram of |lambda| values for Clifford-like:")
    hist, _ = np.histogram(all_eigs_cliff, bins=bins)
    for i in range(len(hist)):
        bar = "#" * (int(hist[i] * 60 / hist.max()) if hist.max() > 0 else 0)
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:4d}  {bar}")
    print()

    # Phase Q: nearest-neighbor spacing distribution (RMT signature)
    print("=" * 70)
    print("Phase Q: nearest-neighbour spacing of |lambda|")
    print("=" * 70)
    print()

    def nn_spacing(eigs):
        sorted_eigs = np.sort(eigs)
        return np.diff(sorted_eigs)

    haar_spacing = nn_spacing(all_eigs_haar)
    cliff_spacing = nn_spacing(all_eigs_cliff)
    haar_spacing = haar_spacing / haar_spacing.mean()  # normalize
    cliff_spacing = cliff_spacing / cliff_spacing.mean()

    # Test for Wigner-Dyson (chaotic): density ~ s exp(-pi s^2 / 4) (GOE)
    # vs Poisson (integrable): density ~ exp(-s)
    haar_mean_log_s = float(np.mean(np.log(np.maximum(haar_spacing, 1e-12))))
    cliff_mean_log_s = float(np.mean(np.log(np.maximum(cliff_spacing, 1e-12))))
    # Poisson: <log s> = -gamma_Euler approximately -0.577
    # GOE: <log s> approximately different
    print(f"  Haar      <log s> = {haar_mean_log_s:+.3f}  (Poisson predicts -0.577)")
    print(f"  Clifford  <log s> = {cliff_mean_log_s:+.3f}")
    print()

    # Phase R: ergodicity (already partially tested in N; here at variety of initial states)
    print("=" * 70)
    print("Phase R: ergodicity at variety of initial conditions")
    print("=" * 70)
    print()

    # Sample 50 Haar channels, each with 20 random rho_init, measure spread of fixed points
    n_U = 50
    n_init = 20
    max_spreads = []
    for _ in range(n_U):
        U = haar_random_unitary(dim_total, rng)
        fps = []
        for _ in range(n_init):
            psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
            psi /= np.linalg.norm(psi)
            r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                                  rho_ctc_init=np.outer(psi, psi.conj()),
                                  tol=1e-10, max_iter=3000)
            fps.append(r["rho_ctc"])
        max_d = 0
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                d = float(np.linalg.norm(fps[i] - fps[j], "fro"))
                max_d = max(max_d, d)
        max_spreads.append(max_d)
    max_spreads = np.array(max_spreads)
    print(f"  Across {n_U} channels x {n_init} rho_init:")
    print(f"  Max pairwise distance: mean={max_spreads.mean():.6f}, max={max_spreads.max():.6f}")
    n_ergodic = int(np.sum(max_spreads < 1e-6))
    print(f"  Ergodic (max dist < 1e-6): {n_ergodic}/{n_U} = {n_ergodic*100/n_U:.0f}%")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    print(f"O: Haar channels have |lambda| concentrated around mean {all_eigs_haar.mean():.3f}")
    print(f"   Clifford channels have |lambda| at extremes more often")
    print(f"   (fraction > 0.9 for Clifford: {float(np.mean(all_eigs_cliff > 0.9)):.3f},")
    print(f"    for Haar: {float(np.mean(all_eigs_haar > 0.9)):.3f})")
    print()
    print(f"Q: nearest-neighbour spacing <log s> = {haar_mean_log_s:.3f} for Haar")
    print(f"   (Poisson predicts -0.577; significant deviation suggests level repulsion)")
    print()
    print(f"R: {n_ergodic*100/n_U:.0f}% of channels reach the same fixed point regardless of rho_init")

    novelty = catch_novelty_in_named_arrays({
        "haar_eigs":     all_eigs_haar,
        "cliff_eigs":    all_eigs_cliff,
        "haar_spacing":  haar_spacing,
        "cliff_spacing": cliff_spacing,
        "ergodic_spreads": max_spreads,
    })
    print(f"Novelty catcher: verdict='{novelty['verdict']}', "
          f"n_sharp={len(novelty['sharp_features'])}")

    out = Path("examples") / "dctc_deep_phase_oqr_results.json"
    with open(out, "w") as f:
        json.dump({
            "haar_eigs": {"mean": float(all_eigs_haar.mean()),
                            "frac_gt_09": float(np.mean(all_eigs_haar > 0.9)),
                            "n_samples": len(all_eigs_haar)},
            "clifford_eigs": {"mean": float(all_eigs_cliff.mean()),
                                "frac_gt_09": float(np.mean(all_eigs_cliff > 0.9)),
                                "n_samples": len(all_eigs_cliff)},
            "spacing_log_means": {"haar": haar_mean_log_s,
                                    "clifford": cliff_mean_log_s,
                                    "poisson_ref": -0.577},
            "ergodicity_fraction": float(n_ergodic / n_U),
            "novelty_catcher": novelty,
        }, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
