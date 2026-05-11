"""D-CTC Phase AJ: D-CTC channel on Systrophe LP background.

Build a D-CTC channel from the actual LP exterior physics: the F(r),
K(r), L(r) functions at sampled radii. Then run the D-CTC iteration
and measure amplification.

Question: does the LP-derived channel produce amplification at all?
Or does the physically-grounded structure trivialise the iteration?
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg

from systrophe.d_ctc import (
    clifford_like_unitary,
    dctc_fixed_point,
    density_matrix_diagnostics,
)
from systrophe.novelty_catcher import catch_novelty_in_named_arrays
from systrophe.vanstockum import VanStockumInterior


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def polar_unitary(M):
    U, _, Vh = np.linalg.svd(M)
    return U @ Vh


def trace_distance(rho_a, rho_b):
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def lp_background_unitary(vs: VanStockumInterior, n_radii: int = 6,
                             r_min: float = 1.1, r_max: float = 10.0,
                             dim_cr: int = 2) -> np.ndarray:
    """Construct U from LP exterior F(r), K(r), L(r) values.

    Strategy:
      - Sample (F, K, L) at n_radii radii spanning the exterior
      - Build a 2(d_ctc)x2(d_ctc) Hermitian-like matrix from these
      - Exponentiate and polar-decompose
    """
    rs = np.linspace(r_min, r_max, n_radii)
    F_vals = np.array([float(vs.analytic_exterior_F(r)) for r in rs])
    K_vals = np.array([float(vs.analytic_exterior_K(r)) for r in rs])
    L_vals = np.array([float(vs.analytic_exterior_L(r)) for r in rs])

    # Normalise (these can be large)
    scale = max(np.max(np.abs(F_vals)), np.max(np.abs(K_vals)), np.max(np.abs(L_vals)))
    F_n = F_vals / scale
    K_n = K_vals / scale
    L_n = L_vals / scale

    dim_total = dim_cr * n_radii  # match dim_ctc = n_radii
    # Build coupling Hamiltonian
    H = np.zeros((dim_total, dim_total), dtype=complex)
    # Diagonal: F at each (cr, r)
    for cr in range(dim_cr):
        for i in range(n_radii):
            H[cr * n_radii + i, cr * n_radii + i] = F_n[i] + 0.2 * cr

    # Off-diagonal in CR direction: K (twist between CR sectors at same r)
    for i in range(n_radii):
        for cr1 in range(dim_cr):
            for cr2 in range(dim_cr):
                if cr1 != cr2:
                    H[cr1 * n_radii + i, cr2 * n_radii + i] = K_n[i] * 1j
                    # Make Hermitian via conjugate
    # In-CR off-diagonal: L (radial mixing within same CR sector)
    for cr in range(dim_cr):
        for i in range(n_radii):
            for j in range(n_radii):
                if i != j:
                    H[cr * n_radii + i, cr * n_radii + j] = 0.3 * L_n[(i + j) % n_radii] * np.exp(1j * np.pi * (j - i) / n_radii)
    # Symmetrise
    H = (H + H.conj().T) / 2

    U_evol = scipy.linalg.expm(-1j * H)

    # Clifford base on dim_total
    rng = np.random.default_rng(42)
    perm = rng.permutation(dim_total)
    P = np.zeros((dim_total, dim_total), dtype=complex)
    for i, p in enumerate(perm):
        P[i, p] = 1.0
    D = np.diag(rng.choice([1, -1, 1j, -1j], dim_total))
    return polar_unitary(U_evol @ P @ D)


def dctc_amplification(U, dim_cr=2, eps=0.1, n_init=5):
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    sigma_a = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_a[0, 0] = 1.0
    sigma_b = np.zeros((dim_cr, dim_cr), dtype=complex)
    sigma_b[0, 0] = 1 - eps; sigma_b[1, 1] = eps
    rng = np.random.default_rng(11)
    best_amp = 0.0
    best_pur = 0.0
    for _ in range(n_init):
        psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
        psi = psi / np.linalg.norm(psi)
        rho_init = np.outer(psi, psi.conj())
        try:
            r_a = dctc_fixed_point(U, sigma_a, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                     tol=1e-10, max_iter=3000)["rho_ctc"]
            r_b = dctc_fixed_point(U, sigma_b, dim_cr=dim_cr, rho_ctc_init=rho_init,
                                     tol=1e-10, max_iter=3000)["rho_ctc"]
            amp = trace_distance(r_a, r_b)
            pur = density_matrix_diagnostics(r_a)["purity"]
            if amp > best_amp:
                best_amp = amp
                best_pur = pur
        except Exception:
            continue
    return best_amp, best_pur


def main():
    print("=" * 70)
    print("Phase AJ: D-CTC channel on Systrophe LP background")
    print("=" * 70)
    print()

    # Test at multiple (omega, R) values
    configs = [
        (0.6, 1.0, "subcritical"),
        (1.0, 1.0, "supercritical-mild"),
        (1.5, 1.0, "supercritical-strong"),
        (2.0, 1.0, "deep-supercritical"),
    ]
    n_radii = 3  # so dim_ctc = 3, dim_total = 6 matching standard test

    print(f"Comparing LP-background channels at multiple (omega, R) vs Clifford/Haar:")
    print()
    print(f"{'config':25s} {'amp':8s} {'purity':8s}")

    rng_ref = np.random.default_rng(7777)

    # Reference: best Clifford from 100 trials
    best_clifford_amp = 0
    best_clifford_pur = 0
    for _ in range(100):
        U = clifford_like_unitary(2 * n_radii, rng_ref)
        amp, pur = dctc_amplification(U, dim_cr=2)
        if amp > best_clifford_amp:
            best_clifford_amp = amp
            best_clifford_pur = pur
    print(f"  {'Clifford (best of 100)':25s} {best_clifford_amp:7.4f}  {best_clifford_pur:7.4f}")

    # Reference: best Haar
    best_haar_amp = 0
    best_haar_pur = 0
    for _ in range(100):
        U = haar_random_unitary(2 * n_radii, rng_ref)
        amp, pur = dctc_amplification(U, dim_cr=2)
        if amp > best_haar_amp:
            best_haar_amp = amp
            best_haar_pur = pur
    print(f"  {'Haar (best of 100)':25s} {best_haar_amp:7.4f}  {best_haar_pur:7.4f}")
    print()

    results = {"references": {"clifford_best": [best_clifford_amp, best_clifford_pur],
                                "haar_best": [best_haar_amp, best_haar_pur]}}

    print(f"{'LP physics config':25s} {'amp':8s} {'purity':8s}")
    for omega, R, label in configs:
        vs = VanStockumInterior(omega=omega, R=R)
        # Pick r_min just outside cylinder
        r_min_use = R + 0.05
        try:
            U = lp_background_unitary(vs, n_radii=n_radii, r_min=r_min_use,
                                          r_max=R + 9.0, dim_cr=2)
            amp, pur = dctc_amplification(U, dim_cr=2)
            full_label = f"omega={omega}, R={R} ({label})"
            print(f"  {full_label:30s} {amp:7.4f}  {pur:7.4f}")
            results[label] = {"omega": omega, "R": R, "amp": float(amp),
                                "purity": float(pur)}
        except Exception as e:
            print(f"  {label}: FAILED ({e})")
            results[label] = {"error": str(e)}
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    print("Question: do LP-physics-derived unitaries support D-CTC amplification?")
    print()
    if results.get("supercritical-strong", {}).get("amp", 0) > best_haar_amp:
        print("YES: LP-derived channels match or exceed the best Haar amplification.")
        print("Physically-grounded D-CTC channels are viable computational resources.")
    elif results.get("supercritical-strong", {}).get("amp", 0) > best_clifford_amp * 0.5:
        print("PARTIAL: LP-derived channels show non-trivial amplification, but")
        print(f"less than Clifford ({best_clifford_amp:.3f}).")
    else:
        print("MARGINAL: LP-derived channels show only weak amplification.")

    # Native novelty catcher: amplification + purity across LP regimes
    # vs the Haar/Clifford references. Constant-mass histograms when
    # only one value is in the array; the catcher handles that.
    catcher_arrays = {}
    catcher_arrays["amp_clifford_best"] = np.array([best_clifford_amp])
    catcher_arrays["amp_haar_best"]     = np.array([best_haar_amp])
    for omega, R, label in configs:
        if label in results and "amp" in results[label]:
            catcher_arrays[f"amp_{label}"] = np.array([results[label]["amp"]])
    novelty = catch_novelty_in_named_arrays(catcher_arrays)
    print(f"Novelty catcher: verdict='{novelty['verdict']}', "
          f"n_sharp={len(novelty['sharp_features'])}")
    results["novelty_catcher"] = novelty

    out = Path("examples") / "dctc_deep_phase_aj_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
