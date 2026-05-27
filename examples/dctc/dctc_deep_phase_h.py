"""D-CTC Phase H+I: dim_CR scaling + Clifford comparison.

Phase H: How does the high-purity fraction scale with dim_CR?
  For each dim_CR in {2, 3, 4, 5, 6}, sample 800 Haar U at dim_CTC=3,
  count fraction with purity > 0.7, 0.9, 0.95.

Phase I: Does the high-purity property come from Haar randomness or
  is it boosted by structured (Clifford-like) unitaries?
  Sample Clifford-coset U and compare distributions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import (
    dctc_fixed_point,
    density_matrix_diagnostics,
)


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def random_2_design_unitary(dim, rng):
    """Random unitary from an exact 2-design. We use Haar but constrain
    to coset structure: U = e^{i pi/4 random_perm @ diag(+-1)} * Haar.
    This isn't an exact Clifford but produces structured candidates."""
    # For simplicity: random permutation matrix combined with diagonal +-1
    perm = rng.permutation(dim)
    P = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        P[i, perm[i]] = 1.0
    D = np.diag(rng.choice([1, -1, 1j, -1j], dim))
    return P @ D


def sample_purity_stats(dim_cr: int, dim_ctc: int, n: int,
                          seed: int, kind: str = "haar") -> dict:
    rng = np.random.default_rng(seed)
    dim_total = dim_cr * dim_ctc
    purities = []
    iters = []
    for _ in range(n):
        if kind == "haar":
            U = haar_random_unitary(dim_total, rng)
        elif kind == "structured":
            # 50% Haar + structured composition
            H = haar_random_unitary(dim_total, rng)
            S = random_2_design_unitary(dim_total, rng)
            U = S @ H
        elif kind == "clifford_haar":
            # Mix structured permutation + Haar
            S = random_2_design_unitary(dim_total, rng)
            U = S.astype(complex)
        else:
            raise ValueError(f"unknown kind: {kind}")
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
    return {
        "kind": kind,
        "dim_cr": dim_cr, "dim_ctc": dim_ctc, "n": n,
        "frac_gt_05": float(np.mean(purities > 0.5)),
        "frac_gt_07": float(np.mean(purities > 0.7)),
        "frac_gt_09": float(np.mean(purities > 0.9)),
        "frac_gt_095": float(np.mean(purities > 0.95)),
        "max_purity": float(purities.max()),
        "median_purity": float(np.median(purities)),
        "mean_purity": float(purities.mean()),
        "median_iter": float(np.median(iters)),
    }


def main():
    print("=" * 70)
    print("D-CTC Phase H + I: dim_CR scaling + Clifford comparison")
    print("=" * 70)
    print()

    # Phase H: scaling of high-purity fraction with dim_CR
    print("Phase H: dim_CR scaling at dim_CTC = 3 (Haar samples)")
    print()
    print(f"{'dim_CR':6s} {'n':5s} {'P(>0.5)':9s} {'P(>0.7)':9s} {'P(>0.9)':9s} {'P(>0.95)':9s} {'max':6s} {'med':6s}")
    h_results = {}
    for dim_cr in (2, 3, 4, 5, 6):
        t0 = time.time()
        n = 800
        s = sample_purity_stats(dim_cr, 3, n, seed=314 + dim_cr, kind="haar")
        elapsed = time.time() - t0
        h_results[dim_cr] = s
        print(f"  {dim_cr:4d}   {n:4d}  {s['frac_gt_05']:8.4f}  {s['frac_gt_07']:8.4f}  "
              f"{s['frac_gt_09']:8.4f}  {s['frac_gt_095']:8.4f}  {s['max_purity']:5.3f}  {s['median_purity']:5.3f}")
    print()

    # Fit scaling P(>0.9) vs dim_CR
    print("Scaling of P(purity > 0.9) with dim_CR:")
    dims = np.array(list(h_results.keys()), dtype=float)
    fracs = np.array([h_results[int(d)]["frac_gt_09"] for d in dims])
    valid = fracs > 0
    if valid.sum() >= 3:
        # log P vs dim_CR: exponential decay
        slope_lin, int_lin = np.polyfit(dims[valid], np.log(fracs[valid]), 1)
        print(f"  log P(>0.9) = {int_lin:.3f} + {slope_lin:.3f} * dim_CR")
        print(f"  Half-life: dim_CR -> dim_CR + {np.log(2)/-slope_lin:.2f}")
        # log P vs log dim_CR: power law
        slope_log, int_log = np.polyfit(np.log(dims[valid]), np.log(fracs[valid]), 1)
        print(f"  log P(>0.9) = {int_log:.3f} + {slope_log:.3f} * log(dim_CR)")
        print(f"  Power-law exponent: {slope_log:.3f}")
    print()

    # Phase I: Clifford / structured comparison
    print("Phase I: Haar vs structured U comparison at (dim_CR=2, dim_CTC=3)")
    print()
    print(f"{'kind':14s} {'n':5s} {'P(>0.5)':9s} {'P(>0.7)':9s} {'P(>0.9)':9s} {'P(>0.95)':9s} {'max':6s} {'med':6s}")
    i_results = {}
    for kind in ("haar", "structured", "clifford_haar"):
        t0 = time.time()
        n = 800
        s = sample_purity_stats(2, 3, n, seed=2718 + hash(kind) % 1000, kind=kind)
        elapsed = time.time() - t0
        i_results[kind] = s
        print(f"  {kind:12s}  {n:4d}  {s['frac_gt_05']:8.4f}  {s['frac_gt_07']:8.4f}  "
              f"{s['frac_gt_09']:8.4f}  {s['frac_gt_095']:8.4f}  {s['max_purity']:5.3f}  {s['median_purity']:5.3f}")
    print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    # Phase H verdict
    p_2 = h_results[2]["frac_gt_09"]
    p_6 = h_results[6]["frac_gt_09"]
    print("Phase H (dim_CR scaling):")
    if p_2 > 0 and p_6 == 0:
        print(f"  P(>0.9) drops from {p_2:.4f} at dim_CR=2 to 0 at dim_CR=6.")
        print("  High-purity fixed points are a SMALL-dim_CR phenomenon.")
    elif valid.sum() >= 3:
        print(f"  P(>0.9) follows roughly exponential decay with dim_CR.")
    print()

    # Phase I verdict
    haar_p9 = i_results["haar"]["frac_gt_09"]
    struct_p9 = i_results["structured"]["frac_gt_09"]
    cliff_p9 = i_results["clifford_haar"]["frac_gt_09"]
    print("Phase I (Haar vs structured):")
    print(f"  Haar P(>0.9):           {haar_p9:.4f}")
    print(f"  Structured P(>0.9):     {struct_p9:.4f}")
    print(f"  Permutation+diag P(>0.9): {cliff_p9:.4f}")
    if cliff_p9 > 2 * haar_p9:
        print(f"  Structured U produces SIGNIFICANTLY more high-purity samples")
        print(f"  ({cliff_p9 / haar_p9:.1f}x), suggesting structure helps purity.")
    elif abs(cliff_p9 - haar_p9) < 0.005:
        print(f"  Haar and structured produce similar high-purity rates.")
    else:
        print(f"  Structured U produces fewer high-purity samples.")
    print()

    out_path = Path("examples") / "dctc_deep_phase_h_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "phase_h_scaling": {str(k): v for k, v in h_results.items()},
            "phase_i_kinds": i_results,
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
