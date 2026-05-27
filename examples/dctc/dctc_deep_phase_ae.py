"""D-CTC Phase AE: state-distinguisher payoff test (Aaronson-Watrous).

Hypothesis: high-purity D-CTC channels can distinguish two close
mixed states sigma_a, sigma_b better than the optimal Helstrom
bound permitted by ordinary quantum mechanics.

Setup:
  1. Pick two close states sigma_a, sigma_b on CR with overlap close
     to 1 (so Helstrom success rate is just above 50%).
  2. For a given U, find the D-CTC fixed point rho_FP(sigma).
  3. Measure: does the fixed-point map sigma_a -> rho_FP_a differ
     more from sigma_b -> rho_FP_b than the Helstrom-allowed
     distinguishability?

Concretely:
  - Helstrom: 1/2 + ||sigma_a - sigma_b||_1 / 4
  - D-CTC:    1/2 + ||rho_FP(sigma_a) - rho_FP(sigma_b)||_1 / 4

If D-CTC succeeds when Helstrom can't (i.e., post-fixed-point
distance > input distance), the channel amplifies distinguishability.

We expect Clifford-structured U with high-purity fixed points to
show amplification; Haar-random U should not.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from systrophe.ctc.d_ctc import dctc_fixed_point, density_matrix_diagnostics


def haar_random_unitary(dim, rng):
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    diag_R = np.diag(R)
    phases = diag_R / np.abs(diag_R)
    return Q * phases


def clifford_like_unitary(dim, rng):
    """Permutation @ diagonal-of-fourth-roots, from Phase I."""
    perm = rng.permutation(dim)
    P = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        P[i, perm[i]] = 1.0
    D = np.diag(rng.choice([1, -1, 1j, -1j], dim))
    return P @ D


def trace_distance(rho_a: np.ndarray, rho_b: np.ndarray) -> float:
    diff = rho_a - rho_b
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return 0.5 * float(np.sum(np.abs(eigs)))


def fixed_point_for_sigma(U, sigma_cr, dim_cr, rho_init):
    r = dctc_fixed_point(U, sigma_cr, dim_cr=dim_cr,
                          rho_ctc_init=rho_init, tol=1e-10, max_iter=5000)
    return r["rho_ctc"], r["iterations"]


def main():
    print("=" * 70)
    print("Phase AE: state distinguisher (Aaronson-Watrous payoff test)")
    print("=" * 70)
    print()

    dim_cr, dim_ctc = 2, 3
    dim_total = dim_cr * dim_ctc
    rng = np.random.default_rng(666)

    # Define close states sigma_a, sigma_b on CR
    # sigma_a = |0><0|, sigma_b = (1-eps)|0><0| + eps |+><+|
    # for various eps. Trace distance is bounded by eps.
    eps = 0.1  # small perturbation
    sigma_a = np.zeros((dim_cr, dim_cr), dtype=complex); sigma_a[0, 0] = 1.0
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    plus_proj = np.outer(plus, plus.conj())
    sigma_b = (1 - eps) * sigma_a + eps * plus_proj

    input_tr_dist = trace_distance(sigma_a, sigma_b)
    helstrom_p_succ = 0.5 + input_tr_dist / 2
    print(f"Input states: sigma_a = |0><0|, sigma_b = mix(eps={eps})")
    print(f"  Input trace distance: {input_tr_dist:.4f}")
    print(f"  Helstrom P(success):  {helstrom_p_succ:.4f}")
    print()

    # Use the same rho_init for both channels (random pure state)
    psi = rng.standard_normal(dim_ctc) + 1j * rng.standard_normal(dim_ctc)
    psi = psi / np.linalg.norm(psi)
    rho_init = np.outer(psi, psi.conj())

    # Sample channels and measure amplification
    n = 500
    for kind in ("haar", "clifford"):
        print(f"--- {kind} channels (n={n}) ---")
        rng_k = np.random.default_rng(7777 + hash(kind) % 1000)
        amplifications = []
        purities_a = []
        purities_b = []
        for _ in range(n):
            if kind == "haar":
                U = haar_random_unitary(dim_total, rng_k)
            else:
                U = clifford_like_unitary(dim_total, rng_k)

            rho_a_fp, _ = fixed_point_for_sigma(U, sigma_a, dim_cr, rho_init)
            rho_b_fp, _ = fixed_point_for_sigma(U, sigma_b, dim_cr, rho_init)
            output_tr_dist = trace_distance(rho_a_fp, rho_b_fp)
            amp = output_tr_dist / max(input_tr_dist, 1e-12)
            amplifications.append(amp)
            purities_a.append(density_matrix_diagnostics(rho_a_fp)["purity"])
            purities_b.append(density_matrix_diagnostics(rho_b_fp)["purity"])

        amplifications = np.array(amplifications)
        purities_a = np.array(purities_a)
        purities_b = np.array(purities_b)
        print(f"  mean amplification: {amplifications.mean():.4f} "
              f"(amplification > 1 means D-CTC > Helstrom)")
        print(f"  max amplification:  {amplifications.max():.4f}")
        print(f"  P(amp > 1):         {float(np.mean(amplifications > 1.0)):.4f}")
        print(f"  P(amp > 1.5):       {float(np.mean(amplifications > 1.5)):.4f}")
        print(f"  P(amp > 2):         {float(np.mean(amplifications > 2.0)):.4f}")
        print(f"  P(amp > 3):         {float(np.mean(amplifications > 3.0)):.4f}")
        print(f"  Pearson(amp, purity_a): {float(np.corrcoef(amplifications, purities_a)[0,1]):+.4f}")
        print(f"  mean purity sigma_a -> rho: {purities_a.mean():.4f}, "
              f"max: {purities_a.max():.4f}")
        print()

    print("=" * 70)
    print("Verdict")
    print("=" * 70)
    print()
    print("If D-CTC channels amplify trace distance beyond 1 (i.e., output")
    print("distance > input distance), the post-iteration state pair can be")
    print("distinguished better than Helstrom allows on the inputs.")
    print()
    print("Aaronson-Watrous: this amplification is the source of D-CTC")
    print("polynomial-time PSPACE computation. We expect Clifford-like")
    print("channels (high-purity class) to show amplification; Haar should not.")


if __name__ == "__main__":
    main()
