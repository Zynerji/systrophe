"""Address-space novelty catcher on the guderley-shock comparison surface.

Sweeps (gamma, omega) and hashes (Guderley power, QFTCS power, residual)
to a bit address. Reports lambda_2 across Hamming-radius cutoffs.

Expected: SMOOTH (the comparison is analytic in both axes). The
Guderley power depends on gamma only; the QFTCS power is universally
-1 across the supercritical regime, so the residual surface should be
flat in omega and shift slightly with gamma.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from systrophe.geometry.vanstockum import VanStockumInterior

from guderley_shock import compare_to_cauchy_horizon


N_BITS_PER_FIELD = 16


def _quantise(v, lo, hi, n):
    out = np.zeros(n, dtype=int)
    if not np.isfinite(v):
        return out
    if v <= lo:
        out[0] = 1; return out
    if v >= hi:
        out[-1] = 1; return out
    out[int((v - lo) / (hi - lo) * n)] = 1
    return out


def address_for(cmp_):
    parts = [
        _quantise(cmp_.guderley_density_power, -1.5, 0.0, N_BITS_PER_FIELD),
        _quantise(cmp_.qftcs_T_tt_power,       -1.5, 0.0, N_BITS_PER_FIELD),
        _quantise(cmp_.absolute_residual,       0.0, 0.5, N_BITS_PER_FIELD),
    ]
    return np.concatenate(parts)


def lambda_2_of_graph(addresses, radius):
    n = len(addresses)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if int(np.sum(addresses[i] != addresses[j])) <= radius:
                A[i, j] = 1; A[j, i] = 1
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    eigs = np.sort(np.linalg.eigvalsh(L))
    return {"lambda_2": float(eigs[1]), "mean_degree": float(deg.mean())}


def main():
    print("Address-space novelty catcher: guderley-shock comparison surface")
    print("=" * 60)

    gammas = [5.0 / 3.0, 7.0 / 5.0]  # only literature values
    omegas = [1.5, 2.0, 2.5, 3.0]  # all supercritical (a > 1/2 at R=1)

    print(f"Sweep: {len(gammas)} gamma x {len(omegas)} omega = "
          f"{len(gammas)*len(omegas)} comparisons")
    addresses = []
    for gamma in gammas:
        for omega in omegas:
            vs = VanStockumInterior(omega=omega, R=1.0)
            cmp_ = compare_to_cauchy_horizon(vs, gamma=gamma, n=3)
            addresses.append(address_for(cmp_))

    print()
    print("lambda_2 vs Hamming-radius threshold:")
    radii = [2, 4, 8, 16, 32]
    l2s = []
    for r in radii:
        out = lambda_2_of_graph(addresses, r)
        l2s.append(out["lambda_2"])
        print(f"  radius={r:3d}: lambda_2 = {out['lambda_2']:.4f}, "
              f"mean deg = {out['mean_degree']:.2f}")
    jumps = [abs(l2s[i+1] - l2s[i]) for i in range(len(l2s) - 1)]
    max_jump = max(jumps) if jumps else 0.0
    if max_jump > 0.5:
        verdict = "novel_structure"
    elif max_jump > 0.1:
        verdict = "smooth"
    else:
        verdict = "uniform"
    print()
    print(f"VERDICT: {verdict}  (max lambda_2 jump = {max_jump:.3f})")

    out_path = pathlib.Path(__file__).parent / "novelty_catcher_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_comparisons": len(addresses),
            "lambda_2_by_radius": {str(r): l for r, l in zip(radii, l2s)},
            "max_jump": float(max_jump),
            "verdict": verdict,
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
