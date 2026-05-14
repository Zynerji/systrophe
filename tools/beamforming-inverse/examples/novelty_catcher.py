"""Novelty catcher on the beamforming-inverse solver.

Sweep target-field profiles, hash each solution's (A, delta, residual)
to a bit address, compute Hamming-graph lambda_2. Sharp jumps would
indicate a regime where the inverse is ill-conditioned.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from beamforming_inverse import (
    BeamformingDesign,
    solve_beamforming_inverse,
)


N_BITS_PER_FIELD = 16


def _quantise(v: float, lo: float, hi: float, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=int)
    if not np.isfinite(v):
        return out
    if v <= lo:
        out[0] = 1; return out
    if v >= hi:
        out[-1] = 1; return out
    out[int((v - lo) / (hi - lo) * n)] = 1
    return out


def result_to_address(result, N: int) -> np.ndarray:
    parts = []
    for i in range(N):
        parts.append(_quantise(result.A[i], 0.0, 5.0, N_BITS_PER_FIELD))
        parts.append(_quantise(result.delta[i], -np.pi, np.pi, N_BITS_PER_FIELD))
    parts.append(_quantise(
        np.log10(max(result.relative_residual, 1e-12)),
        -12.0, 0.0, N_BITS_PER_FIELD,
    ))
    parts.append(_quantise(
        np.log10(max(result.condition_number, 1.0)),
        0.0, 16.0, N_BITS_PER_FIELD,
    ))
    return np.concatenate(parts)


def lambda_2_of_graph(addresses, radius):
    n = len(addresses)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = int(np.sum(addresses[i] != addresses[j]))
            if d <= radius:
                A[i, j] = 1; A[j, i] = 1
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    eigs = np.sort(np.linalg.eigvalsh(L))
    return {"lambda_2": float(eigs[1]), "mean_degree": float(deg.mean())}


def main():
    a = np.array([0.6, 0.75, 0.9])
    alpha = np.sqrt(4 * a * a - 1)
    design = BeamformingDesign(R=np.ones(3), alpha=alpha, a=a, p=np.ones(3))
    rs = np.linspace(1.05, 5.0, 40)

    # Sweep target profiles: vary "main lobe" centre + width
    centres = np.linspace(1.3, 4.5, 6)
    widths = np.linspace(0.1, 0.5, 5)
    print(f"Sweep: {len(centres)} x {len(widths)} = {len(centres)*len(widths)} targets")

    addresses = []
    for c in centres:
        for w in widths:
            z = 3.0 * np.exp(-(np.log(rs / c) ** 2) / w)
            z = z.astype(complex)
            res = solve_beamforming_inverse(design, rs, z)
            addresses.append(result_to_address(res, N=design.N))

    print()
    print("lambda_2 vs Hamming-radius threshold:")
    radii = [2, 4, 8, 16, 32, 64]
    l2s = []
    for r in radii:
        out = lambda_2_of_graph(addresses, r)
        l2s.append(out["lambda_2"])
        print(f"  radius={r:3d}: lambda_2 = {out['lambda_2']:.4f}, "
              f"mean deg = {out['mean_degree']:.1f}")

    jumps = [abs(l2s[i+1] - l2s[i]) for i in range(len(l2s) - 1)]
    max_jump = max(jumps) if jumps else 0.0

    print()
    if max_jump > 0.5:
        verdict = "novel_structure"
    elif max_jump > 0.1:
        verdict = "smooth"
    else:
        verdict = "uniform"
    print(f"VERDICT: {verdict}  (max lambda_2 jump = {max_jump:.3f})")

    out_path = pathlib.Path(__file__).parent / "novelty_catcher_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_targets": len(addresses),
            "lambda_2_by_radius": {str(r): l for r, l in zip(radii, l2s)},
            "max_jump": float(max_jump),
            "verdict": verdict,
        }, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
