"""Address-space novelty catcher on the implosion-carving parameter space.

Follows the HASH-QUINE / tHHmL rule mandated for every Systrophe
deliverable: hash the carved-pocket geometric descriptor to a bit
address, compute Hamming-graph lambda_2 across an (omega, r_target)
sweep, report sharp features.

Inputs to address: M_engineered, impact_parameter, omega_orbit,
is_stable, closure_residual_dbdr (clipped). Each is quantised to
log-bin occupancy in a fixed range; the address is the concatenation.

Expected verdict: SMOOTH (the carving map is analytic in (omega,
r_target); we should see lambda_2 trend monotonically with the
Hamming-radius cutoff, no sharp jumps). A jump would indicate a
discontinuity in the carving pipeline (e.g. a region where Brent
solver fails to converge).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from implosion_carving import ImplosionCarver


N_BITS_PER_FIELD = 16
N_FIELDS = 5  # M, b, omega_orbit, stable_bit, log10|res|
N_BITS = N_BITS_PER_FIELD * N_FIELDS


def _quantise(value: float, v_min: float, v_max: float, n_bits: int) -> np.ndarray:
    """One-hot bin: discretise [v_min, v_max] into n_bits cells, return
    a bit vector with the occupied bin set to 1. NaN -> all zeros."""
    out = np.zeros(n_bits, dtype=int)
    if not np.isfinite(value):
        return out
    if value <= v_min:
        out[0] = 1
        return out
    if value >= v_max:
        out[-1] = 1
        return out
    bin_idx = int((value - v_min) / (v_max - v_min) * n_bits)
    bin_idx = max(0, min(n_bits - 1, bin_idx))
    out[bin_idx] = 1
    return out


def pocket_to_address(pocket) -> np.ndarray:
    """Hash a PocketGeometry into a bit address."""
    M = pocket.M_engineered if pocket.M_engineered is not None else float("nan")
    parts = [
        _quantise(M,            0.0, 5.0, N_BITS_PER_FIELD),
        _quantise(pocket.impact_parameter, -10.0, 10.0, N_BITS_PER_FIELD),
        _quantise(pocket.omega_orbit,      -2.0,  2.0, N_BITS_PER_FIELD),
        _quantise(1.0 if pocket.is_stable else 0.0, 0.0, 1.0, N_BITS_PER_FIELD),
        _quantise(
            np.log10(max(abs(pocket.closure_residual_dbdr), 1e-12)),
            -12.0, 0.0, N_BITS_PER_FIELD,
        ),
    ]
    return np.concatenate(parts)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def lambda_2_of_graph(addresses: list[np.ndarray], radius: int) -> dict:
    n = len(addresses)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(addresses[i], addresses[j])
            if d <= radius:
                A[i, j] = 1
                A[j, i] = 1
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    eigs = np.sort(np.linalg.eigvalsh(L))
    return {
        "lambda_1": float(eigs[0]),
        "lambda_2": float(eigs[1]),
        "lambda_max": float(eigs[-1]),
        "mean_degree": float(deg.mean()),
    }


def main():
    print("Address-space novelty catcher on implosion-carving")
    print("=" * 60)

    omegas = np.linspace(0.5, 2.5, 5)
    r_targets = np.linspace(1.2, 2.5, 5)
    print(f"Sweep: {len(omegas)} x {len(r_targets)} = "
          f"{len(omegas)*len(r_targets)} pockets")

    addresses = []
    n_carved = 0
    n_failed = 0
    for omega in omegas:
        for r_t in r_targets:
            car = ImplosionCarver(omega=float(omega), R=1.0)
            pocket = car.carve(r_target=float(r_t))
            addresses.append(pocket_to_address(pocket))
            if pocket.is_carved:
                n_carved += 1
            else:
                n_failed += 1
    print(f"  carved: {n_carved}  failed: {n_failed}")

    print()
    print("lambda_2 vs Hamming-radius threshold:")
    radii = [2, 4, 8, 16, 32]
    results = {}
    for r in radii:
        out = lambda_2_of_graph(addresses, radius=r)
        results[r] = out
        print(f"  radius={r:2d}: lambda_2 = {out['lambda_2']:.4f}, "
              f"mean deg = {out['mean_degree']:.1f}")

    # Detect a sharp jump in lambda_2 across the radius sweep.
    l2s = [results[r]["lambda_2"] for r in radii]
    jumps = [abs(l2s[i+1] - l2s[i]) for i in range(len(l2s) - 1)]
    max_jump = max(jumps) if jumps else 0.0

    print()
    if max_jump > 0.5:
        verdict = "novel_structure"
        msg = (f"Found max lambda_2 jump = {max_jump:.3f} across radii — "
               "candidate phase boundary in carving parameter space.")
    elif max_jump > 0.1:
        verdict = "smooth"
        msg = (f"Max lambda_2 jump = {max_jump:.3f}. Smooth crossover, no "
               "discontinuity in the carving pipeline.")
    else:
        verdict = "uniform"
        msg = (f"Max lambda_2 jump = {max_jump:.3f}. Spectrum is uniform — "
               "carving map is analytic in (omega, r_target).")

    print(f"VERDICT: {verdict}")
    print(f"  {msg}")

    payload = {
        "n_pockets": len(addresses),
        "n_carved": n_carved,
        "n_failed": n_failed,
        "lambda_2_by_radius": {str(r): results[r]["lambda_2"] for r in radii},
        "max_lambda_2_jump": float(max_jump),
        "verdict": verdict,
    }
    out_path = pathlib.Path(__file__).parent / "carve_novelty_catcher_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
