"""Stress Test 2b: address-space novelty-catcher on cascade-DSI.

Implements the HASH-QUINE / tHHmL address-space encoding rule:

  data -> integer addresses -> Hamming-distance graph in address space
  -> spectral analysis (lambda_2 of graph Laplacian).

For cascade-DSI we hash each (sigma, rho) configuration's zero set into
a bit-string address by:
  - discretising the zeros' ln-coordinates onto a 1D grid
  - building a binary occupancy vector

Then compute pairwise Hamming distances among addresses across the
(sigma, rho) grid. Build a Hamming-radius graph; compute lambda_2
(algebraic connectivity) of its Laplacian.

Sharp lambda_2 features as a function of (sigma, rho) parameter
location signal phase transitions or universality classes.

Output
------
examples/stress_cascade_novelty_results.json with lambda_2 surface and
landmark points.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from systrophe.geometry.tipler_fractal import CascadeDSI


def zero_set_to_address(zeros: np.ndarray, n_bits: int = 64,
                          u_min: float = -1.0, u_max: float = 10.0) -> np.ndarray:
    """Hash a zero set to a binary occupancy vector in u = ln r coordinates.

    Discretises [u_min, u_max] into n_bits bins; each bin is 1 if any zero
    falls in it, else 0.
    """
    if len(zeros) == 0:
        return np.zeros(n_bits, dtype=int)
    u = np.log(np.maximum(zeros, 1e-12))
    valid = u[(u >= u_min) & (u < u_max)]
    if len(valid) == 0:
        return np.zeros(n_bits, dtype=int)
    bin_idx = np.floor((valid - u_min) / (u_max - u_min) * n_bits).astype(int)
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bits)]
    addr = np.zeros(n_bits, dtype=int)
    addr[bin_idx] = 1
    return addr


def address_for_grid_point(
    sigma: float, rho: float, levels: int = 4, n_bits: int = 64,
    r_max: float = np.exp(10),
) -> np.ndarray:
    """Cascade-DSI -> zero set -> address."""
    cascade = CascadeDSI(
        R=1.0, alpha_0=0.8, A_0=1.0, delta_0=0.0,
        levels=levels, scale_factor=sigma, amp_decay=rho,
    )
    zeros = cascade.zeros(r_min=1.0, r_max=r_max, n_grid=20_001)
    return zero_set_to_address(zeros, n_bits=n_bits)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Standard Hamming distance between two bit vectors."""
    return int(np.sum(a != b))


def lambda_2_of_graph(addresses: list[np.ndarray], radius: int) -> dict:
    """Build a Hamming-radius graph among addresses; return its lambda_2.

    Two addresses are connected if their Hamming distance is <= radius.
    lambda_2 is the second-smallest eigenvalue of the graph Laplacian
    (algebraic connectivity); near-0 means the graph is poorly connected
    (or disconnected), positive means it's connected.
    """
    n = len(addresses)
    A = np.zeros((n, n))
    distances = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(addresses[i], addresses[j])
            distances[i, j] = d
            distances[j, i] = d
            if d <= radius and i != j:
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
        "median_hamming": float(np.median(distances[np.triu_indices(n, k=1)])),
    }


def find_sharp_lambda_2_features(
    sigmas: np.ndarray, rhos: np.ndarray,
    lambda_2_map: np.ndarray,
) -> dict:
    """Find (sigma, rho) cells where lambda_2 jumps sharply against neighbours."""
    n_sig, n_rho = lambda_2_map.shape
    sharp = []
    for i in range(1, n_sig - 1):
        for j in range(1, n_rho - 1):
            here = lambda_2_map[i, j]
            neigh = (lambda_2_map[i-1, j] + lambda_2_map[i+1, j]
                     + lambda_2_map[i, j-1] + lambda_2_map[i, j+1]) / 4
            jump = abs(here - neigh)
            if jump > 0.5:  # ad hoc threshold
                sharp.append({"sigma": float(sigmas[i]), "rho": float(rhos[j]),
                              "lambda_2": float(here), "neighbor_mean": float(neigh),
                              "jump": float(jump)})
    return {"sharp_points": sharp}


def main():
    print("=" * 60)
    print("Address-space novelty-catcher on cascade-DSI")
    print("=" * 60)
    print()

    # Coarser grid to keep cost down (each point requires a zero-set computation
    # and the lambda_2 calculation is O(n^3))
    sigmas = np.linspace(1.5, 8.0, 8)
    rhos = np.linspace(0.4, 0.99, 8)

    print(f"Grid {len(sigmas)}x{len(rhos)} = {len(sigmas)*len(rhos)} points")
    print(f"Hashing zero sets to 64-bit addresses...")

    addresses = []
    grid_index = []
    for i, sigma in enumerate(sigmas):
        for j, rho in enumerate(rhos):
            addr = address_for_grid_point(float(sigma), float(rho))
            addresses.append(addr)
            grid_index.append((i, j))

    # Global lambda_2 calculation across all addresses
    n = len(addresses)
    print(f"\nGlobal Hamming-distance statistics ({n} addresses):")
    dist_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(addresses[i], addresses[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    upper = dist_matrix[np.triu_indices(n, k=1)]
    print(f"  median Hamming = {np.median(upper):.1f}")
    print(f"  max Hamming    = {upper.max()}")
    print(f"  min Hamming    = {upper.min()}")

    # Compute lambda_2 at several radius thresholds
    print()
    print("lambda_2 vs Hamming-radius threshold:")
    radii = [4, 8, 12, 16, 24]
    global_results = {}
    for radius in radii:
        result = lambda_2_of_graph(addresses, radius)
        global_results[radius] = result
        print(f"  radius={radius:2d}: lambda_2 = {result['lambda_2']:.4f}, "
              f"mean degree = {result['mean_degree']:.1f}")

    # Local lambda_2: for each grid point, do a 3x3 neighbourhood lambda_2
    print()
    print("Local 3x3-neighbourhood lambda_2 surface:")
    radius = 8  # typical
    local_l2 = np.zeros((len(sigmas), len(rhos)))
    for i in range(len(sigmas)):
        for j in range(len(rhos)):
            i_range = range(max(0, i-1), min(len(sigmas), i+2))
            j_range = range(max(0, j-1), min(len(rhos), j+2))
            local_addrs = []
            for ii in i_range:
                for jj in j_range:
                    idx = ii * len(rhos) + jj
                    local_addrs.append(addresses[idx])
            r = lambda_2_of_graph(local_addrs, radius=radius)
            local_l2[i, j] = r["lambda_2"]

    sharp = find_sharp_lambda_2_features(sigmas, rhos, local_l2)
    print(f"  sharp features (|jump| > 0.5): {len(sharp['sharp_points'])}")
    for sp in sharp["sharp_points"][:10]:
        print(f"    sigma={sp['sigma']:.2f}, rho={sp['rho']:.2f}, "
              f"lambda_2={sp['lambda_2']:.3f}, jump={sp['jump']:.3f}")

    print()
    print("Local-lambda_2 matrix:")
    for i in range(len(sigmas)):
        row = "  ".join(f"{local_l2[i,j]:.2f}" for j in range(len(rhos)))
        print(f"  sigma={sigmas[i]:.2f}: {row}")

    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if len(sharp["sharp_points"]) >= 2:
        print(f"Found {len(sharp['sharp_points'])} sharp lambda_2 features --- candidate")
        print("phase boundaries in the cascade-DSI parameter space.")
    else:
        print("No sharp lambda_2 features; cascade-DSI parameter space appears smooth")
        print("in address-space encoding at this resolution.")
    print()

    payload = {
        "sigmas": sigmas.tolist(),
        "rhos": rhos.tolist(),
        "local_lambda_2_surface": local_l2.tolist(),
        "global_at_radii": global_results,
        "sharp_features": sharp,
        "n_addresses": n,
        "median_hamming": float(np.median(upper)),
    }
    out_path = Path("examples") / "stress_cascade_novelty_results.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
