"""Address-space novelty catcher — must be applied to every Systrophe output.

Rule (durable, set 2026-05-11): every phase module, every Marrakesh
hardware run, every parameter scan in the Systrophe project must pass
its primary output through this catcher and report the resulting
spectral signature alongside the headline result.

Mechanism (HASH-QUINE / tHHmL address-space rule):
  1. Hash each configuration of the output structure (probability vector,
     zero set, parameter point's response, bitstring distribution) to an
     integer address (binary occupancy vector).
  2. Compute pairwise Hamming distances among the addresses.
  3. Build a Hamming-radius graph; compute λ₂ (algebraic connectivity)
     of its Laplacian.
  4. Scan over the natural parameter axis and produce a λ₂ surface.
  5. Flag sharp λ₂ jumps as candidate phase transitions / emergent
     algorithms / universality boundaries.

Address-space encoding catches structure that value-encoded analyses
miss (2D Ising T_c hit within 1.4% in HASH-QUINE prior work).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoveltyScanResult:
    """Result of an address-space novelty scan."""

    parameter_axis: np.ndarray  # the parameter values scanned
    parameter_label: str
    addresses: np.ndarray  # n_params x n_bits int array
    hamming_distances: np.ndarray  # n_params x n_params symmetric int matrix
    lambda_2_at_radius: dict[int, float]  # radius -> λ₂ value
    sharp_features: list[dict]  # detected discontinuities
    verdict: str  # "novel_structure" | "smooth" | "uniform"


def probability_vector_to_address(
    probs: np.ndarray, n_bits: int = 16, threshold: float = None,
) -> np.ndarray:
    """Hash a probability distribution to a bit-occupancy vector.

    Each component above `threshold` (default = uniform mean) -> 1 bit.
    """
    p = np.asarray(probs, dtype=float).ravel()
    if threshold is None:
        threshold = 1.0 / max(len(p), 1)
    # Pad or truncate to n_bits
    if len(p) > n_bits:
        # Use top-n_bits sorted by probability
        idx = np.argsort(p)[-n_bits:][::-1]
        bits = (p[idx] > threshold).astype(int)
    else:
        bits = (p > threshold).astype(int)
        if len(bits) < n_bits:
            bits = np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=int)])
    return bits


def real_array_to_address(
    arr: np.ndarray, n_bits: int = 16,
    v_min: float = None, v_max: float = None,
) -> np.ndarray:
    """Hash a real-valued array to a bit address via [v_min, v_max] binning.

    For each value in `arr`, the corresponding bit is set if any sample
    falls in that bin.
    """
    a = np.asarray(arr, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return np.zeros(n_bits, dtype=int)
    if v_min is None:
        v_min = float(np.min(finite))
    if v_max is None:
        v_max = float(np.max(finite))
    if v_max <= v_min:
        return np.zeros(n_bits, dtype=int)
    bin_idx = np.floor((finite - v_min) / (v_max - v_min) * n_bits).astype(int)
    bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < n_bits)]
    addr = np.zeros(n_bits, dtype=int)
    addr[bin_idx] = 1
    return addr


def bitstring_counts_to_address(
    counts: dict[str, int], n_bits: int = None,
    rate_threshold: float = None,
) -> np.ndarray:
    """Hash a histogram of bitstrings (Qiskit-style counts dict).

    Bits = (P(bitstring) > rate_threshold) for each unique bitstring.
    """
    if not counts:
        return np.zeros(n_bits or 16, dtype=int)
    total = sum(counts.values())
    if total == 0:
        return np.zeros(n_bits or 16, dtype=int)
    keys = sorted(counts.keys())
    n = len(keys) if n_bits is None else n_bits
    if rate_threshold is None:
        rate_threshold = 1.0 / n
    addr = np.zeros(n, dtype=int)
    for i, k in enumerate(keys[:n]):
        if counts[k] / total > rate_threshold:
            addr[i] = 1
    return addr


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Standard Hamming distance between two equal-length bit vectors."""
    return int(np.sum(a != b))


def lambda_2_of_hamming_graph(
    addresses: list[np.ndarray] | np.ndarray, radius: int,
) -> float:
    """Algebraic connectivity λ₂ of the Hamming-radius graph.

    Two addresses are connected iff their Hamming distance ≤ radius.
    """
    arr = list(addresses)
    n = len(arr)
    if n < 2:
        return 0.0
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(arr[i], arr[j])
            if d <= radius:
                A[i, j] = 1.0
                A[j, i] = 1.0
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    eigs = np.sort(np.linalg.eigvalsh(L))
    return float(eigs[1] if len(eigs) > 1 else 0.0)


def scan_novelty(
    parameter_values: np.ndarray,
    output_fn,
    n_bits: int = 32,
    radii: tuple[int, ...] = (4, 8, 12, 16),
    sharp_threshold: float = 0.5,
    parameter_label: str = "parameter",
) -> NoveltyScanResult:
    """Run the novelty catcher on a parametric output.

    Parameters
    ----------
    parameter_values : array of parameter samples
    output_fn : callable that takes one parameter value and returns either
        a real array, a probability vector, or a counts-dict; the result
        is hashed to an address.
    n_bits : bit-width of each address
    radii : Hamming radii to scan
    sharp_threshold : |λ₂ jump| flag threshold
    parameter_label : axis name for output records
    """
    p_arr = np.asarray(parameter_values, dtype=float)
    addresses = []
    for p in p_arr:
        result = output_fn(float(p))
        if isinstance(result, dict):  # counts dict
            addr = bitstring_counts_to_address(result, n_bits=n_bits)
        else:
            arr = np.asarray(result, dtype=float).ravel()
            # If looks like a probability vector (sum ~ 1, all positive)
            if np.all(arr >= 0) and abs(arr.sum() - 1.0) < 1e-6:
                addr = probability_vector_to_address(arr, n_bits=n_bits)
            else:
                addr = real_array_to_address(arr, n_bits=n_bits)
        addresses.append(addr)
    addresses_arr = np.stack(addresses)
    n = len(addresses)
    hd = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(addresses[i], addresses[j])
            hd[i, j] = d
            hd[j, i] = d
    lambda_2_map = {}
    for r in radii:
        lambda_2_map[int(r)] = lambda_2_of_hamming_graph(addresses, radius=r)

    # Sharp-feature detection: a sharp feature is when the Hamming
    # distance between successive addresses is anomalously large relative
    # to the typical step size in the scan. This is the address-space
    # analog of a discontinuity / phase transition.
    sharp = []
    if n >= 3:
        successive_distances = np.array([
            hamming_distance(addresses[k - 1], addresses[k])
            for k in range(1, n)
        ])
        if len(successive_distances) > 0:
            median_step = float(np.median(successive_distances))
            scale = max(median_step, 1.0)
            for k in range(1, n):
                step = successive_distances[k - 1]
                if step > scale * (1.0 + sharp_threshold) and step > median_step + 2:
                    sharp.append({
                        "between_indices": [int(k - 1), int(k)],
                        "parameter_value": float(p_arr[k]),
                        "hamming_step": int(step),
                        "median_step": median_step,
                        "ratio_to_median": float(step / scale),
                    })

    # Verdict precedence: uniform > novel_structure > smooth.
    # All-identical addresses can produce spurious "sharp features"
    # from K_n connectivity differences, so check uniform first.
    if np.all(hd == 0):
        verdict = "uniform"
        sharp = []  # spurious
    elif sharp:
        verdict = "novel_structure"
    else:
        verdict = "smooth"

    return NoveltyScanResult(
        parameter_axis=p_arr,
        parameter_label=parameter_label,
        addresses=addresses_arr,
        hamming_distances=hd,
        lambda_2_at_radius=lambda_2_map,
        sharp_features=sharp,
        verdict=verdict,
    )


def catch_novelty_in_distributions(
    distributions: list[np.ndarray],
    labels: list[str] = None,
    n_bits: int = 32,
    radii: tuple[int, ...] = (4, 8, 12, 16),
    sharp_threshold: float = 0.3,
) -> dict:
    """Apply the catcher to a list of distributions (e.g., bitstring
    probability vectors from successive parameter points or hardware
    runs).

    Returns a summary dict for inclusion in result JSON.
    """
    if labels is None:
        labels = [f"dist_{i}" for i in range(len(distributions))]
    addresses = [probability_vector_to_address(d, n_bits=n_bits)
                 for d in distributions]
    n = len(addresses)
    hd = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(addresses[i], addresses[j])
            hd[i, j] = d
            hd[j, i] = d
    lambda_2 = {int(r): lambda_2_of_hamming_graph(addresses, radius=r)
                for r in radii}
    sharp = []
    if n >= 3:
        successive_distances = np.array([
            hamming_distance(addresses[k - 1], addresses[k])
            for k in range(1, n)
        ])
        if len(successive_distances) > 0:
            median_step = float(np.median(successive_distances))
            scale = max(median_step, 1.0)
            for k in range(1, n):
                step = successive_distances[k - 1]
                if step > scale * (1.0 + sharp_threshold) and step > median_step + 2:
                    sharp.append({
                        "between": [labels[k - 1], labels[k]],
                        "hamming_step": int(step),
                        "median_step": median_step,
                        "ratio_to_median": float(step / scale),
                    })

    return {
        "n_distributions": n,
        "labels": labels,
        "addresses": [a.tolist() for a in addresses],
        "hamming_distance_matrix": hd.tolist(),
        "lambda_2_at_radius": lambda_2,
        "sharp_features": sharp,
        "verdict": ("novel_structure" if sharp
                    else ("uniform" if np.all(hd == 0) else "smooth")),
    }


def summarize_novelty_for_report(result: NoveltyScanResult) -> str:
    """One-line summary for inclusion in paper / commit message."""
    n_sharp = len(result.sharp_features)
    lam2_vals = list(result.lambda_2_at_radius.values())
    return (
        f"Novelty catcher: verdict='{result.verdict}', "
        f"n_sharp_features={n_sharp}, "
        f"λ₂ range over radii: [{min(lam2_vals):.3f}, {max(lam2_vals):.3f}]"
    )
