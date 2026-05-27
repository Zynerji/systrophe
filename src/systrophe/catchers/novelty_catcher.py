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


def _rank_thermometer_address(
    column_values: np.ndarray, point_index: int, bits_per_comp: int,
) -> np.ndarray:
    """For a single component's values across the scan, produce a
    thermometer-coded address for the requested point's rank.
    """
    # Use stable sort so ties are broken deterministically across platforms
    # (numpy's default 'quicksort' is not stable and can disagree across
    # OS/version, breaking CI on quantised data with many tied values).
    ranks = np.argsort(np.argsort(column_values, kind="stable"), kind="stable")
    n = max(len(column_values) - 1, 1)
    bin_idx = int(ranks[point_index] / n * bits_per_comp)
    bits = np.zeros(bits_per_comp, dtype=int)
    bits[:bin_idx] = 1
    return bits


def scan_novelty(
    parameter_values: np.ndarray,
    output_fn,
    n_bits: int = 32,
    radii: tuple[int, ...] = (4, 8, 12, 16),
    sharp_threshold: float = 0.5,
    parameter_label: str = "parameter",
    data_adaptive: bool = True,
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
    sharp_threshold : sharp-feature detection threshold
    parameter_label : axis name for output records
    data_adaptive : if True (default), use rank-thermometer-encoded
        addresses for real-array outputs, computed globally across all
        parameter points. This removes false "uniform" verdicts from
        outputs that span many decades or get clipped at boundaries.
        Set to False to use the original per-output binning.
    """
    p_arr = np.asarray(parameter_values, dtype=float)

    # First pass: collect raw outputs
    raw_outputs = [output_fn(float(p)) for p in p_arr]

    # Determine output type: dict (counts) vs real array
    all_dicts = all(isinstance(r, dict) for r in raw_outputs)
    all_arrays = all(not isinstance(r, dict) for r in raw_outputs)

    addresses = []
    if data_adaptive and all_arrays:
        # Stack outputs into a (n_params, n_components) matrix
        arrs = [np.asarray(r, dtype=float).ravel() for r in raw_outputs]
        n_comp = max((len(a) for a in arrs), default=1)
        padded = np.zeros((len(arrs), n_comp))
        for i, a in enumerate(arrs):
            # Replace non-finite with median for ranking
            finite_a = np.where(np.isfinite(a), a, np.nan)
            padded[i, :len(a)] = finite_a
        # Replace columns of NaNs with zeros
        for c in range(n_comp):
            col = padded[:, c]
            mask = np.isnan(col)
            if mask.any():
                if (~mask).any():
                    median = float(np.median(col[~mask]))
                else:
                    median = 0.0
                padded[mask, c] = median
        bits_per_comp = max(1, n_bits // max(n_comp, 1))
        for i in range(len(p_arr)):
            parts = []
            for c in range(n_comp):
                parts.append(_rank_thermometer_address(
                    padded[:, c], i, bits_per_comp
                ))
            addr = np.concatenate(parts)
            if len(addr) < n_bits:
                addr = np.pad(addr, (0, n_bits - len(addr)))
            else:
                addr = addr[:n_bits]
            addresses.append(addr)
    else:
        # Original per-output binning
        for r in raw_outputs:
            if isinstance(r, dict):
                addr = bitstring_counts_to_address(r, n_bits=n_bits)
            else:
                arr = np.asarray(r, dtype=float).ravel()
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

    # Sharp-feature detection: a step is sharp if it is a "qualitative"
    # outlier --- both statistically (MAD) and in absolute terms (a
    # significant fraction of the maximum address Hamming distance,
    # which is n_bits). This is conservative; some real but subtle
    # features (rank permutations of magnitude similar to bin width)
    # will be filtered, in exchange for very few false positives from
    # bin-boundary or evenly-spaced artifacts.
    sharp = []
    if n >= 3:
        successive_distances = np.array([
            hamming_distance(addresses[k - 1], addresses[k])
            for k in range(1, n)
        ])
        if len(successive_distances) > 0:
            median_step = float(np.median(successive_distances))
            mad = float(np.median(np.abs(successive_distances - median_step)))
            # Absolute floor = 25% of n_bits (or 4, whichever larger),
            # ensures only qualitatively large excursions count when MAD ~ 0.
            absolute_floor = max(int(0.25 * n_bits), 4)
            outlier_floor = max(3 * mad, float(absolute_floor))
            for k in range(1, n):
                step = int(successive_distances[k - 1])
                excess = step - median_step
                if step >= absolute_floor and excess >= outlier_floor:
                    sharp.append({
                        "between_indices": [int(k - 1), int(k)],
                        "parameter_value": float(p_arr[k]),
                        "hamming_step": step,
                        "median_step": median_step,
                        "mad": mad,
                        "excess_over_median": float(excess),
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


def _array_to_histogram_probability(
    arr, n_bins: int = 32,
) -> np.ndarray:
    """Bin a real-valued array into a normalised histogram.

    Robust to constant columns (`hi - lo` below numpy's bin-edge floor):
    in that case all mass is placed in bin 0.
    """
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.zeros(n_bins)
    lo, hi = float(a.min()), float(a.max())
    eps = n_bins * np.finfo(float).eps * max(abs(lo), abs(hi), 1.0)
    if hi - lo <= eps:
        h = np.zeros(n_bins)
        h[0] = a.size
    else:
        h, _ = np.histogram(a, bins=n_bins, range=(lo, hi))
    s = h.sum()
    return h.astype(float) / s if s > 0 else h.astype(float)


def catch_novelty_in_named_arrays(
    named_arrays: dict,
    n_bins: int = 32,
    radii: tuple[int, ...] = (4, 8, 12, 16),
    sharp_threshold: float = 0.3,
) -> dict:
    """Convenience wrapper: hash each labelled array into a histogram
    probability vector and run the catcher on the resulting list.

    Each input array becomes one node in the address-space Hamming graph.
    Sharp Hamming steps between successive entries (in dict-insertion
    order) flag suspect structural transitions.

    Useful for retrofitting the catcher into existing scripts where the
    natural unit of comparison is "this Haar ensemble's iter-count
    histogram vs that Haar ensemble's iter-count histogram" etc.
    """
    labels = list(named_arrays.keys())
    dists = [_array_to_histogram_probability(named_arrays[k], n_bins=n_bins)
             for k in labels]
    return catch_novelty_in_distributions(
        dists, labels=labels, n_bits=n_bins, radii=radii,
        sharp_threshold=sharp_threshold,
    )


def catch_novelty_per_quantity(
    per_quantity: dict,
    n_bins: int = 32,
    radii: tuple[int, ...] = (4, 8, 12, 16),
    sharp_threshold: float = 0.3,
) -> dict:
    """Run the catcher SEPARATELY per quantity to avoid the cross-quantity
    labelling artefact.

    `per_quantity` is a nested dict::

        {
          "purity": {"haar": np.ndarray, "cliff": np.ndarray, ...},
          "iter":   {"haar": np.ndarray, "cliff": np.ndarray, ...},
          ...
        }

    Each inner dict is fed independently to `catch_novelty_in_named_arrays`,
    so the Hamming-step comparisons only ever happen between distributions
    of the *same* observable. The wrapper aggregates per-quantity verdicts
    plus a single ``aggregate_verdict`` for downstream gating.
    """
    per: dict = {}
    aggregate = "smooth"
    for quantity, named in per_quantity.items():
        if not named or len(named) < 2:
            per[quantity] = {
                "verdict": "insufficient",
                "n_distributions": len(named),
                "labels": list(named.keys()),
            }
            continue
        per[quantity] = catch_novelty_in_named_arrays(
            named, n_bins=n_bins, radii=radii,
            sharp_threshold=sharp_threshold,
        )
        v = per[quantity]["verdict"]
        if v == "novel_structure":
            aggregate = "novel_structure"
        elif v == "smooth" and aggregate != "novel_structure":
            aggregate = "smooth"
    return {
        "per_quantity": per,
        "aggregate_verdict": aggregate,
        "novel_quantities": [q for q, r in per.items()
                              if r.get("verdict") == "novel_structure"],
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
