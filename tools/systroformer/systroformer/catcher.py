"""Catcher primitives reused from systrophe.catchers.novelty_catcher.

This module provides thin torch-aware wrappers and one new utility
(power-iteration λ₂) tuned for inline neural-net use. The framework's
core address-space hashing and Hamming-graph definitions are imported
directly from `systrophe.catchers.novelty_catcher` to avoid duplication.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Re-export the framework primitives so callers can stay inside one
# module if they prefer.
from systrophe.catchers.novelty_catcher import (
    hamming_distance,
    lambda_2_of_hamming_graph as hamming_graph_lambda2,
    real_array_to_address,
)


def address_from_activation(
    activation: np.ndarray, n_bits: int = 32,
) -> np.ndarray:
    """Address-space encoding of a 1D activation vector.

    Uses Systrophe's `real_array_to_address` (rank-thermometer style
    binarisation). Returns a 1D int array of length n_bits.

    Operates on numpy. The block-level Systroformer code calls this
    on a detached CPU copy of the activation tensor.
    """
    return real_array_to_address(activation, n_bits=n_bits)


def hamming_graph_lambda2_power_iter(
    addresses: Iterable[np.ndarray], radius: int = 5,
    n_iter: int = 20, max_nodes: int = 512,
    rng: np.random.Generator | None = None,
) -> float:
    """λ₂ approximation via power iteration on the Hamming-graph Laplacian.

    Much faster than the exact eigvalsh (used in the framework's
    `lambda_2_of_hamming_graph`) -- 100x-ish at n=512. Suitable for
    inline use in a forward pass.

    Strategy:
      1. Subsample to at most `max_nodes` addresses.
      2. Build the binary adjacency matrix (Hamming dist ≤ radius).
      3. Compute the Laplacian L = D - A.
      4. Power iteration with periodic re-orthogonalisation against
         the all-ones eigenvector (the trivial λ=0 eigenvector).
      5. Return the Rayleigh quotient as λ₂ estimate.
    """
    addrs = list(addresses)
    if len(addrs) > max_nodes:
        # Deterministic stride subsample (so test results don't depend on rng)
        stride = max(1, len(addrs) // max_nodes)
        addrs = addrs[::stride][:max_nodes]
    n = len(addrs)
    if n < 2:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(0)

    # Adjacency
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        ai = addrs[i]
        for j in range(i + 1, n):
            d = int(np.sum(ai != addrs[j]))
            if d <= radius:
                A[i, j] = 1.0
                A[j, i] = 1.0
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    # Project out the all-ones (λ_1 = 0 eigenvector) every iteration.
    ones = np.ones(n) / np.sqrt(n)
    v = rng.normal(size=n)
    v = v - np.dot(v, ones) * ones
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return 0.0
    v = v / norm

    # Use shift-and-invert trick: largest eigval of (||L||*I - L) is
    # ||L|| - λ_2, so we power-iterate (||L||*I - L) instead.
    shift = float(np.max(deg)) * 2.0 + 1.0
    M = shift * np.eye(n) - L
    for _ in range(n_iter):
        v = M @ v
        # Re-orthogonalise against trivial eigvec
        v = v - np.dot(v, ones) * ones
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            break
        v = v / nv
    # Rayleigh quotient on L
    lambda_2 = float(v @ L @ v)
    return max(0.0, lambda_2)


def derivative_catcher(lambda_history: list, window: int = 5) -> float:
    """Mean first-difference of the last `window` λ₂ values.

    Large positive values flag a recent sharp increase in λ₂ —
    a candidate phase transition / emergent structure event.
    """
    if len(lambda_history) < window:
        return 0.0
    recent = np.array(lambda_history[-window:], dtype=float)
    return float(np.mean(np.diff(recent)))
