"""Systroformer utilities: LSH approximation + learned address net.

These let the catcher scale beyond the O(N^2) full-graph regime.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def lsh_subsample(
    addresses: list[np.ndarray], max_nodes: int = 256, rng_seed: int = 0,
) -> list[np.ndarray]:
    """LSH-style subsampling: keep at most max_nodes evenly-strided.

    Deterministic by default (stride sample); set rng_seed != 0 to
    permute first. Trivial baseline for full-graph approximation; a
    real LSH (MinHash / SimHash) would cluster similar addresses first.
    """
    if len(addresses) <= max_nodes:
        return list(addresses)
    if rng_seed == 0:
        stride = max(1, len(addresses) // max_nodes)
        return list(addresses[::stride][:max_nodes])
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(len(addresses), size=max_nodes, replace=False)
    idx.sort()
    return [addresses[int(i)] for i in idx]


class LearnedAddressNet(nn.Module):
    """Tiny linear-then-binarise address net.

    Maps d_model activations to an n_bits bit-vector via a learnable
    linear projection followed by sign-of-output binarisation. The
    binarisation gradient is straight-through.
    """

    def __init__(self, d_model: int, n_bits: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_bits, bias=True)
        self.n_bits = int(n_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> bits: (..., n_bits) float in {0., 1.}.

        Uses straight-through estimator: forward is hard-binarise, but
        the gradient passes through the linear projection unmodified.
        """
        logits = self.proj(x)
        hard = (logits > 0).float()
        # Straight-through trick
        return hard + (logits - logits.detach())
