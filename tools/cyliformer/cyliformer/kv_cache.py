"""Selective KV cache.

Stub for the "cylinder-aware KV cache" described in Cyliformer.txt.
At inference time, attention K and V are stored only for the cylinders
deemed coherent enough (high lambda_2 / low back-reaction). Pruned
cylinders reuse the previous step's KV or contribute nothing.

This module provides the minimal data structure; the actual attention-
layer integration is left as a hook for downstream users adapting
Cyliformer to specific base models. The default Cyliformer
implementation uses the standard nn.MultiheadAttention cache (no
selective compression). The selective variant is more invasive to wire
into stock attention; we expose the building block here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SelectiveKVCache:
    """Per-layer, per-cylinder K/V buffer with a coherence-gated drop policy.

    Attributes
    ----------
    k_cache: dict[int, torch.Tensor]
        Per-cylinder K (n_cylinders -> tensor of shape (1, seq, n_heads, head_dim)).
    v_cache: dict[int, torch.Tensor]
        Per-cylinder V.
    backreaction_threshold: float
        Cylinders whose last back-reaction proxy exceeds this are
        not refreshed in the cache (their slot reuses the prior step's
        tensor; new attention queries do not append).
    """

    n_cylinders: int
    backreaction_threshold: float = 0.85
    k_cache: dict = None  # int -> Tensor
    v_cache: dict = None

    def __post_init__(self) -> None:
        if self.k_cache is None:
            self.k_cache = {}
        if self.v_cache is None:
            self.v_cache = {}

    def update(self, cylinder_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
                  backreaction: float) -> None:
        """Append k_new, v_new to this cylinder's buffer iff coherent."""
        if backreaction > self.backreaction_threshold:
            return  # drop this update
        if cylinder_idx not in self.k_cache:
            self.k_cache[cylinder_idx] = k_new
            self.v_cache[cylinder_idx] = v_new
        else:
            self.k_cache[cylinder_idx] = torch.cat(
                [self.k_cache[cylinder_idx], k_new], dim=1,
            )
            self.v_cache[cylinder_idx] = torch.cat(
                [self.v_cache[cylinder_idx], v_new], dim=1,
            )

    def get(self, cylinder_idx: int) -> tuple:
        """Return (k, v) for this cylinder, or (None, None) if empty."""
        return (self.k_cache.get(cylinder_idx), self.v_cache.get(cylinder_idx))

    def memory_footprint_bytes(self) -> int:
        """Total bytes across all cylinders' K and V tensors."""
        total = 0
        for t in list(self.k_cache.values()) + list(self.v_cache.values()):
            if isinstance(t, torch.Tensor):
                total += int(t.numel()) * int(t.element_size())
        return total

    def reset(self) -> None:
        self.k_cache = {}
        self.v_cache = {}
