"""Differentiable address-space catcher for Cyliformer.

Unlike the systroformer/catcher.py wrapper (which converts to numpy and
runs on CPU), this implementation stays in pytorch so the catcher
signal participates in autograd. Power-iteration lambda_2 + straight-
through binarisation; runs on the same device as the input.

Provenance: this reuses the *concept* of the Systrophe address-space
lambda_2 (`systrophe.novelty_catcher.lambda_2_of_hamming_graph`) but
re-implements it in torch for inline use during forward / backward.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedAddressCatcher(nn.Module):
    """Differentiable Hamming-graph lambda_2 over learned binary addresses.

    Forward pass:
      1. Project activations (..., d_model) -> logits (..., n_bits).
      2. Binarise via straight-through estimator.
      3. Subsample to at most `max_nodes` addresses.
      4. Build the binary adjacency matrix using Hamming distance <= radius.
      5. Compute the graph Laplacian L = D - A.
      6. Power-iterate (||L||*I - L) starting from a random vector projected
         away from the all-ones eigenvector; the largest eigenvalue of
         (||L||*I - L) corresponds to the smallest non-trivial L eigenvalue.
      7. Return the Rayleigh quotient v^T L v as a scalar tensor with
         autograd attached to the projection weights.

    Output is normalised by the number of nodes so the scale is roughly
    in [0, max_deg/n] -- typically [0, 1] in practice.
    """

    def __init__(
        self,
        d_model: int,
        n_bits: int = 32,
        radius: int = 8,
        max_nodes: int = 128,
        n_power_iter: int = 8,
        normalise: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_bits = int(n_bits)
        self.radius = int(radius)
        self.max_nodes = int(max_nodes)
        self.n_power_iter = int(n_power_iter)
        self.normalise = bool(normalise)
        self.proj = nn.Linear(d_model, n_bits, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model). Returns scalar lambda_2 tensor."""
        flat = x.reshape(-1, self.d_model)
        logits = self.proj(flat)
        hard = (logits > 0).float()
        bits = hard + (logits - logits.detach())  # straight-through

        n = bits.shape[0]
        if n > self.max_nodes:
            stride = max(1, n // self.max_nodes)
            bits = bits[::stride][: self.max_nodes]
            n = bits.shape[0]
        if n < 2:
            return torch.zeros((), device=x.device, dtype=x.dtype)

        # Hamming distance == L1 distance for {0,1} vectors
        d_mat = torch.cdist(bits, bits, p=1.0)
        adj = (d_mat <= float(self.radius)).float()
        adj = adj * (1.0 - torch.eye(n, device=x.device, dtype=adj.dtype))
        deg = adj.sum(dim=-1)
        L = torch.diag(deg) - adj

        shift = deg.max().detach() * 2.0 + 1.0
        M = shift * torch.eye(n, device=x.device, dtype=L.dtype) - L

        ones = torch.full((n,), 1.0 / float(n) ** 0.5, device=x.device, dtype=L.dtype)
        # Deterministic init (rotates with the projection's weights)
        v = bits[:, : 1].squeeze(-1)
        v = v - (v @ ones) * ones
        norm = v.norm() + 1e-12
        if float(norm.item()) < 1e-6:
            # Fallback to a per-node-index parity vector
            v = torch.arange(n, device=x.device, dtype=L.dtype) % 2 * 2.0 - 1.0
            v = v - (v @ ones) * ones
            v = v / (v.norm() + 1e-12)
        else:
            v = v / norm

        for _ in range(self.n_power_iter):
            v = M @ v
            v = v - (v @ ones) * ones
            nv = v.norm()
            if float(nv.item()) < 1e-12:
                break
            v = v / nv

        # Rayleigh quotient on L
        lambda2 = (v @ L @ v).clamp(min=0.0)
        if self.normalise:
            lambda2 = lambda2 / float(n)
        return lambda2.clamp(min=0.0, max=4.0)
