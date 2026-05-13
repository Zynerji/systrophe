"""MiniSystroformer: minimal 4-layer Systroformer for copy-task validation.

A vocab-token-in / vocab-token-out model with a stack of
SystroformerBlocks. Used as the smallest end-to-end demonstration that
Systrophe-catcher modulation integrates cleanly into a transformer
training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import SystroformerBlock


class MiniSystroformer(nn.Module):
    """Tiny stacked-Systroformer-block language model."""

    def __init__(
        self,
        vocab_size: int = 10,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        radius: int = 5,
        max_seq_len: int = 64,
        n_bits: int = 32,
        lambda_scale_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            SystroformerBlock(
                d_model=d_model, n_heads=n_heads, radius=radius,
                n_bits=n_bits, lambda_scale_init=lambda_scale_init,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = int(max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq) long tensor of token ids.

        Returns logits of shape (batch, seq, vocab_size).
        """
        bs, sl = x.shape
        positions = torch.arange(sl, device=x.device).unsqueeze(0).expand(bs, sl)
        h = self.embed(x) + self.pos(positions)
        for block in self.blocks:
            h = block(h)
        return self.head(h)

    def lambda_history(self) -> list[list[float]]:
        """Return the λ₂ history from each block, one list per block."""
        return [list(b.lambda_history) for b in self.blocks]

    def current_lambdas(self) -> list[float]:
        """Most recent λ₂ value per block."""
        return [float(b.last_lambda2) for b in self.blocks]

    def lambda_scales(self) -> list[float]:
        """Current learnable modulation scales per block."""
        return [float(b.lambda_scale.detach().cpu()) for b in self.blocks]
