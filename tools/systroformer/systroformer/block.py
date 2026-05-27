"""SystroformerBlock: a transformer block whose FFN output is modulated by
the address-space λ₂ of its attention activations.

After self-attention, the activations are hashed to address-space
(rank-thermometer 32-bit), a Hamming-radius graph is built, and λ₂
(algebraic connectivity) is computed. The FFN output is multiplied by
(1 + lambda_scale * λ₂) before the residual add — so the model can
learn to amplify or dampen its computation in response to detected
information-topological structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .catcher import (
    address_from_activation,
    derivative_catcher,
    hamming_graph_lambda2_power_iter,
)


class SystroformerBlock(nn.Module):
    """Transformer block with Systrophe-catcher modulation."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ffn_mult: int = 4,
        radius: int = 5,
        n_bits: int = 32,
        lambda_scale_init: float = 0.05,
        max_nodes: int = 256,
        approximate_lambda2: bool = True,
        history_window: int = 5,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Learnable modulation strength
        self.lambda_scale = nn.Parameter(torch.tensor(float(lambda_scale_init)))

        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.max_nodes = int(max_nodes)
        self.approximate = bool(approximate_lambda2)
        self.history_window = int(history_window)

        # Diagnostic state (not in state_dict)
        self.lambda_history: list[float] = []
        self.last_lambda2: float = 0.0
        self.last_derivative: float = 0.0

    def _compute_lambda2(self, x: torch.Tensor) -> float:
        """Address-hash + λ₂ on the flat (batch * seq, d_model) view."""
        if self.training:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x.detach().cpu().numpy()
        bs, sl, _ = x_np.shape
        addresses = []
        for b in range(bs):
            for s in range(sl):
                addresses.append(address_from_activation(
                    x_np[b, s], n_bits=self.n_bits,
                ))
        if self.approximate:
            return hamming_graph_lambda2_power_iter(
                addresses, radius=self.radius, max_nodes=self.max_nodes,
            )
        # Fall back to the (slower) exact eigvalsh from the framework
        from systrophe.catchers.novelty_catcher import lambda_2_of_hamming_graph
        addrs = addresses[: self.max_nodes]
        return float(lambda_2_of_hamming_graph(addrs, radius=self.radius))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Catcher signal (computed on detached activations)
        lambda2 = self._compute_lambda2(x)
        self.lambda_history.append(float(lambda2))
        if len(self.lambda_history) > 200:
            self.lambda_history = self.lambda_history[-200:]
        self.last_lambda2 = float(lambda2)
        self.last_derivative = derivative_catcher(
            self.lambda_history, window=self.history_window,
        )

        ffn_out = self.ffn(x)
        modulation = 1.0 + self.lambda_scale * float(lambda2)
        ffn_out = ffn_out * modulation
        x = self.norm2(x + ffn_out)
        return x
