"""SelectiveSSMAdapter -- Mamba S6 style state dynamics inside a Dianoia-pattern
additive residual adapter.

Motivation: Dianoia FINDINGS pointed here. Parity / add-chain required
input-dependent state dynamics that the wave basis cannot express;
linear SSMs (cumsum, exp-decay) and the Cyliformer wave-basis lineage
were all falsified for the same mathematical reason. Mamba S6 supplies
the missing primitive: input-dependent A, B (via Δ_t) and C transforms,
so the hidden state at each token is a *selected* function of past
tokens rather than a fixed convolution.

This module ports the minimal S6 form into the Dianoia/ResonanceAdapter
insertion pattern:

    h_state[t] = exp(Δ[t] · A) · h_state[t-1] + Δ[t] · B[t] · x[t]
    y[t]       = C[t] · h_state[t]
    h_out      = h + W_out · y                # residual; W_out near zero

with:

  * Δ[t] = softplus(W_Δ · x[t])               input-dependent step size > 0
  * B[t] = W_B · x[t]                         input-dependent input gate
  * C[t] = W_C · x[t]                         input-dependent output gate
  * A    = -softplus(log_A)                   negative diagonal (stable)

A small bottleneck linear `up_proj` maps the d_model residual stream
into the SSM's state_dim coordinates. After the recurrence, `out_proj`
maps back to d_model with near-zero init so the adapter is identity at
construction (pretrained model unchanged).

This is the same near-identity / matched-MLP-control setup as
ResonanceAdapter (v4); the only thing that changes is the bottleneck
inner block.

Compute: sequential scan in PyTorch. At 7B / seq_len=512 / 28 layers
this is slow (each scan step is a CUDA kernel launch). Parallel scan
optimisations exist (Mamba's official kernel, PyTorch associative_scan)
but are out of scope for this PoC. We pay the throughput cost up
front and let the quality measurement decide whether the optimisation
work is justified.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSMAdapter(nn.Module):
    """Mamba S6 style selective recurrence in a low-dim bottleneck.

    Args:
        d_model: surrounding transformer hidden size.
        state_dim: bottleneck / SSM state dimension.
        init_gain: small init for the back-projection.
        delta_bias_init: initial bias for the softplus(Δ) projection.
                         Positive value -> larger Δ at init -> faster
                         forgetting; 0.0 is a reasonable default.
        dt_rank: rank of the low-rank decomposition for the Δ projection.
                 None = use full state_dim x state_dim Linear (cheap at
                 small state_dim; only matters for large bottlenecks).
        compute_in_eval: True (default). We do NOT skip the SSM in
                         eval -- a Mamba block stripped at inference is
                         a different model.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int = 32,
        init_gain: float = 0.02,
        delta_bias_init: float = 0.0,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.state_dim = int(state_dim)

        # Bottleneck up-projection
        self.up_proj = nn.Linear(d_model, state_dim)

        # Δ_t = softplus(dt_proj(x_t)); input-dependent step size
        if dt_rank is None:
            self.dt_proj = nn.Linear(state_dim, state_dim)
        else:
            self.dt_rank = int(dt_rank)
            self.dt_low = nn.Linear(state_dim, dt_rank)
            self.dt_up = nn.Linear(dt_rank, state_dim)
            self.dt_proj = None  # marker
        # Push Δ to a positive bias so the recurrence is non-trivial at init
        if self.dt_proj is not None:
            nn.init.constant_(self.dt_proj.bias, float(delta_bias_init))

        # Input-dependent B and C projections
        self.B_proj = nn.Linear(state_dim, state_dim)
        self.C_proj = nn.Linear(state_dim, state_dim)

        # Static negative diagonal A: A_i = -softplus(log_A_i). Init so
        # A spans a reasonable range of timescales (Mamba uses HiPPO init;
        # we use a simpler log-spaced fallback).
        log_A_init = torch.linspace(
            math.log(0.1), math.log(1.0), state_dim,
        )
        self.log_A = nn.Parameter(log_A_init)

        # Skip-connection scalar D (Mamba's D term)
        self.D = nn.Parameter(torch.zeros(state_dim))

        # Back-projection: near zero at init -> adapter is identity at start
        self.out_proj = nn.Linear(state_dim, d_model)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=init_gain / math.sqrt(state_dim))
        nn.init.zeros_(self.out_proj.bias)

        # Diagnostics
        self.last_state_norm: float = 0.0
        self.last_delta_mean: float = 0.0

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        if self.dt_proj is not None:
            return F.softplus(self.dt_proj(x))
        else:
            return F.softplus(self.dt_up(self.dt_low(x)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, d_model). Returns same shape."""
        # 1. Project to bottleneck.
        x = self.up_proj(h)                       # (B, T, state_dim)
        bs, T, sd = x.shape

        # 2. Input-dependent Δ, B, C
        delta = self._delta(x)                    # (B, T, state_dim) > 0
        B = self.B_proj(x)                        # (B, T, state_dim)
        C = self.C_proj(x)                        # (B, T, state_dim)

        # 3. Discretise A: A_bar = exp(Δ * A_diag), diagonal so element-wise.
        A = -F.softplus(self.log_A)               # (state_dim,) negative
        A_bar = torch.exp(delta * A.unsqueeze(0).unsqueeze(0))   # (B, T, sd)
        # Discretised B: B_bar = Δ * B (zero-order hold simplification)
        B_bar_x = delta * B * x                   # (B, T, sd); element-wise

        # 4. Sequential scan: state[t] = A_bar[t] * state[t-1] + B_bar_x[t]
        state = torch.zeros(bs, sd, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            state = A_bar[:, t, :] * state + B_bar_x[:, t, :]
            y_t = C[:, t, :] * state             # (B, sd)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)               # (B, T, sd)

        # 5. Add D skip and project back
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        delta_h = self.out_proj(y)               # (B, T, d_model)

        # Diagnostics
        with torch.no_grad():
            self.last_state_norm = float(state.norm().item() / max(1, bs * sd) ** 0.5)
            self.last_delta_mean = float(delta.mean().item())

        return h + delta_h


def count_params(m: nn.Module) -> int:
    return sum(int(p.numel()) for p in m.parameters())
