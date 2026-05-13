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

        # Bottleneck up-projection. SMALL init so the bottleneck activation
        # x = up_proj(h) stays in a bounded BF16-safe range even when the
        # surrounding residual stream is large (Qwen2.5-7B post-block
        # residuals can be on the order of ±50 in BF16).
        self.up_proj = nn.Linear(d_model, state_dim)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02 / math.sqrt(d_model))
        nn.init.zeros_(self.up_proj.bias)

        # Δ_t = softplus(dt_proj(x_t)); input-dependent step size.
        # Default nn.Linear init gives weights ~ ±sqrt(1/state_dim) which,
        # combined with a bottleneck activation of magnitude O(1), would
        # produce Δ on the order of softplus(state_dim * weight * x) which
        # is unbounded. We init dt_proj with tiny weights and a small
        # bias so Δ ≈ softplus(delta_bias_init) at construction.
        if dt_rank is None:
            self.dt_proj = nn.Linear(state_dim, state_dim)
            nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.dt_proj.bias, float(delta_bias_init))
        else:
            self.dt_rank = int(dt_rank)
            self.dt_low = nn.Linear(state_dim, dt_rank)
            self.dt_up = nn.Linear(dt_rank, state_dim)
            self.dt_proj = None  # marker
            nn.init.normal_(self.dt_low.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.dt_up.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.dt_low.bias)
            nn.init.constant_(self.dt_up.bias, float(delta_bias_init))

        # Input-dependent B and C projections. Small init for the same
        # reason: keep state magnitude bounded at construction.
        self.B_proj = nn.Linear(state_dim, state_dim)
        self.C_proj = nn.Linear(state_dim, state_dim)
        nn.init.normal_(self.B_proj.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.B_proj.bias)
        nn.init.normal_(self.C_proj.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.C_proj.bias)

        # Static negative diagonal A: A_i = -softplus(log_A_i). Init so
        # A spans a reasonable range of timescales (Mamba uses HiPPO init;
        # we use a simpler log-spaced fallback).
        #
        # FROZEN by default. Reason: in the 7B + LoRA setting we
        # observed gradient explosion through the cumulative product
        # of A_bar = exp(Δ*A) over T=512 timesteps; even modest LoRA
        # learning rates (2e-4) drive log_A NaN within 25 optimizer
        # steps. Mamba's official kernels train log_A with a much
        # smaller LR and a custom init, neither of which we replicate
        # here. Freezing log_A leaves the *selective* part of the SSM
        # (input-dependent Δ, B, C) trainable -- the part Dianoia's
        # FINDINGS specifically pointed to.
        log_A_init = torch.linspace(
            math.log(0.1), math.log(1.0), state_dim,
        )
        self.log_A = nn.Parameter(log_A_init, requires_grad=False)

        # Skip-connection scalar D (Mamba's D term).
        # Also frozen by default for the same gradient-stability reason
        # (D's gradient is unconditioned by the recurrence so it's less
        # explosive, but easier to keep the policy uniform).
        self.D = nn.Parameter(torch.zeros(state_dim), requires_grad=False)

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
        """h: (B, T, d_model). Returns same shape.

        The SSM scan is done in fp32 internally for numerical stability;
        the input and output dtypes are preserved (BF16 in -> BF16 out).
        Sequential recurrence + BF16 + gradient checkpointing was found
        to produce NaN gradients within 25 LoRA steps; promoting the
        scan to fp32 stabilises it. The cost is roughly 2x activation
        memory for the scan -- still small at state_dim=32.
        """
        in_dtype = h.dtype
        # 1. Project to bottleneck (in whatever dtype the model uses).
        x = self.up_proj(h)                       # (B, T, state_dim)

        # 2. Promote SSM internals to fp32 for numerical stability.
        x_f = x.float()
        bs, T, sd = x_f.shape

        # 3. Input-dependent Δ, B, C. Clamp Δ <= 10 to prevent runaway
        # under out-of-distribution input.
        delta = self._delta(x).float()            # (B, T, state_dim) > 0
        delta = torch.clamp(delta, max=10.0)
        B = self.B_proj(x).float()                # (B, T, state_dim)
        C = self.C_proj(x).float()                # (B, T, state_dim)

        # 4. Discretise A: A_bar = exp(Δ * A_diag).
        # A < 0 -> Δ*A <= 0 -> exp(Δ*A) in (0, 1].
        A = -F.softplus(self.log_A).float()       # (state_dim,) negative
        exponent = delta * A.unsqueeze(0).unsqueeze(0)
        exponent = torch.clamp(exponent, min=-30.0, max=0.0)
        A_bar = torch.exp(exponent)               # (B, T, sd), in (0, 1]
        # Discretised B: B_bar = Δ * B (zero-order hold simplification)
        B_bar_x = delta * B * x_f                 # (B, T, sd); element-wise

        # 5. Sequential scan in fp32: state[t] = A_bar[t] * state[t-1] + B_bar_x[t]
        state = torch.zeros(bs, sd, device=x_f.device, dtype=torch.float32)
        ys = []
        for t in range(T):
            state = A_bar[:, t, :] * state + B_bar_x[:, t, :]
            y_t = C[:, t, :] * state             # (B, sd)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)               # (B, T, sd)

        # 6. Add D skip (in fp32), cast back to model dtype, project to d_model.
        y = y + self.D.float().unsqueeze(0).unsqueeze(0) * x_f
        y = y.to(in_dtype)
        delta_h = self.out_proj(y)               # (B, T, d_model)

        # Diagnostics
        with torch.no_grad():
            self.last_state_norm = float(state.norm().item() / max(1, bs * sd) ** 0.5)
            self.last_delta_mean = float(delta.mean().item())

        return h + delta_h


def count_params(m: nn.Module) -> int:
    return sum(int(p.numel()) for p in m.parameters())
