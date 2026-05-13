"""CylinderBlock: the heart of Cyliformer.

Each block:
  1. Pre-norm + multi-head attention (standard).
  2. Pre-norm.
  3. N "cylinders": each rotates the input via a 2D channel-pair rotation
     by the cylinder's learnable phasor, then passes through a *shared* FFN.
  4. Per-cylinder lambda_2 catcher signal.
  5. Back-reaction proxy: large at low lambda_2 (= incoherent, "unstable").
  6. Soft prune via (1 - backreaction_scale * backreact); hard prune
     during inference if backreact > prune_threshold.
  7. Beam-sum: average over surviving cylinders, residual add.

Parameter count:
  Shared FFN: d_model * (2 * d_model * ffn_mult + d_model * ffn_mult) for the
              two Linear layers (same as a vanilla block).
  Phasors:   n_cylinders learnable scalars (negligible).
  Catcher:   d_model * n_bits + n_bits projection weights.

So the per-block parameter count is **roughly the same** as a vanilla
transformer block; what changes is the *effective representational
capacity* (N rotated views processed by the same FFN, with coherence
gating) and the activation sparsity (soft + hard cylinder pruning).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .catcher import LearnedAddressCatcher


@dataclass
class CylinderBlockConfig:
    d_model: int = 64
    n_heads: int = 4
    n_cylinders: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0
    lambda_target: float = 0.20
    backreaction_scale_init: float = 0.15
    prune_threshold: float = 0.85
    catcher_n_bits: int = 24
    catcher_radius: int = 6
    catcher_max_nodes: int = 96
    catcher_power_iter: int = 6


class CylinderBlock(nn.Module):
    """Resonant-cylinder transformer block with shared FFN + lambda_2 gating."""

    def __init__(self, config: CylinderBlockConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = CylinderBlockConfig(**kwargs)
        elif kwargs:
            raise TypeError("provide either config or kwargs, not both")
        self.cfg = config

        if self.cfg.d_model % 2 != 0:
            raise ValueError("d_model must be even (channels split into real/imag pairs)")

        self.ln1 = nn.LayerNorm(self.cfg.d_model)
        self.attn = nn.MultiheadAttention(
            self.cfg.d_model, self.cfg.n_heads,
            dropout=self.cfg.dropout, batch_first=True,
        )

        self.ln2 = nn.LayerNorm(self.cfg.d_model)

        # Shared FFN: same weights used by every cylinder.
        self.shared_ffn = nn.Sequential(
            nn.Linear(self.cfg.d_model, self.cfg.d_model * self.cfg.ffn_mult),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.d_model * self.cfg.ffn_mult, self.cfg.d_model),
            nn.Dropout(self.cfg.dropout),
        )

        # Cylinder phasors: learnable angles, one per cylinder.
        # Init: small random perturbations near 0 so the block starts
        # close to a no-op rotation (beam_gain ~ 1 at init -- preserves
        # any pretrained behaviour of a shared FFN) and gradients break
        # the symmetry during training.
        #
        # NOTE: linspace[-pi/2, pi/2] is degenerate at n=2: the two
        # phases give +-90 deg channel rotations that CANCEL under
        # averaging (beam_gain ~ 0, signal destroyed). Verified against
        # Qwen2.5-7B FFN-only conversion 2026-05-13.
        init_phases = torch.randn(self.cfg.n_cylinders) * 0.05
        self.phasors = nn.Parameter(init_phases)

        self.backreaction_scale = nn.Parameter(
            torch.tensor(float(self.cfg.backreaction_scale_init)),
        )

        self.catcher = LearnedAddressCatcher(
            d_model=self.cfg.d_model,
            n_bits=self.cfg.catcher_n_bits,
            radius=self.cfg.catcher_radius,
            max_nodes=self.cfg.catcher_max_nodes,
            n_power_iter=self.cfg.catcher_power_iter,
        )

        # Per-forward diagnostics (not persisted state)
        self.last_lambda2_per_cylinder: list[float] = []
        self.last_backreaction_per_cylinder: list[float] = []
        self.last_pruned_mask: list[bool] = []

    # ------------------------------------------------------------------
    # Phasor rotation
    # ------------------------------------------------------------------
    @staticmethod
    def _rotate_channel_pairs(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Rotate (real, imag) channel halves of x by angle delta.

        Splits the last dim into two halves; treats first half as real,
        second as imag; applies a 2D rotation by delta to every (r, i)
        pair. delta is a scalar tensor (cylinder phase).
        """
        d = x.shape[-1]
        half = d // 2
        re = x[..., :half]
        im = x[..., half:]
        c = torch.cos(delta)
        s = torch.sin(delta)
        re_new = re * c - im * s
        im_new = re * s + im * c
        return torch.cat([re_new, im_new], dim=-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        infer_prune: bool = False,
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        residual = x
        x_n = self.ln1(x)
        attn_out, _ = self.attn(
            x_n, x_n, x_n,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + attn_out

        # Pre-norm FFN block
        residual = x
        x_n = self.ln2(x)

        cylinder_outs = []
        lambda2_diag: list[float] = []
        backreact_diag: list[float] = []
        pruned_diag: list[bool] = []

        for c in range(self.cfg.n_cylinders):
            x_rot = self._rotate_channel_pairs(x_n, self.phasors[c])
            ffn_out = self.shared_ffn(x_rot)

            lambda2 = self.catcher(ffn_out)
            # High lambda_2 -> low back-reaction; low lambda_2 -> high back-reaction.
            backreact = torch.sigmoid(
                -8.0 * (lambda2 - float(self.cfg.lambda_target))
            )

            lambda2_diag.append(float(lambda2.detach().item()))
            backreact_diag.append(float(backreact.detach().item()))

            pruned = False
            if infer_prune and not self.training:
                # Hard prune: skip this cylinder entirely
                if float(backreact.item()) > float(self.cfg.prune_threshold):
                    pruned = True
            pruned_diag.append(pruned)

            if pruned:
                continue
            cylinder_outs.append(
                ffn_out * (1.0 - self.backreaction_scale * backreact)
            )

        # Beam-sum: equally-weighted (matched amplitudes) average over
        # surviving cylinders.
        if len(cylinder_outs) == 0:
            combined = torch.zeros_like(x_n)
        else:
            combined = torch.stack(cylinder_outs, dim=0).mean(dim=0)

        self.last_lambda2_per_cylinder = lambda2_diag
        self.last_backreaction_per_cylinder = backreact_diag
        self.last_pruned_mask = pruned_diag

        return residual + combined

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def beam_gain(self) -> float:
        """Magnitude of the cylinder-phasor sum (Phase 3a array factor).

        |sum_c exp(i * delta_c)| / n_cylinders. 1.0 when all phasors
        align (constructive beam), 0.0 at uniform-phase comb extinction.
        """
        with torch.no_grad():
            z = torch.cos(self.phasors).sum() ** 2 + torch.sin(self.phasors).sum() ** 2
        return float(torch.sqrt(z).item()) / float(self.cfg.n_cylinders)
