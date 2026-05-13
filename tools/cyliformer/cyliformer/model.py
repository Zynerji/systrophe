"""Cyliformer language model: stack of CylinderBlocks + LM head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .block import CylinderBlock, CylinderBlockConfig


@dataclass
class CyliformerConfig:
    vocab_size: int = 10
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_cylinders: int = 4
    ffn_mult: int = 4
    max_seq_len: int = 64
    dropout: float = 0.0
    lambda_target: float = 0.20
    catcher_n_bits: int = 24
    catcher_radius: int = 6
    catcher_max_nodes: int = 96
    catcher_power_iter: int = 6
    tie_embeddings: bool = True


class Cyliformer(nn.Module):
    """Resonant Cylinder Transformer LM."""

    def __init__(self, config: CyliformerConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = CyliformerConfig(**kwargs)
        elif kwargs:
            raise TypeError("provide either config or kwargs, not both")
        self.cfg = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        block_cfg = CylinderBlockConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_cylinders=config.n_cylinders,
            ffn_mult=config.ffn_mult,
            dropout=config.dropout,
            lambda_target=config.lambda_target,
            catcher_n_bits=config.catcher_n_bits,
            catcher_radius=config.catcher_radius,
            catcher_max_nodes=config.catcher_max_nodes,
            catcher_power_iter=config.catcher_power_iter,
        )
        self.blocks = nn.ModuleList([
            CylinderBlock(block_cfg) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)

        if config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.lm_head.weight = self.embed.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        infer_prune: bool = False,
    ) -> torch.Tensor:
        """input_ids: (batch, seq). Returns logits (batch, seq, vocab)."""
        bs, sl = input_ids.shape
        if sl > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {sl} > max_seq_len {self.cfg.max_seq_len}"
            )
        positions = torch.arange(sl, device=input_ids.device).unsqueeze(0)
        h = self.drop(self.embed(input_ids) + self.pos(positions))
        for block in self.blocks:
            h = block(h, attn_mask=attn_mask,
                       key_padding_mask=key_padding_mask,
                       infer_prune=infer_prune)
        h = self.ln_f(h)
        return self.lm_head(h)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def all_lambdas(self) -> list[list[float]]:
        return [list(b.last_lambda2_per_cylinder) for b in self.blocks]

    def all_backreactions(self) -> list[list[float]]:
        return [list(b.last_backreaction_per_cylinder) for b in self.blocks]

    def beam_gains(self) -> list[float]:
        return [b.beam_gain() for b in self.blocks]

    # ------------------------------------------------------------------
    # Parameter accounting
    # ------------------------------------------------------------------
    def param_count_breakdown(self) -> dict:
        """Decompose total params into shared FFN vs other vs catcher vs phasor."""
        total = 0
        ffn = 0
        attn = 0
        catcher = 0
        phasor = 0
        embedding = 0
        other = 0
        seen_ids = set()

        def _add(p, key):
            nonlocal total
            if id(p) in seen_ids:
                return
            seen_ids.add(id(p))
            n = int(p.numel())
            total += n
            return n

        for name, p in self.named_parameters():
            n = _add(p, name)
            if n is None:
                continue
            if "shared_ffn" in name:
                ffn += n
            elif "attn" in name:
                attn += n
            elif "catcher" in name:
                catcher += n
            elif "phasors" in name or "backreaction_scale" in name:
                phasor += n
            elif "embed" in name or "pos" in name or "lm_head" in name:
                embedding += n
            else:
                other += n

        return {
            "total": total,
            "shared_ffn": ffn,
            "attention": attn,
            "catcher": catcher,
            "phasor": phasor,
            "embedding_and_lm_head": embedding,
            "other": other,
            "n_layers": self.cfg.n_layers,
            "n_cylinders": self.cfg.n_cylinders,
            "ffn_share_factor": float(self.cfg.n_cylinders),
            "note": (
                "FFN weights are shared across all n_cylinders cylinders "
                "per block. A vanilla transformer with the same per-layer "
                f"capacity but separate FFNs would have {self.cfg.n_cylinders}x "
                "the FFN parameters."
            ),
        }
