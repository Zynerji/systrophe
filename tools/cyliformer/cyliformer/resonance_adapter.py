"""ResonanceAdapter: Dianoia-pattern additive block gated by the Systrophe λ₂ catcher.

Lineage:
  * **Engineering** from `C:/Users/cknop/.local/bin/Dianoia/dianoia/`:
    - additive residual insertion via `augment_model` after each transformer
      block (preserves pretrained behaviour exactly at init via tiny
      back-projection weights);
    - matched-parameter MLP-adapter control for honest A/B.

  * **Compute primitive** REMOVED: the wave basis (cos/sin/oscillator)
    has been falsified three times in this lineage --
    qGPT-Infinity (KL wall), Overtone (SVD dominance), Dianoia (no
    widening-gap on parity/add-chain). The previous Cyliformer drafts
    (v1-v3) used phasor channel-rotation = wave basis on channels;
    they are iteration #4 of the same falsified pattern and not
    competitive with a vanilla LoRA at 7B scale.

  * **What survives**: the Systrophe **address-space λ₂ catcher**
    (`systrophe.catchers.novelty_catcher.lambda_2_of_hamming_graph` -- the
    only unfalsified Systrophe-derived LLM primitive). It enters
    here as a per-block gating signal for a SMALL near-identity-init
    adapter -- *not* as a routing signal for parallel cylinder
    paths through a shared FFN.

The block is:

    h_out = h + g(λ₂(h)) * down_proj(silu(up_proj(h)))

where:
  * `up_proj : d_model -> r` (small bottleneck dimension r, default 32)
  * `down_proj : r -> d_model` initialised at near zero so the
    residual is ~ 0 at init (the model is unchanged at conversion time)
  * `λ₂(h)` is the differentiable address-space lambda_2 from
    `cyliformer.catcher.LearnedAddressCatcher`
  * `g(·) = sigmoid(α (λ₂ - target))` is a learnable scalar gate that
    grows the adapter contribution only when the catcher signals
    coherence above `target`. `α` and `target` are learnable scalars,
    `α` initialised at 0 (so g ≈ 0.5 -> contribution still ~0 via the
    zero-init down_proj) and `target` initialised at 0.

ResonanceAdapter has ~ `2 * d_model * r + r + d_model + n_bits * d_model`
parameters. With r=32, n_bits=32, d_model=3584 (Qwen2.5-7B), that's
~ 360k per block, ~ 10M across 28 blocks -- comparable to a small
LoRA adapter.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .catcher import LearnedAddressCatcher


class ResonanceAdapter(nn.Module):
    """Tiny additive residual adapter, λ₂-gated. Inserted after each layer.

    Args:
        d_model: surrounding transformer hidden size.
        r: bottleneck width inside the adapter. Smaller r -> fewer params,
           lower capacity.
        n_bits: catcher address width.
        catcher_radius: Hamming-graph edge threshold.
        catcher_max_nodes: cap for catcher subsampling.
        catcher_power_iter: power-iteration depth.
        init_gain: small init scale for the back-projection so the
                   residual is essentially zero at construction.
        compute_catcher_in_eval: whether to compute λ₂ at inference. With
                   False the gate is whatever it learns to be, and the
                   per-token cost is one small MLP + one constant gate.
    """

    def __init__(
        self,
        d_model: int,
        r: int = 32,
        n_bits: int = 32,
        catcher_radius: int = 8,
        catcher_max_nodes: int = 96,
        catcher_power_iter: int = 6,
        init_gain: float = 0.02,
        compute_catcher_in_eval: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.r = int(r)
        self.compute_catcher_in_eval = bool(compute_catcher_in_eval)

        # Small bottleneck adapter
        self.up_proj = nn.Linear(d_model, r)
        self.down_proj = nn.Linear(r, d_model)
        # Near-identity at init: tiny down_proj
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=init_gain / math.sqrt(r))
        nn.init.zeros_(self.down_proj.bias)

        # Catcher on the bottleneck activations (smaller dim -> cheaper)
        self.catcher = LearnedAddressCatcher(
            d_model=r,
            n_bits=n_bits,
            radius=catcher_radius,
            max_nodes=catcher_max_nodes,
            n_power_iter=catcher_power_iter,
        )

        # Learnable gate parameters (alpha=0 at init -> g = sigmoid(0) = 0.5)
        self.gate_alpha = nn.Parameter(torch.tensor(0.0))
        self.gate_target = nn.Parameter(torch.tensor(0.0))

        # Cached forward diagnostics
        self.last_lambda2: float = 0.0
        self.last_gate: float = 0.5

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        bottleneck = F.silu(self.up_proj(h))   # (..., r)
        run_catcher = self.training or self.compute_catcher_in_eval
        if run_catcher:
            lambda2 = self.catcher(bottleneck)
            gate = torch.sigmoid(self.gate_alpha * (lambda2 - self.gate_target))
            self.last_lambda2 = float(lambda2.detach().item())
            self.last_gate = float(gate.detach().item())
        else:
            # Use a fixed gate value from the last forward; the gate is
            # a global scalar so reusing the cached value is reasonable.
            gate = torch.tensor(self.last_gate, device=h.device, dtype=h.dtype)
        delta = self.down_proj(bottleneck) * gate
        return h + delta


def matched_mlp_adapter(d_model: int, target_params: int,
                          init_gain: float = 0.02) -> nn.Module:
    """Dense d -> h -> d MLP sized to match `target_params` parameters.

    Used as the apples-to-apples control for ResonanceAdapter. Same
    insertion pattern, same near-identity init, no catcher.
    """
    # d -> h -> d:  fc1: d*h + h params,  fc2: h*d + d params
    # total = 2*d*h + h + d
    # so h = (target - d) / (2d + 1)
    h = max(1, round((target_params - d_model) / (2 * d_model + 1)))

    class MLPAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d_model, h)
            self.fc2 = nn.Linear(h, d_model)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=init_gain / math.sqrt(h))
            nn.init.zeros_(self.fc2.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.fc2(F.silu(self.fc1(x)))

    return MLPAdapter()


# ---------------------------------------------------------------------------
# Insertion utility (port of Dianoia's augment_model, adapted for arbitrary
# adapter factories).
# ---------------------------------------------------------------------------


class _BlockWithAdapter(nn.Module):
    """Wrap an existing transformer layer so its output runs through `adapter`."""

    def __init__(self, inner: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            h = self.adapter(out[0])
            return (h,) + out[1:]
        return self.adapter(out)


def _find_layer_list(model: nn.Module) -> tuple[nn.ModuleList, str]:
    candidates = [
        "transformer.h",
        "model.layers",
        "gpt_neox.layers",
        "encoder.layer",
    ]
    for dotted in candidates:
        parts = dotted.split(".")
        obj = model
        for p in parts:
            if not hasattr(obj, p):
                obj = None
                break
            obj = getattr(obj, p)
        if isinstance(obj, nn.ModuleList):
            return obj, dotted
    raise SystemExit(
        "Could not autodetect transformer layer list. "
        "Pass `layers_attr` explicitly."
    )


def augment_with_adapter(
    model: nn.Module,
    adapter_factory,
    layers_attr: str | None = None,
    every: int = 1,
) -> dict:
    """Insert `adapter_factory(layer_index)` after each (every-th) layer.

    Returns a summary dict with the layer indices inserted and the count
    of newly-added parameters. The original transformer layers are not
    modified -- only wrapped.
    """
    if layers_attr is None:
        layer_list, layers_attr = _find_layer_list(model)
    else:
        obj = model
        for p in layers_attr.split("."):
            obj = getattr(obj, p)
        layer_list = obj

    sample_param = next(model.parameters())
    dtype = sample_param.dtype
    device = sample_param.device

    inserted = []
    added = 0
    for i in range(len(layer_list)):
        if (i + 1) % every != 0:
            continue
        adapter = adapter_factory(i).to(device=device, dtype=dtype)
        wrapped = _BlockWithAdapter(inner=layer_list[i], adapter=adapter)
        layer_list[i] = wrapped
        inserted.append(i)
        added += sum(int(p.numel()) for p in adapter.parameters())

    return {
        "layers_attr": layers_attr,
        "inserted_indices": inserted,
        "n_inserted": len(inserted),
        "n_layers_total": len(layer_list),
        "added_params": int(added),
    }
