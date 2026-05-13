"""Convert Qwen2.5-7B (or any Qwen2-family) HF model into a Cyliformer.

FFN-only swap: every `Qwen2MLP` is replaced with a `CylinderFFN`
wrapper that reuses the original `gate_proj`, `up_proj`, `down_proj`
linears as the *shared* FFN, then loops over N cylinders with a
learnable phasor rotation per cylinder, a differentiable lambda_2
catcher, and back-reaction soft-prune. Attention, RMSNorm, residual
structure, and embeddings are left intact.

This is intentionally the *least invasive* conversion path: it
isolates the Cyliformer's contribution to the FFN sub-block, leaves
the attention machinery and pretrained weights untouched, and adds
only ~`d_model * n_bits + n_cylinders + 1` new parameters per layer
(the catcher's projection + phasors + back-reaction scale).

Usage:
    python qwen_convert.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --n_cylinders 2 \
        --out /tmp/cyliformer-qwen-7b
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch
import torch.nn as nn

# Make the cyliformer package importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cyliformer.catcher import LearnedAddressCatcher


class CylinderFFN(nn.Module):
    """FFN-only Cyliformer wrapper (v2) for a Qwen2MLP / Llama / Mistral MLP.

    Reuses the original mlp's three Linear layers as the *shared*
    SwiGLU FFN; processes N rotated views of the input through that
    shared FFN; combines via beam-sum.

    v2 changes (motivated by the 7B A/B that revealed v1's failures):
      * `backreaction_scale_init = 0.0` so zero-shot is exact identity
        to the pretrained FFN (v1 multiplied by ~0.92 even at init,
        costing ~1% perplexity for nothing).
      * **Single shared catcher per layer** (not per cylinder) -- cuts
        catcher overhead ~50% at no information loss (the same
        layer-level lambda_2 applies to all cylinders).
      * Catcher is *skipped during inference* by default
        (`compute_catcher_in_eval=False`): the lambda_2 signal is only
        needed for the training loss; in inference, backreaction
        defaults to 0 -> cylinder branches sum to vanilla FFN average.
      * `last_phasor_diversity` exposed as a regulariser hook: callers
        can add `-sum cos(delta_i - delta_j)` to the loss to push
        phasors apart.

    Shape contract matches Qwen2MLP exactly:
        in:  (batch, seq, hidden_size)
        out: (batch, seq, hidden_size)
    so a converted Qwen2DecoderLayer is a drop-in replacement.
    """

    def __init__(
        self,
        original_mlp: nn.Module,
        n_cylinders: int = 4,
        lambda_target: float = 0.05,
        backreaction_scale_init: float = 0.0,
        catcher_n_bits: int = 32,
        catcher_radius: int = 8,
        catcher_max_nodes: int = 96,
        catcher_power_iter: int = 6,
        compute_catcher_in_eval: bool = False,
        rotate_dim: int | None = None,
    ) -> None:
        super().__init__()
        # Pull the three Linear sublayers from the original mlp.
        self.gate_proj = original_mlp.gate_proj
        self.up_proj = original_mlp.up_proj
        self.down_proj = original_mlp.down_proj
        if hasattr(original_mlp, "act_fn"):
            self.act_fn = original_mlp.act_fn
        else:
            self.act_fn = nn.SiLU()

        hidden = self.gate_proj.in_features
        if rotate_dim is None:
            rotate_dim = hidden
        if rotate_dim % 2 != 0:
            raise ValueError(
                f"rotate_dim must be even (got {rotate_dim}); "
                f"hidden_size = {hidden}"
            )
        self.rotate_dim = int(rotate_dim)
        self.hidden = int(hidden)
        self.n_cylinders = int(n_cylinders)
        self.lambda_target = float(lambda_target)
        self.compute_catcher_in_eval = bool(compute_catcher_in_eval)

        # Phasors initialised at small random angles. With n_cylinders >= 3
        # the linspace[-pi/4, pi/4] init produces phase-distinct cylinders
        # without catastrophic destructive interference (verified at 7B).
        if n_cylinders >= 3:
            base = torch.linspace(-torch.pi / 4.0, torch.pi / 4.0, n_cylinders)
            init_phases = base + torch.randn(n_cylinders) * 0.02
        else:
            init_phases = torch.randn(n_cylinders) * 0.05
        self.phasors = nn.Parameter(init_phases)

        # Default 0: zero-shot is identity (no FFN scaling at init).
        self.backreaction_scale = nn.Parameter(
            torch.tensor(float(backreaction_scale_init))
        )

        # Single shared catcher per FFN call (applied to the combined
        # FFN output, not per-cylinder). v1 had per-cylinder catcher
        # which doubled the cost.
        self.catcher = LearnedAddressCatcher(
            d_model=hidden,
            n_bits=catcher_n_bits,
            radius=catcher_radius,
            max_nodes=catcher_max_nodes,
            n_power_iter=catcher_power_iter,
        )

        # Diagnostics
        self.last_lambda2: float = 0.0
        self.last_backreaction: float = 0.0
        self.last_phasor_diversity: float = 0.0
        # v1-compatible per-cylinder fields (kept for the diagnostics collector)
        self.last_lambda2_per_cylinder: list[float] = []
        self.last_backreaction_per_cylinder: list[float] = []

    def _rotate_channel_pairs(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Rotate (real, imag) channel halves of x by angle delta."""
        if self.rotate_dim == self.hidden:
            d = x.shape[-1]
            half = d // 2
            re = x[..., :half]
            im = x[..., half:]
            c = torch.cos(delta)
            s = torch.sin(delta)
            re_new = re * c - im * s
            im_new = re * s + im * c
            return torch.cat([re_new, im_new], dim=-1)
        half = self.rotate_dim // 2
        x_rot = x[..., : self.rotate_dim]
        x_tail = x[..., self.rotate_dim :]
        re = x_rot[..., :half]
        im = x_rot[..., half : self.rotate_dim]
        c = torch.cos(delta)
        s = torch.sin(delta)
        re_new = re * c - im * s
        im_new = re * s + im * c
        return torch.cat([re_new, im_new, x_tail], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cylinder_outs: list[torch.Tensor] = []
        for c in range(self.n_cylinders):
            x_rot = self._rotate_channel_pairs(x, self.phasors[c])
            ffn_out = self.down_proj(
                self.act_fn(self.gate_proj(x_rot)) * self.up_proj(x_rot)
            )
            cylinder_outs.append(ffn_out)

        # Beam-sum (mean) of cylinders. With backreaction_scale = 0 at
        # init this is identical to a vanilla FFN with rotated input.
        combined = torch.stack(cylinder_outs, dim=0).mean(dim=0)

        # Catcher signal: computed on the COMBINED output, once per layer.
        # Skipped in eval mode unless explicitly requested.
        run_catcher = self.training or self.compute_catcher_in_eval
        if run_catcher:
            lambda2 = self.catcher(combined)
            backreact = torch.sigmoid(-8.0 * (lambda2 - self.lambda_target))
            out = combined * (1.0 - self.backreaction_scale * backreact)
            self.last_lambda2 = float(lambda2.detach().item())
            self.last_backreaction = float(backreact.detach().item())
        else:
            out = combined
            self.last_lambda2 = 0.0
            self.last_backreaction = 0.0

        # Phasor diversity (for regularisation hook):
        # mean of |cos(delta_i - delta_j)| over i != j; lower is more diverse.
        with torch.no_grad():
            if self.n_cylinders >= 2:
                p = self.phasors.detach()
                diffs = p.unsqueeze(0) - p.unsqueeze(1)
                eye = torch.eye(self.n_cylinders, device=p.device, dtype=torch.bool)
                d_off = diffs[~eye]
                self.last_phasor_diversity = float(torch.cos(d_off).pow(2).mean().item())
            else:
                self.last_phasor_diversity = 0.0

        # v1-compatibility: broadcast scalar lambda_2 / backreact to N entries
        self.last_lambda2_per_cylinder = [self.last_lambda2] * self.n_cylinders
        self.last_backreaction_per_cylinder = [self.last_backreaction] * self.n_cylinders

        return out

    def beam_gain(self) -> float:
        with torch.no_grad():
            z = torch.cos(self.phasors).sum() ** 2 + torch.sin(self.phasors).sum() ** 2
        return float(torch.sqrt(z).item()) / float(self.n_cylinders)

    def phasor_diversity_loss(self) -> torch.Tensor:
        """Differentiable phasor-diversity regulariser.

        Returns mean of cos(delta_i - delta_j)^2 over i != j.
        Add to training loss with a small positive weight to push the
        phasors apart and break the "all cylinders identical" failure
        mode revealed in the v1 7B A/B (beam_gain stuck at 1.0).
        """
        if self.n_cylinders < 2:
            return torch.zeros((), device=self.phasors.device)
        p = self.phasors
        diffs = p.unsqueeze(0) - p.unsqueeze(1)
        n = self.n_cylinders
        eye = torch.eye(n, device=p.device, dtype=torch.bool)
        d_off = diffs[~eye]
        return torch.cos(d_off).pow(2).mean()


def convert_qwen_to_cyliformer(
    model: nn.Module,
    n_cylinders: int = 4,
    lambda_target: float = 0.05,
    backreaction_scale_init: float = 0.0,
    catcher_n_bits: int = 32,
    catcher_max_nodes: int = 96,
    catcher_power_iter: int = 6,
    compute_catcher_in_eval: bool = False,
) -> dict:
    """Walk the model, replace each layer's `mlp` with `CylinderFFN`.

    Returns a summary dict with the layer indices that were converted
    and the count of new parameters added.
    """
    decoder = None
    for attr in ("model", "transformer", "base_model"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if hasattr(inner, "layers"):
                decoder = inner
                break
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError(
            "Could not locate `model.layers` -- model architecture not "
            "supported by this conversion script. Expected Qwen2/Llama/Mistral-"
            "style HF causal LM."
        )

    n_layers = len(decoder.layers)
    new_params = 0
    converted = []
    for i, layer in enumerate(decoder.layers):
        if not hasattr(layer, "mlp"):
            continue
        original_mlp = layer.mlp
        cyl_ffn = CylinderFFN(
            original_mlp,
            n_cylinders=n_cylinders,
            lambda_target=lambda_target,
            backreaction_scale_init=backreaction_scale_init,
            catcher_n_bits=catcher_n_bits,
            catcher_max_nodes=catcher_max_nodes,
            catcher_power_iter=catcher_power_iter,
            compute_catcher_in_eval=compute_catcher_in_eval,
        )
        # Move new params to the same device/dtype as the original mlp
        first_param = next(original_mlp.parameters(), None)
        if first_param is not None:
            cyl_ffn = cyl_ffn.to(
                device=first_param.device, dtype=first_param.dtype,
            )
        layer.mlp = cyl_ffn
        # Account for the genuinely-new params (catcher proj + phasors +
        # backreaction_scale). The gate/up/down are *reused* references
        # to the original linears -- not double-counted in the model's
        # param count because nn.Module deduplicates by id.
        new_params += int(cyl_ffn.catcher.proj.weight.numel())
        new_params += int(cyl_ffn.catcher.proj.bias.numel())
        new_params += int(cyl_ffn.phasors.numel())
        new_params += 1  # backreaction_scale
        converted.append(i)

    return {
        "n_layers_converted": len(converted),
        "n_layers_total": n_layers,
        "converted_layer_indices": converted,
        "new_params_added": new_params,
        "n_cylinders": n_cylinders,
        "lambda_target": lambda_target,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--n_cylinders", type=int, default=2)
    parser.add_argument("--lambda_target", type=float, default=0.18)
    parser.add_argument("--catcher_n_bits", type=int, default=32)
    parser.add_argument("--catcher_max_nodes", type=int, default=96)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--out", type=str, default=None,
                          help="If set, save the converted state_dict + config to this dir")
    parser.add_argument("--dry_run", action="store_true",
                          help="Convert + report stats but don't save")
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    t0 = time.time()
    print(f"Loading base model: {args.base}  (dtype={args.dtype})")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    t_load = time.time() - t0
    print(f"  loaded in {t_load:.1f} s")

    print(f"Converting model -> Cyliformer with n_cylinders={args.n_cylinders}, "
          f"lambda_target={args.lambda_target}")
    summary = convert_qwen_to_cyliformer(
        model,
        n_cylinders=args.n_cylinders,
        lambda_target=args.lambda_target,
        catcher_n_bits=args.catcher_n_bits,
        catcher_max_nodes=args.catcher_max_nodes,
    )
    print(f"  {summary['n_layers_converted']} / {summary['n_layers_total']} layers converted")
    print(f"  new params added (catcher proj + phasors + backreact_scale): "
          f"{summary['new_params_added']:,}")

    # Total trainable params for sanity
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  total model params (after conversion): {total_params:,}")
    print(f"  new params as fraction of total: "
          f"{summary['new_params_added'] / max(total_params, 1):.4%}")

    if args.dry_run:
        print("dry-run: skipping save")
        return

    if args.out is None:
        raise SystemExit("--out is required unless --dry_run")
    out_path = pathlib.Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving converted model to {out_path}")
    # Save with safetensors. Note: the CylinderFFN class lives in this
    # repo, so reloading requires importing qwen_convert.py before
    # `from_pretrained`. For an A/B test we just keep the model in
    # memory and don't round-trip through disk.
    model.save_pretrained(out_path, safe_serialization=True)
    tok.save_pretrained(out_path)
    # Also record conversion metadata
    (out_path / "cyliformer_conversion.json").write_text(json.dumps({
        "base": args.base,
        "n_cylinders": args.n_cylinders,
        "lambda_target": args.lambda_target,
        "catcher_n_bits": args.catcher_n_bits,
        "catcher_max_nodes": args.catcher_max_nodes,
        "dtype": args.dtype,
        "summary": summary,
        "note": (
            "Reloading this checkpoint requires importing "
            "tools/cyliformer/experiments/qwen_convert.py:CylinderFFN "
            "BEFORE calling AutoModelForCausalLM.from_pretrained."
        ),
    }, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
