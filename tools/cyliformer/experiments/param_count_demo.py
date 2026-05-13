"""Parameter-count comparison: Cyliformer vs equivalent dense baseline.

Compares a Cyliformer (with shared FFN across N cylinders) to a
hypothetical dense transformer with N independent FFNs per layer
(the "equivalent capacity" baseline). The shared-FFN savings are
real *vs that baseline* but NOT vs a single-FFN-per-layer vanilla
transformer. This script makes the comparison explicit so readers
can judge for themselves.
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cyliformer import Cyliformer


def count_dense_baseline_ffn(d_model: int, ffn_mult: int, n_layers: int,
                                 n_independent_ffns_per_layer: int) -> int:
    """N independent FFNs per layer (the apples-to-apples capacity baseline)."""
    per_ffn = d_model * d_model * ffn_mult + d_model * ffn_mult * d_model
    # Add biases
    per_ffn += d_model * ffn_mult + d_model
    return per_ffn * n_independent_ffns_per_layer * n_layers


def count_vanilla_transformer_total(
    vocab_size: int, d_model: int, ffn_mult: int, n_layers: int, n_heads: int,
    max_seq_len: int,
) -> dict:
    """Approximate vanilla-transformer parameter count for reference."""
    attn = (4 * d_model * d_model + 4 * d_model) * n_layers  # Qkv + out projections + biases
    ffn = (d_model * d_model * ffn_mult + d_model * ffn_mult * d_model
            + d_model * ffn_mult + d_model) * n_layers
    embed = vocab_size * d_model + max_seq_len * d_model
    lm_head = vocab_size * d_model  # tied weights -> count once
    ln = 2 * d_model * n_layers + d_model  # 2 LayerNorms per block + final ln_f
    return {
        "attention": attn,
        "ffn": ffn,
        "embedding_and_lm_head": embed,
        "layernorms": ln,
        "total": attn + ffn + embed + ln,
    }


def main() -> dict:
    rows = []
    for n_cylinders in (1, 2, 4, 6, 8):
        model = Cyliformer(
            vocab_size=32000, d_model=128, n_layers=4, n_heads=4,
            n_cylinders=n_cylinders, ffn_mult=4, max_seq_len=64,
        )
        breakdown = model.param_count_breakdown()
        vanilla = count_vanilla_transformer_total(
            vocab_size=32000, d_model=128, ffn_mult=4, n_layers=4,
            n_heads=4, max_seq_len=64,
        )
        # N-independent-FFN baseline
        n_ffn_baseline = count_dense_baseline_ffn(
            d_model=128, ffn_mult=4, n_layers=4,
            n_independent_ffns_per_layer=n_cylinders,
        )
        # The Cyliformer's shared FFN is the same as a vanilla transformer's
        # single FFN -- the saving is *vs* the N-independent-FFN baseline.
        rows.append({
            "n_cylinders": n_cylinders,
            "cyliformer_total": breakdown["total"],
            "cyliformer_shared_ffn": breakdown["shared_ffn"],
            "vanilla_transformer_total": vanilla["total"],
            "n_ffn_baseline_ffn_only": n_ffn_baseline,
            "savings_vs_n_ffn_baseline_pct": (
                100.0 * (1.0 - breakdown["shared_ffn"] / n_ffn_baseline)
                if n_ffn_baseline > 0 else 0.0
            ),
        })
        del model

    out = {
        "tool": "cyliformer",
        "task": "param_count_comparison",
        "rows": rows,
        "note": (
            "The Cyliformer's shared FFN is parameterised identically to a "
            "vanilla transformer's single FFN. The 'savings' line is *vs* "
            "an N-independent-FFNs-per-layer baseline -- the architecture "
            "whose effective representational capacity Cyliformer "
            "approximates via N rotated views. Comparing against vanilla "
            "(single-FFN) gives ~0% savings -- the Cyliformer is not "
            "fewer-FFN-params; it is the same FFN params reused for N "
            "rotated computations."
        ),
    }

    out_path = pathlib.Path(__file__).with_name("param_count_demo_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("Cyliformer parameter count: shared-FFN vs N-independent-FFN baseline")
    print("=" * 78)
    print(" n_cyl   cyli_total   cyli_FFN    n_ffn_baseline   savings_vs_baseline")
    for r in rows:
        print(
            f"  {r['n_cylinders']:3d}   {r['cyliformer_total']:>10,d}  "
            f"{r['cyliformer_shared_ffn']:>9,d}   {r['n_ffn_baseline_ffn_only']:>13,d}   "
            f"{r['savings_vs_n_ffn_baseline_pct']:>6.2f}%"
        )
    print()
    print("Note: vs vanilla transformer (single FFN per layer), the parameter")
    print("count is approximately equal. The Cyliformer's value proposition")
    print("is structural (N rotated views, coherence gating, soft pruning),")
    print("NOT raw parameter reduction. Honest about that.")
    print(f"\nResults: {out_path}")
    return out


if __name__ == "__main__":
    main()
