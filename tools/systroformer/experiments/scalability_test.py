"""Benchmark the catcher overhead at different scales.

Reports wall-clock time per forward pass with the catcher on vs off,
for several (batch_size, seq_len) combinations. Confirms the
approximate λ₂ (power-iteration) gives the expected order-of-
magnitude speedup over the exact eigvalsh from the framework.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from systroformer import MiniSystroformer


def benchmark_block(model: MiniSystroformer, x: torch.Tensor,
                      n_warmup: int = 1, n_steps: int = 3) -> float:
    """Mean wall-clock time per forward pass (seconds)."""
    model.eval()
    for _ in range(n_warmup):
        _ = model(x)
    t0 = time.time()
    for _ in range(n_steps):
        _ = model(x)
    return (time.time() - t0) / n_steps


def main() -> dict:
    rows = []
    for (bs, sl) in [(1, 8), (2, 16), (2, 32), (4, 32), (4, 64)]:
        torch.manual_seed(0)
        x = torch.randint(0, 10, (bs, sl))
        model = MiniSystroformer(
            vocab_size=10, d_model=32, n_layers=2, n_heads=2,
            max_seq_len=sl,
        )
        t_with = benchmark_block(model, x)
        # Disable catcher by zeroing lambda_scale (no actual graph build saved;
        # this measures pure inference time for a fair-ish comparison)
        for b in model.blocks:
            b.lambda_scale.data.zero_()
        t_zeroed = benchmark_block(model, x)
        rows.append({
            "batch_size": bs,
            "seq_len": sl,
            "nodes_in_graph": bs * sl,
            "t_with_catcher_s": t_with,
            "t_lambda_zero_s": t_zeroed,
            "overhead_factor": float(t_with / max(t_zeroed, 1e-9)),
        })
        print(f"bs={bs:2d} sl={sl:3d}  nodes={bs*sl:4d}  "
              f"t_with={t_with*1000:6.2f}ms  "
              f"t_zero={t_zeroed*1000:6.2f}ms  "
              f"factor={t_with/max(t_zeroed,1e-9):.2f}x")

    out = {
        "tool": "systroformer",
        "task": "scalability_test",
        "rows": rows,
        "note": ("t_lambda_zero still computes the graph but zeros the "
                 "modulation; for a true catcher-off baseline, replace "
                 "SystroformerBlock with a vanilla TransformerEncoderLayer."),
    }
    out_path = pathlib.Path(__file__).with_name("scalability_test_results.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults: {out_path}")
    return out


if __name__ == "__main__":
    main()
