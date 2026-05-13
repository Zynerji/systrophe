"""Train a small Cyliformer on the sequence-copy task.

This is the same trivial classification target used by Systroformer
(`tools/systroformer/experiments/train_copy.py`), so the two can be
compared head-to-head at matched scale.

Records:
  - per-step CE loss
  - per-layer per-cylinder lambda_2 history
  - per-layer beam-gain evolution
  - parameter-count breakdown at startup
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cyliformer import Cyliformer, cyliformer_loss


def gen_batch(batch_size: int, seq_len: int, vocab_size: int,
                rng: np.random.Generator) -> torch.Tensor:
    x = rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64)
    return torch.from_numpy(x)


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_cylinders", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lambda_target", type=float, default=0.20)
    parser.add_argument("--lambda_weight", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    model = Cyliformer(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        n_cylinders=args.n_cylinders, max_seq_len=args.seq_len,
        lambda_target=args.lambda_target,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    breakdown = model.param_count_breakdown()
    print("Parameter breakdown:")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")
    print()

    loss_history = []
    lambda_per_epoch = []
    beam_per_epoch = []
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        for step in range(args.steps_per_epoch):
            x = gen_batch(args.batch_size, args.seq_len, args.vocab_size, rng).to(device)
            logits = model(x)
            loss = cyliformer_loss(
                logits, x,
                model.all_lambdas(),
                lambda_target=args.lambda_target,
                lambda_weight=args.lambda_weight,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(epoch_losses)
        lambda_now = model.all_lambdas()
        beam_now = model.beam_gains()
        lambda_per_epoch.append(lambda_now)
        beam_per_epoch.append(beam_now)
        avg = float(np.mean(epoch_losses))
        # Flatten per-layer means for compact print
        per_layer_lam = [float(np.mean(layer)) if layer else 0.0 for layer in lambda_now]
        print(f"epoch {epoch:2d}  loss={avg:.4f}  "
              f"lambda_per_layer={[f'{v:.3f}' for v in per_layer_lam]}  "
              f"beam_per_layer={[f'{v:.3f}' for v in beam_now]}")

    elapsed = time.time() - t_start
    final_loss = float(np.mean(loss_history[-1]))

    out = {
        "tool": "cyliformer",
        "task": "copy",
        "args": vars(args),
        "param_breakdown": breakdown,
        "loss_history": loss_history,
        "final_loss": final_loss,
        "lambda_per_epoch_per_layer_per_cylinder": lambda_per_epoch,
        "beam_per_epoch_per_layer": beam_per_epoch,
        "elapsed_seconds": elapsed,
    }
    if args.out:
        out_path = pathlib.Path(args.out)
    else:
        out_path = pathlib.Path(__file__).with_name("train_copy_results.json")
    out_path.write_text(json.dumps(out, indent=2))

    print()
    print(f"Final loss: {final_loss:.4f}")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Results: {out_path}")
    return out


if __name__ == "__main__":
    main()
