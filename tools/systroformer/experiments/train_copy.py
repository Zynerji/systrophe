"""Train MiniSystroformer on the sequence-copy task.

The model must reproduce its input token sequence exactly. The task
is trivial classically; here it serves as a controlled bed to verify
the catcher integrates cleanly into the training loop and that the
catcher signal stays informative through the optimisation.

Records:
  - per-epoch loss
  - per-block λ₂ history
  - per-block lambda_scale evolution

Run: ``python experiments/train_copy.py``
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the systroformer package importable when run directly
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from systroformer import MiniSystroformer


def gen_batch(batch_size: int, seq_len: int, vocab_size: int,
                rng: np.random.Generator, emergence_rate: float = 0.0,
                emergence_cluster_size: int = 3) -> torch.Tensor:
    """Generate copy-task batch.

    Each sample is a random token sequence; the model must reproduce it.
    If emergence_rate > 0, that fraction of samples will have a small
    contiguous "emergence" cluster of repeated tokens.
    """
    x = rng.integers(0, vocab_size, size=(batch_size, seq_len)).astype(np.int64)
    if emergence_rate > 0:
        for b in range(batch_size):
            if rng.random() < emergence_rate:
                pos = rng.integers(0, seq_len - emergence_cluster_size + 1)
                token = rng.integers(0, vocab_size)
                x[b, pos: pos + emergence_cluster_size] = token
    return torch.from_numpy(x)


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--emergence_rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    model = MiniSystroformer(
        vocab_size=args.vocab_size, d_model=args.d_model,
        n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=args.seq_len,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_history = []
    lambda_avg_per_epoch = []
    scale_evolution = []
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_losses = []
        for step in range(args.steps_per_epoch):
            x = gen_batch(args.batch_size, args.seq_len, args.vocab_size,
                          rng, emergence_rate=args.emergence_rate).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), x.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        loss_history.append(epoch_losses)
        lambdas_now = model.current_lambdas()
        lambda_avg_per_epoch.append([float(x) for x in lambdas_now])
        scale_evolution.append(model.lambda_scales())
        avg_loss = float(np.mean(epoch_losses))
        print(f"epoch {epoch:2d}  avg_loss={avg_loss:.4f}  "
              f"lambdas={[f'{x:.2f}' for x in lambdas_now]}  "
              f"scales={[f'{s:.3f}' for s in scale_evolution[-1]]}")

    elapsed = time.time() - t_start
    final_loss = float(np.mean(loss_history[-1]))
    out = {
        "tool": "systroformer",
        "task": "copy",
        "args": vars(args),
        "loss_history": loss_history,
        "final_loss": final_loss,
        "lambda_per_epoch_per_block": lambda_avg_per_epoch,
        "lambda_scale_per_epoch_per_block": scale_evolution,
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
