"""Correctness check: Triton SHA-256 vs hashlib on N random short inputs.

Run on a CUDA host. Usage:
    python tools/brain-wallet-auditor/kernels/test_triton_correctness.py
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from kernels.triton_sha256 import sha256_batch


def make_batch(n: int, max_len: int = 55, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, list[bytes]]:
    """Random bytes with random lengths in [0, max_len]."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, max_len + 1, size=n).astype(np.int32)
    msgs = np.zeros((n, 64), dtype=np.uint8)
    raws: list[bytes] = []
    for i in range(n):
        L = int(lengths[i])
        if L > 0:
            payload = rng.integers(0, 256, size=L).astype(np.uint8)
            msgs[i, :L] = payload
            raws.append(bytes(payload))
        else:
            raws.append(b"")
    return (
        torch.from_numpy(msgs).cuda(),
        torch.from_numpy(lengths).cuda(),
        raws,
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA not available; skipping")
        return
    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}")

    n = 1024
    msgs, lens, raws = make_batch(n, max_len=55, seed=0)
    t0 = time.time()
    out = sha256_batch(msgs, lens)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Triton SHA-256: {n} digests in {elapsed*1000:.2f} ms "
          f"({n/elapsed:,.0f} per sec)")

    # Verify against hashlib
    out_np = out.cpu().numpy()
    n_bad = 0
    for i, raw in enumerate(raws):
        truth = hashlib.sha256(raw).digest()
        got = bytes(out_np[i])
        if got != truth:
            n_bad += 1
            if n_bad <= 3:
                print(f"  MISMATCH at i={i} len={len(raw)}:")
                print(f"    raw   = {raw.hex()}")
                print(f"    truth = {truth.hex()}")
                print(f"    got   = {got.hex()}")
    if n_bad == 0:
        print(f"All {n} digests match hashlib. PASS.")
    else:
        print(f"FAIL: {n_bad}/{n} digests do not match.")


if __name__ == "__main__":
    main()
