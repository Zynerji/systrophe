"""Throughput benchmark for the Triton SHA-256 kernel.

Excludes JIT compile time via warmup. Sweeps batch size to find the
hardware-saturating point.
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


def make_batch(n: int, max_len: int = 55, seed: int = 0):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(0, max_len + 1, size=n).astype(np.int32)
    msgs = np.zeros((n, 64), dtype=np.uint8)
    for i in range(n):
        L = int(lengths[i])
        if L > 0:
            msgs[i, :L] = rng.integers(0, 256, size=L)
    return (
        torch.from_numpy(msgs).cuda(),
        torch.from_numpy(lengths).cuda(),
    )


def time_n(n: int, n_warmup: int = 1, n_repeat: int = 5) -> dict:
    msgs, lens = make_batch(n)
    for _ in range(n_warmup):
        _ = sha256_batch(msgs, lens)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_repeat):
        _ = sha256_batch(msgs, lens)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / n_repeat
    return {
        "n": n,
        "elapsed_s": elapsed,
        "hashes_per_sec": n / elapsed,
        "gb_per_sec": n * 64.0 / elapsed / 1e9,
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA not available"); return
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    print(f"{'batch':>10s}  {'elapsed':>10s}  {'hashes/s':>15s}  {'in GB/s':>10s}")
    for n in (1024, 4096, 16384, 65536, 262144, 1048576):
        try:
            r = time_n(n)
            print(f"{n:>10d}  {r['elapsed_s']*1000:>8.2f}ms  "
                  f"{r['hashes_per_sec']:>15,.0f}  {r['gb_per_sec']:>8.2f}")
        except Exception as e:
            print(f"{n:>10d}  ERROR: {e}")
            break


if __name__ == "__main__":
    main()
