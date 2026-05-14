"""GPU acceleration kernels for the brain-wallet auditor.

These are Triton kernels for the byte-level primitives in the
derivation pipeline:

  * `triton_sha256` -- SHA-256 of `(batch, max_len)` byte inputs.
  * `triton_ripemd160` -- RIPEMD-160 of `(batch, 32)` byte inputs.

The secp256k1 scalar multiplication is the load-bearing kernel for
end-to-end throughput; it is NOT in Triton because Triton is a poor
fit for modular-inverse-heavy elliptic-curve code. That kernel is
planned for a CUDA C++ implementation (Phase 3, separate file).

Phase 2 scope: prove Triton kernels for the hash stages, validate
bit-exactness against `hashlib`, microbenchmark on the target
hardware. The numbers feed Phase 3's secp256k1 budget.
"""

from .triton_sha256 import sha256_batch
from .triton_ripemd160 import ripemd160_batch

__all__ = [
    "sha256_batch",
    "ripemd160_batch",
]
