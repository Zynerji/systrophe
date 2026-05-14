# secp256k1-cuda

GPU-accelerated `pub = priv * G` on the secp256k1 curve. Rust host
driver + CUDA C kernel (NVRTC-compiled at runtime via `cudarc`).
Phase 3 of the brain-wallet auditor's hash pipeline: Triton SHA-256
(Phase 2) provides the hash stages; this crate provides the
elliptic-curve scalar multiplication.

## Status

* Smoke OK: derives canonical "correct horse battery staple" pubkey
  identically to libsecp256k1.
* Verified bit-exact against `secp256k1` Rust crate (libsecp256k1
  wrapper) on 1024 random keys.
* Throughput on NVIDIA RTX PRO 6000 Blackwell Workstation Edition,
  CUDA 13.2, cudarc 0.19.6, Rust 1.95:

  | batch     | elapsed (ms) | keys/sec    |
  | --------- | -----------: | ----------: |
  | 4 096     |        6.54  |     626 280 |
  | 16 384    |        6.49  |   2 525 133 |
  | 65 536    |        9.38  |   6 985 274 |
  | 262 144   |       30.29  |   8 655 588 |
  | 1 048 576 |      121.66  |   8 618 908 |

  Sustained: **8.6 M keys/sec** at batch ≥ 256 k. Plateau is set by
  the kernel cost; DMA is not the bottleneck at this batch size.

## Honest limits

* Algorithm: textbook Jacobian double-and-add, MSB→LSB, no
  windowed-NAF, no fixed-point precomputed multiples of G. Real
  vanitygen / BitCrack-tier implementations apply a 4-bit or 8-bit
  precomputed kG table and gain ~5-10x. We don't.
* Field arithmetic: simple pseudo-Mersenne reduction (p = 2^256 -
  0x1000003D1). Modular inversion via Fermat's little theorem
  (a^(p-2)). A constant-time-side-channel-resistant Montgomery
  ladder + constant-time-inversion would be needed for hostile
  contexts; we don't need that here (this is offline batch derivation).
* Per-batch NVRTC compile is paid once in `Secp256k1Cuda::new()`; reuse
  the struct across many `derive_pubkeys` calls.
* Kernel is for valid private keys 1 <= k < n. We don't reject k >= n
  on the GPU (the rejection rate is < 2^-128 for random k); the
  brain-wallet caller validates by deriving back and checking.

## Build

```bash
# Requires Rust 1.87+ (for is_multiple_of) and CUDA 13.x toolkit
# with `nvrtc` available (nvcc + libnvrtc.so).
cargo build --release
```

## Run

```bash
./target/release/smoke     # canonical "correct horse" sanity check
./target/release/verify    # 1024 random keys: GPU vs libsecp256k1
./target/release/bench     # throughput across batch sizes
```

## Layout

```
secp256k1_rs/
├── Cargo.toml
├── src/
│   ├── lib.rs       public Secp256k1Cuda type
│   ├── ctx.rs       cudarc orchestration: NVRTC compile + launch
│   ├── kernel.rs    CUDA C kernel source as a Rust const string
│   └── bin/
│       ├── smoke.rs   anchor sanity check
│       ├── verify.rs  1024-key correctness vs libsecp256k1
│       └── bench.rs   throughput vs batch size
```

## License

MIT.
