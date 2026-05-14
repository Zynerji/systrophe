//! GPU-accelerated secp256k1 scalar multiplication for the brain-wallet auditor.
//!
//! Phase 3 of the brain-wallet auditor's GPU pipeline. The Triton SHA-256
//! and RIPEMD-160 kernels (Phase 2, Python) handle the hash stages; this
//! crate handles the load-bearing primitive: `pub = priv * G` on the
//! secp256k1 curve.
//!
//! Architecture: a Rust host driver (this crate) loads a CUDA C kernel via
//! NVRTC at runtime, marshals batches of 32-byte private keys to the GPU,
//! and returns batches of 33-byte compressed public keys. Correctness is
//! verified against the `secp256k1` Rust crate.

pub mod ctx;
pub mod kernel;

pub use ctx::{Secp256k1Cuda, BLOCK_DIM};
