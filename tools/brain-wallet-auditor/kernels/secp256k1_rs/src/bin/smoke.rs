//! Phase 3a smoke test: prove cudarc + NVRTC compile + kernel launch + DMA round trip.
//!
//! Derives the public key for the famous "correct horse battery staple"
//! brainwallet private key on both the GPU (our kernel) and CPU (the
//! `secp256k1` Rust crate, which wraps libsecp256k1 from Bitcoin Core)
//! and asserts they agree.

use anyhow::{anyhow, Result};
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use secp256k1_cuda::Secp256k1Cuda;
use sha2::{Digest, Sha256};

fn main() -> Result<()> {
    let mut h = Sha256::new();
    h.update(b"correct horse battery staple");
    let priv_be = h.finalize();
    println!("priv (BE):  {}", hex::encode(&priv_be));

    // CPU reference
    let cpu = Secp256k1::new();
    let sk = SecretKey::from_slice(&priv_be)?;
    let cpu_pub = PublicKey::from_secret_key(&cpu, &sk).serialize();
    println!("cpu  (BE):  {}", hex::encode(&cpu_pub));

    // GPU
    let g = Secp256k1Cuda::new()?;
    let gpu_pub = g.derive_pubkeys(&priv_be)?;
    assert_eq!(gpu_pub.len(), 33);
    println!("gpu  (BE):  {}", hex::encode(&gpu_pub));

    if cpu_pub[..] == gpu_pub[..] {
        println!("MATCH ✓ cudarc + NVRTC + secp256k1 kernel + DMA round trip OK");
        Ok(())
    } else {
        Err(anyhow!("GPU pubkey != CPU pubkey on canonical 'correct horse' key"))
    }
}
