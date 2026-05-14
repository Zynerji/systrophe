//! Phase 3b correctness check: GPU pubkey == CPU pubkey on 1024 random keys.
//!
//! Generates random private keys, derives the compressed pubkey on the
//! GPU (our kernel) and on the CPU (the `secp256k1` Rust crate, which
//! wraps libsecp256k1 from Bitcoin Core). Any mismatch is fatal.

use anyhow::{anyhow, Result};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use secp256k1::{PublicKey, Secp256k1, SecretKey};
use secp256k1_cuda::Secp256k1Cuda;

const N_KEYS: usize = 1024;

fn main() -> Result<()> {
    let cpu = Secp256k1::new();
    let gpu = Secp256k1Cuda::new()?;

    let mut rng = StdRng::seed_from_u64(0xCAFEBABE);
    let mut priv_be = vec![0u8; 32 * N_KEYS];
    let mut cpu_pubs = vec![0u8; 33 * N_KEYS];

    // Build a batch of valid private keys and derive on CPU.
    let mut i = 0;
    while i < N_KEYS {
        let mut k = [0u8; 32];
        rng.fill_bytes(&mut k);
        // SecretKey rejects 0 and >=n; just try again on failure.
        let sk = match SecretKey::from_slice(&k) {
            Ok(sk) => sk,
            Err(_) => continue,
        };
        let pk = PublicKey::from_secret_key(&cpu, &sk);
        let comp = pk.serialize(); // 33 bytes compressed
        priv_be[i * 32..(i + 1) * 32].copy_from_slice(&k);
        cpu_pubs[i * 33..(i + 1) * 33].copy_from_slice(&comp);
        i += 1;
    }

    // GPU derivation
    let gpu_pubs = gpu.derive_pubkeys(&priv_be)?;
    if gpu_pubs.len() != cpu_pubs.len() {
        return Err(anyhow!("length mismatch: gpu={} cpu={}", gpu_pubs.len(), cpu_pubs.len()));
    }

    let mut mismatches = 0usize;
    for j in 0..N_KEYS {
        let g = &gpu_pubs[j * 33..(j + 1) * 33];
        let c = &cpu_pubs[j * 33..(j + 1) * 33];
        if g != c {
            mismatches += 1;
            if mismatches <= 4 {
                println!("MISMATCH key #{}", j);
                println!("  priv: {}", hex::encode(&priv_be[j * 32..(j + 1) * 32]));
                println!("  cpu:  {}", hex::encode(c));
                println!("  gpu:  {}", hex::encode(g));
            }
        }
    }

    if mismatches == 0 {
        println!("MATCH ✓ {} / {} keys agree (CPU libsecp256k1 == GPU CUDA kernel)", N_KEYS, N_KEYS);
        Ok(())
    } else {
        Err(anyhow!("{}/{} mismatches", mismatches, N_KEYS))
    }
}
