//! Phase 3c throughput benchmark on Blackwell.
//!
//! Streams batches of random 32-byte private keys through the kernel,
//! reports keys/sec amortised. The first batch is a warmup (NVRTC
//! compile + first-launch overhead is paid once in Secp256k1Cuda::new
//! and the launch warmup here).

use anyhow::Result;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use secp256k1_cuda::Secp256k1Cuda;
use std::time::Instant;

fn fill_random_priv(buf: &mut [u8], rng: &mut StdRng) {
    rng.fill_bytes(buf);
    // Don't bother rejecting >=n; the kernel still produces an output
    // (just not a useful one), and the rejection rate is < 2^-128.
}

fn main() -> Result<()> {
    let gpu = Secp256k1Cuda::new()?;
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);

    // Warmup
    {
        let mut priv_be = vec![0u8; 32 * 4096];
        fill_random_priv(&mut priv_be, &mut rng);
        let _ = gpu.derive_pubkeys(&priv_be)?;
    }

    println!("{:>10}  {:>12}  {:>10}", "batch", "elapsed_ms", "keys/sec");
    for batch in [4096usize, 16_384, 65_536, 262_144, 1_048_576] {
        let mut priv_be = vec![0u8; 32 * batch];
        fill_random_priv(&mut priv_be, &mut rng);

        // Time three runs, keep the fastest.
        let mut best_ms = f64::INFINITY;
        for _ in 0..3 {
            let t0 = Instant::now();
            let _ = gpu.derive_pubkeys(&priv_be)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if ms < best_ms {
                best_ms = ms;
            }
        }
        let kps = (batch as f64) / (best_ms / 1e3);
        println!("{:>10}  {:>12.2}  {:>10.0}", batch, best_ms, kps);
    }
    Ok(())
}
