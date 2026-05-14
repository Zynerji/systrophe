//! Rust host driver: load the secp256k1 CUDA kernel via NVRTC, marshal batches,
//! collect compressed pubkeys.

use anyhow::{anyhow, Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use std::sync::Arc;

use crate::kernel::SECP256K1_CUDA_SRC;

/// CUDA block size used for the kernel launches. One thread per private key.
pub const BLOCK_DIM: u32 = 64;

/// Compiled secp256k1 CUDA module bound to a specific GPU context.
///
/// Construct once, call `derive_pubkeys` many times. The NVRTC compile
/// is a few hundred ms; we don't want to repeat it per batch.
pub struct Secp256k1Cuda {
    ctx: Arc<CudaContext>,
    func: cudarc::driver::CudaFunction,
}

impl Secp256k1Cuda {
    /// Compile the kernel for the default CUDA device (ordinal 0).
    pub fn new() -> Result<Self> {
        Self::new_with_device(0)
    }

    /// Compile for a specific device ordinal.
    pub fn new_with_device(device: usize) -> Result<Self> {
        let ctx = CudaContext::new(device)
            .with_context(|| format!("CudaContext::new({device}) failed"))?;
        // NVRTC compile of the kernel source. The PTX comes out
        // device-specific via the compute capability flag.
        let ptx = cudarc::nvrtc::compile_ptx(SECP256K1_CUDA_SRC)
            .with_context(|| "NVRTC compile of secp256k1 kernel failed")?;
        let module = ctx.load_module(ptx)
            .with_context(|| "Loading PTX module failed")?;
        let func = module
            .load_function("secp256k1_priv_to_pub")
            .with_context(|| "Locating `secp256k1_priv_to_pub` in the module failed")?;
        Ok(Self { ctx, func })
    }

    /// Derive compressed pubkeys for a batch of private keys.
    ///
    /// `private_keys_be`: `n` * 32 big-endian bytes. Returns `n` * 33
    /// big-endian bytes (1 prefix byte + 32-byte x coordinate).
    pub fn derive_pubkeys(&self, private_keys_be: &[u8]) -> Result<Vec<u8>> {
        if !private_keys_be.len().is_multiple_of(32) {
            return Err(anyhow!(
                "private_keys_be length {} not a multiple of 32",
                private_keys_be.len()
            ));
        }
        let n = (private_keys_be.len() / 32) as i32;
        if n == 0 {
            return Ok(vec![]);
        }

        let stream = self.ctx.default_stream();
        let priv_dev: CudaSlice<u8> = stream
            .clone_htod(private_keys_be)
            .with_context(|| "memcpy host -> device for private keys failed")?;
        let mut pub_dev: CudaSlice<u8> = stream
            .alloc_zeros::<u8>((n as usize) * 33)
            .with_context(|| "allocating device-side pubkey buffer failed")?;

        let grid = (n as u32 + BLOCK_DIM - 1) / BLOCK_DIM;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.func);
        launch.arg(&priv_dev);
        launch.arg(&mut pub_dev);
        launch.arg(&n);
        unsafe { launch.launch(cfg) }
            .with_context(|| "secp256k1_priv_to_pub kernel launch failed")?;

        let pub_host: Vec<u8> = stream
            .clone_dtoh(&pub_dev)
            .with_context(|| "memcpy device -> host for pubkeys failed")?;
        stream.synchronize()?;
        Ok(pub_host)
    }
}
