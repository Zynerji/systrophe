"""Derivation backends for audit_passphrases().

Three backends:

* ``cpu_single`` — the original single-threaded coincurve path
  (~1200 brainwallet_sha256 derivations/sec on a typical laptop CPU).

* ``cpu_mp`` — a `multiprocessing.Pool`-backed parallel CPU path.
  Linear-ish speedup with core count for the brainwallet_sha256 and
  bip39 schemes; WarpWallet is so slow per-derivation that even
  serial mode is acceptable, but parallel still works.

* ``gpu_cuda`` — stub for the secp256k1 Rust+CUDA kernel
  (``tools/brain-wallet-auditor/kernels/secp256k1_rs``). Loads the
  built shared library via ctypes if present. If the library is
  missing (no GPU / no Rust build), raises a clear RuntimeError at
  construction. Once the kernel is wired in, the CPU-side pubkey ->
  address conversion (SHA256 + RIPEMD160 + base58) still runs on the
  caller's CPU; the GPU only accelerates the scalar-multiplication
  step.

Pick a backend with ``audit_passphrases(..., backend='cpu_mp')``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from .derivation import AddressDerivation, Scheme


def _derive_one(args: tuple[Scheme, str, dict]) -> str:
    """Module-level worker function (picklable for multiprocessing)."""
    scheme, passphrase, kwargs = args
    deriv = AddressDerivation(scheme=scheme, **kwargs)
    return deriv.derive(passphrase)


class DerivationBackend(ABC):
    """Common backend interface."""

    name: str = "unknown"

    @abstractmethod
    def derive_batch(self, scheme: Scheme, passphrases: list[str],
                     derivation_kwargs: dict) -> list[str]:
        """Return the derived address for each passphrase, in input order."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class CPUSingleBackend(DerivationBackend):
    """Single-process, single-thread baseline. Matches the original API."""

    name = "cpu_single"

    def derive_batch(self, scheme: Scheme, passphrases: list[str],
                     derivation_kwargs: dict) -> list[str]:
        deriv = AddressDerivation(scheme=scheme, **derivation_kwargs)
        return [deriv.derive(p) for p in passphrases]


class CPUMultiprocessingBackend(DerivationBackend):
    """multiprocessing.Pool-backed parallel CPU path.

    Default n_workers = os.cpu_count(). chunk_size controls the per-
    worker batch granularity; default 64 is a reasonable sweet spot.

    NOTE on Windows: process spawn is expensive (~hundreds of ms per
    pool startup) and re-imports every module per worker. For fast
    schemes like brainwallet_sha256 (~130 µs each), the overhead can
    exceed the work and this backend is SLOWER than cpu_single. Use
    ``cpu_threads`` for those instead. Multiprocessing wins for slow
    schemes (warpwallet, bip39 with many addresses).
    """

    name = "cpu_mp"

    def __init__(self, n_workers: int | None = None,
                 chunk_size: int = 64) -> None:
        self.n_workers = int(n_workers) if n_workers else max(1, os.cpu_count() or 1)
        self.chunk_size = int(chunk_size)

    def derive_batch(self, scheme: Scheme, passphrases: list[str],
                     derivation_kwargs: dict) -> list[str]:
        if not passphrases:
            return []
        # Below a threshold, the process-pool overhead dominates;
        # fall back to single-thread.
        if len(passphrases) < 32:
            return CPUSingleBackend().derive_batch(
                scheme, passphrases, derivation_kwargs,
            )
        args = [(scheme, p, derivation_kwargs) for p in passphrases]
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            return list(pool.map(_derive_one, args, chunksize=self.chunk_size))


class CPUThreadsBackend(DerivationBackend):
    """ThreadPoolExecutor-backed parallel CPU path.

    Works because libsecp256k1 (via coincurve) and hashlib release the
    GIL during their C-level work — so the elliptic-curve scalar
    multiplication and SHA256 run in parallel across threads.

    On Windows-style hosts where process spawn is expensive, this is
    typically the right choice for fast schemes (brainwallet_sha256,
    BIP-39 single-address) — no per-task pickle round-trip, no
    interpreter spawn.
    """

    name = "cpu_threads"

    def __init__(self, n_workers: int | None = None) -> None:
        self.n_workers = int(n_workers) if n_workers else max(1, os.cpu_count() or 1)

    def derive_batch(self, scheme: Scheme, passphrases: list[str],
                     derivation_kwargs: dict) -> list[str]:
        if not passphrases:
            return []
        if len(passphrases) < 32:
            return CPUSingleBackend().derive_batch(
                scheme, passphrases, derivation_kwargs,
            )
        deriv = AddressDerivation(scheme=scheme, **derivation_kwargs)
        # The bound method holds C-extension entry points; threads
        # call straight in.
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            return list(pool.map(deriv.derive, passphrases))


class GPUCudaBackend(DerivationBackend):
    """Stub that loads the Rust+CUDA secp256k1 kernel via ctypes.

    Construction raises RuntimeError if the shared library is missing.
    Useful as a feature-detection probe: try to construct, catch the
    error, fall back to CPU.

    To enable, build the kernel on a CUDA-capable host:

        cd tools/brain-wallet-auditor/kernels/secp256k1_rs
        cargo build --release

    And ensure ``target/release/libsecp256k1_cuda.{so,dylib,dll}``
    exists. The current Rust crate produces a Rust ``rlib`` only —
    the C-ABI exposure is the next step (cdylib + a small ``extern "C"``
    wrapper around Secp256k1Cuda::derive_pubkeys).
    """

    name = "gpu_cuda"

    def __init__(self, kernel_dir: str | None = None) -> None:
        import ctypes

        if kernel_dir is None:
            here = Path(__file__).resolve().parents[1]
            kernel_dir = str(here / "kernels" / "secp256k1_rs" / "target" / "release")
        candidates = [
            Path(kernel_dir) / "libsecp256k1_cuda.so",
            Path(kernel_dir) / "libsecp256k1_cuda.dylib",
            Path(kernel_dir) / "secp256k1_cuda.dll",
        ]
        for c in candidates:
            if c.exists():
                try:
                    self._lib = ctypes.CDLL(str(c))
                    # The Rust crate currently builds as rlib only. The
                    # cdylib C-ABI surface (`derive_pubkeys_c`) is the
                    # remaining wiring step.
                    if not hasattr(self._lib, "derive_pubkeys_c"):
                        raise RuntimeError(
                            f"Found {c} but it does not export "
                            f"`derive_pubkeys_c`. Rebuild the kernel with "
                            f"the cdylib crate-type and the extern \"C\" "
                            f"wrapper (see kernels/secp256k1_rs/README.md)."
                        )
                    self._path = c
                    return
                except OSError as e:
                    raise RuntimeError(f"Failed to load {c}: {e}") from e
        raise RuntimeError(
            f"No libsecp256k1_cuda.{{so,dylib,dll}} in {kernel_dir}. "
            f"Build it with `cargo build --release` in "
            f"tools/brain-wallet-auditor/kernels/secp256k1_rs/ on a "
            f"CUDA-capable host."
        )

    def derive_batch(self, scheme: Scheme, passphrases: list[str],
                     derivation_kwargs: dict) -> list[str]:
        # Reaches here only if the .dll/.so loaded and exposes
        # derive_pubkeys_c. Wire-up still TODO. For now fall back to
        # single-process CPU so an accidentally-loaded stub doesn't
        # silently return wrong addresses.
        raise NotImplementedError(
            "GPU kernel detected but Python <-> Rust C-ABI wiring is "
            "not yet implemented. Use backend='cpu_mp' for a working "
            "speedup today."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_BACKEND_REGISTRY = {
    "cpu_single": CPUSingleBackend,
    "cpu_mp": CPUMultiprocessingBackend,
    "cpu_threads": CPUThreadsBackend,
    "gpu_cuda": GPUCudaBackend,
}


def get_backend(name: str, **kwargs) -> DerivationBackend:
    """Construct a backend by name. kwargs forwarded to the constructor."""
    if name not in _BACKEND_REGISTRY:
        raise ValueError(
            f"unknown backend {name!r}; choose from {list(_BACKEND_REGISTRY)}"
        )
    return _BACKEND_REGISTRY[name](**kwargs)
