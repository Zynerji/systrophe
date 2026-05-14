"""Tests for the derivation backends.

The critical property: same passphrases must produce the same
addresses across every backend. If a backend disagrees, hits and
misses both become unreliable.
"""

from __future__ import annotations

import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brain_wallet_auditor import (
    CPUMultiprocessingBackend,
    CPUSingleBackend,
    CPUThreadsBackend,
    DerivationBackend,
    FundedAddressSet,
    GPUCudaBackend,
    audit_passphrases,
    brainwallet_address,
    get_backend,
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_get_backend_cpu_single():
    b = get_backend("cpu_single")
    assert isinstance(b, CPUSingleBackend)


def test_get_backend_cpu_mp():
    b = get_backend("cpu_mp", n_workers=2, chunk_size=8)
    assert isinstance(b, CPUMultiprocessingBackend)
    assert b.n_workers == 2
    assert b.chunk_size == 8


def test_get_backend_cpu_threads():
    b = get_backend("cpu_threads", n_workers=4)
    assert isinstance(b, CPUThreadsBackend)
    assert b.n_workers == 4


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("not_a_real_backend")


def test_get_backend_gpu_raises_when_missing():
    """No built CUDA kernel → GPUCudaBackend construction raises clearly."""
    with pytest.raises(RuntimeError):
        get_backend("gpu_cuda")


# ---------------------------------------------------------------------------
# CPU single backend (baseline; should already match the existing path)
# ---------------------------------------------------------------------------


def test_cpu_single_matches_direct_derivation():
    pps = ["password", "qwerty", "MyDog2014!", "correct horse battery staple"]
    b = CPUSingleBackend()
    addrs = b.derive_batch("brainwallet_sha256", pps, {})
    expected = [brainwallet_address(p) for p in pps]
    assert addrs == expected


# ---------------------------------------------------------------------------
# Backend equivalence: cpu_single == cpu_mp on identical input
# ---------------------------------------------------------------------------


def test_cpu_mp_matches_cpu_single_brainwallet():
    """The multiprocessing backend must produce identical addresses."""
    pps = [f"phrase_number_{i:04d}" for i in range(64)]
    addrs_single = CPUSingleBackend().derive_batch("brainwallet_sha256", pps, {})
    addrs_mp = CPUMultiprocessingBackend(n_workers=2, chunk_size=8).derive_batch(
        "brainwallet_sha256", pps, {},
    )
    assert addrs_single == addrs_mp


def test_cpu_mp_matches_cpu_single_bip39():
    """Same equivalence for BIP-39 (slower, just 4 candidates)."""
    pps = [
        "abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon abandon abandon about",
        "legal winner thank year wave sausage worth useful "
        "legal winner thank yellow",
        "letter advice cage absurd amount doctor acoustic "
        "avoid letter advice cage above",
        "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
    ]
    # Force multiprocessing on this small batch by setting threshold low
    # via direct call (the audit_passphrases threshold is 32).
    # We test via the public derive_batch which will fall back to single
    # for batches < 32. So manually use Pool directly via a 64-batch.
    pps_padded = pps * 16  # 64 entries — above the threshold
    addrs_single = CPUSingleBackend().derive_batch("bip39", pps_padded, {})
    addrs_mp = CPUMultiprocessingBackend(n_workers=2, chunk_size=4).derive_batch(
        "bip39", pps_padded, {},
    )
    assert addrs_single == addrs_mp


def test_cpu_threads_matches_cpu_single():
    """Threads must produce identical addresses to single-thread."""
    pps = [f"thread_test_{i:04d}" for i in range(64)]
    addrs_single = CPUSingleBackend().derive_batch("brainwallet_sha256", pps, {})
    addrs_threads = CPUThreadsBackend(n_workers=4).derive_batch(
        "brainwallet_sha256", pps, {},
    )
    assert addrs_single == addrs_threads


def test_cpu_threads_below_threshold_falls_back():
    """Tiny batches use single-thread."""
    pps = ["x", "y", "z"]
    addrs_threads = CPUThreadsBackend(n_workers=4).derive_batch(
        "brainwallet_sha256", pps, {},
    )
    expected = [brainwallet_address(p) for p in pps]
    assert addrs_threads == expected


def test_cpu_mp_below_threshold_falls_back_to_single():
    """Tiny batches use single-thread to avoid pool spawn overhead."""
    pps = ["a", "b", "c", "d", "e"]  # 5 < 32 threshold
    addrs_mp = CPUMultiprocessingBackend(n_workers=4).derive_batch(
        "brainwallet_sha256", pps, {},
    )
    expected = [brainwallet_address(p) for p in pps]
    assert addrs_mp == expected


# ---------------------------------------------------------------------------
# audit_passphrases backend argument
# ---------------------------------------------------------------------------


def test_audit_default_backend_is_cpu_single():
    """Back-compat: default backend produces same results as before."""
    pps = [f"x{i}" for i in range(40)]
    report = audit_passphrases(
        pps, schemes=["brainwallet_sha256"],
        funded_set=FundedAddressSet(), run_diagnostic=False,
    )
    assert len(report.results) == 40


def test_audit_cpu_mp_backend_matches_cpu_single():
    """audit_passphrases with backend='cpu_mp' must produce identical
    addresses to backend='cpu_single' on the same input."""
    pps = [f"audit_test_{i:03d}" for i in range(48)]
    funded = FundedAddressSet()

    r_single = audit_passphrases(
        pps, schemes=["brainwallet_sha256"], funded_set=funded,
        run_diagnostic=False, backend="cpu_single",
    )
    r_mp = audit_passphrases(
        pps, schemes=["brainwallet_sha256"], funded_set=funded,
        run_diagnostic=False, backend="cpu_mp",
        backend_kwargs={"n_workers": 2, "chunk_size": 8},
    )
    addrs_single = [r.address for r in r_single.results]
    addrs_mp = [r.address for r in r_mp.results]
    assert addrs_single == addrs_mp


def test_audit_finds_planted_hit_via_cpu_mp():
    """The end-to-end speed path must still find planted hits."""
    target = brainwallet_address("MyDog2014!")
    funded = FundedAddressSet.from_iterable([target])
    pps = [f"decoy_{i}" for i in range(48)] + ["MyDog2014!"]
    r = audit_passphrases(
        pps, schemes=["brainwallet_sha256"], funded_set=funded,
        run_diagnostic=False, backend="cpu_mp",
        backend_kwargs={"n_workers": 2},
    )
    assert r.n_hits == 1
    assert r.hits[0].passphrase == "MyDog2014!"
    assert r.hits[0].address == target


def test_audit_unknown_backend_raises():
    with pytest.raises(ValueError):
        audit_passphrases(
            ["a", "b"], schemes=["brainwallet_sha256"],
            funded_set=FundedAddressSet(), backend="not_a_backend",
            run_diagnostic=False,
        )
