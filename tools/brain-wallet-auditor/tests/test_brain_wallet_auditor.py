"""Tests for brain-wallet-auditor. All synthetic; no blockchain access required."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brain_wallet_auditor import (
    AddressDerivation,
    AuditReport,
    AuditResult,
    FundedAddressSet,
    PoolDiagnostic,
    audit_passphrases,
    bip39_address,
    brainwallet_address,
    diagnose_candidate_pool,
)


# ---------------------------------------------------------------------------
# Brainwallet SHA256 (Vasek-paper-target form)
# ---------------------------------------------------------------------------


def test_brainwallet_compressed_known_value():
    """The "correct horse battery staple" passphrase, modern compressed-pubkey form."""
    addr = brainwallet_address("correct horse battery staple", compressed=True)
    assert addr == "1C7zdTfnkzmr13HfA2vNm5SJYRK6nEKyq8"


def test_brainwallet_uncompressed_historical_value():
    """The historical uncompressed-pubkey form (the address that was actually drained in 2013)."""
    addr = brainwallet_address("correct horse battery staple", compressed=False)
    assert addr == "1JwSSubhmg6iPtRjtyqhUYYH7bZg3Lfy1T"


def test_brainwallet_distinct_passphrases_give_distinct_addresses():
    a = brainwallet_address("password")
    b = brainwallet_address("Password")
    c = brainwallet_address("p@ssword")
    assert len({a, b, c}) == 3


def test_brainwallet_address_format():
    """Mainnet P2PKH addresses start with '1' and are 26-35 chars."""
    addr = brainwallet_address("anything")
    assert addr.startswith("1")
    assert 26 <= len(addr) <= 35


# ---------------------------------------------------------------------------
# BIP-39
# ---------------------------------------------------------------------------


def test_bip39_known_test_vector():
    """BIP-39 reference test vector (12 'abandon' + 'about').
    Verified independently via several wallet implementations: this
    mnemonic + path m/44'/0'/0'/0/0 yields the well-known address
    `1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA`.
    """
    mnemonic = "abandon abandon abandon abandon abandon abandon " \
               "abandon abandon abandon abandon abandon about"
    addr = bip39_address(mnemonic, passphrase="", derivation_path="m/44'/0'/0'/0/0")
    assert addr == "1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA"


def test_bip39_different_passphrase_gives_different_address():
    mnemonic = "abandon abandon abandon abandon abandon abandon " \
               "abandon abandon abandon abandon abandon about"
    a = bip39_address(mnemonic, passphrase="")
    b = bip39_address(mnemonic, passphrase="TREZOR")
    assert a != b


# ---------------------------------------------------------------------------
# FundedAddressSet (plain + Bloom)
# ---------------------------------------------------------------------------


def test_funded_set_plain():
    s = FundedAddressSet.from_iterable([
        "1JwSSubhmg6iPtRjtyqhUYYH7bZg3Lfy1T",
        "1C7zdTfnkzmr13HfA2vNm5SJYRK6nEKyq8",
    ])
    assert "1JwSSubhmg6iPtRjtyqhUYYH7bZg3Lfy1T" in s
    assert "1NotARealAddress0000000000000000" not in s
    assert len(s) == 2


def test_funded_set_bloom_no_false_negatives():
    """Bloom filter: must never return False on a member."""
    addresses = [f"1FakeAddr{i:024d}" for i in range(200)]
    s = FundedAddressSet.from_bloom(addresses, expected=200, fpr=1e-3)
    for a in addresses:
        assert a in s


def test_funded_set_bloom_low_fpr():
    """FPR should be approximately the requested rate."""
    addresses = [f"1Real{i:028d}" for i in range(500)]
    s = FundedAddressSet.from_bloom(addresses, expected=500, fpr=1e-3)
    # Test 1000 non-members
    fp = 0
    for i in range(1000):
        if f"1Bogus{i:027d}" in s:
            fp += 1
    # Empirical FPR < 5% to be safe (target 1e-3 = 0.1%)
    assert fp < 50, f"got {fp} false positives in 1000 lookups"


# ---------------------------------------------------------------------------
# AddressDerivation dispatch
# ---------------------------------------------------------------------------


def test_derivation_brainwallet_dispatch():
    d = AddressDerivation(scheme="brainwallet_sha256")
    assert d.derive("test") == brainwallet_address("test")


def test_derivation_bip39_dispatch():
    d = AddressDerivation(scheme="bip39")
    mnemonic = "abandon abandon abandon abandon abandon abandon " \
               "abandon abandon abandon abandon abandon about"
    assert d.derive(mnemonic) == bip39_address(mnemonic)


def test_derivation_unknown_scheme_raises():
    d = AddressDerivation(scheme="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        d.derive("test")


# ---------------------------------------------------------------------------
# audit_passphrases end-to-end
# ---------------------------------------------------------------------------


def test_audit_finds_planted_hit():
    """Audit finds the planted weak passphrase ('password') by deriving
    its address and matching it against a funded set we constructed."""
    target = brainwallet_address("password")
    funded = FundedAddressSet.from_iterable([target])
    candidates = ["password", "qwerty123", "hunter2", "MyDog2014!"]
    report = audit_passphrases(
        candidates, schemes=["brainwallet_sha256"],
        funded_set=funded, run_diagnostic=False,
    )
    assert isinstance(report, AuditReport)
    assert report.n_hits == 1
    hit = report.hits[0]
    assert hit.passphrase == "password"
    assert hit.address == target


def test_audit_returns_one_result_per_passphrase_per_scheme():
    candidates = ["a", "b"]
    schemes = ["brainwallet_sha256"]  # bip39 takes a real mnemonic; skip in batch
    report = audit_passphrases(
        candidates, schemes=schemes,
        funded_set=FundedAddressSet(), run_diagnostic=False,
    )
    assert report.n_passphrases == 2
    assert report.n_schemes == 1
    assert len(report.results) == 2


def test_audit_empty_funded_set_no_hits():
    report = audit_passphrases(
        ["a", "b", "c"], schemes=["brainwallet_sha256"],
        funded_set=FundedAddressSet(), run_diagnostic=False,
    )
    assert report.n_hits == 0


def test_audit_derivations_per_second_positive():
    report = audit_passphrases(
        ["a", "b", "c", "d", "e"], schemes=["brainwallet_sha256"],
        funded_set=FundedAddressSet(), run_diagnostic=False,
    )
    assert report.derivations_per_second > 0


# ---------------------------------------------------------------------------
# Catcher diagnostic
# ---------------------------------------------------------------------------


def test_diagnose_uniform_on_random_passphrase_pool():
    """Random unrelated passphrases should produce a 'uniform' verdict
    (the SHA256 of each passphrase produces uncorrelated addresses)."""
    import random
    rng = random.Random(0)
    pool = []
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    for _ in range(40):
        n = rng.randint(8, 16)
        pool.append("".join(rng.choices(chars, k=n)))
    deriv = AddressDerivation(scheme="brainwallet_sha256")
    diag = diagnose_candidate_pool(pool, deriv.derive)
    assert isinstance(diag, PoolDiagnostic)
    assert diag.n_candidates == 40
    # Should be uniform (or smooth at worst); strong novel_structure
    # on truly random passphrases would be a real KDF surprise.
    assert diag.verdict in {"uniform", "smooth"}


def test_diagnose_too_small_pool():
    diag = diagnose_candidate_pool(["a", "b"], brainwallet_address)
    assert diag.verdict == "too_small"


def test_audit_runs_diagnostic_by_default():
    """A multi-candidate audit returns a non-null PoolDiagnostic."""
    pool = [f"phrase_{i}" for i in range(20)]
    report = audit_passphrases(
        pool, schemes=["brainwallet_sha256"],
        funded_set=FundedAddressSet(),
    )
    assert report.pool_diagnostic is not None
    assert report.pool_diagnostic.n_candidates == 20
