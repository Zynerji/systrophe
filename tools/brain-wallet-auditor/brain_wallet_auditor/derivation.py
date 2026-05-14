"""Address derivation for Bitcoin brain-wallet schemes.

Three supported schemes:

  * `brainwallet_sha256` -- the historical / Vasek-paper-target form:
        priv = SHA256(passphrase)
    No KDF, no salt; catastrophic by modern standards.

  * `warpwallet` -- Castellucci 2014 design:
        priv = scrypt(passphrase, salt, ...) XOR pbkdf2(passphrase, salt, ...)
    Designed to be slow ($O(seconds)$ per derivation) so dictionary
    attacks are uneconomic. Salt is mandatory.

  * `bip39` -- modern BIP-39 mnemonic standard:
        seed = pbkdf2_hmac_sha512(mnemonic, "mnemonic" + passphrase,
                                   iterations=2048, dklen=64)
        priv = BIP32 derivation from seed at the requested path
    The default derivation path is m/44'/0'/0'/0/0 (BIP-44 first
    account, first external chain, first address).

All three produce a compressed-pubkey P2PKH address (the modern
convention starting with "1"). The historical (uncompressed-pubkey)
address can also be requested via `compressed=False` for repro of
2012-era brain-wallet drain studies.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Literal

import base58
import coincurve


Scheme = Literal["brainwallet_sha256", "warpwallet", "bip39"]


# ---------------------------------------------------------------------------
# Low-level: private key -> address
# ---------------------------------------------------------------------------


def _privkey_to_address(priv: bytes, compressed: bool = True) -> str:
    """Compressed (default) or uncompressed P2PKH mainnet address."""
    if len(priv) != 32:
        raise ValueError(f"private key must be 32 bytes, got {len(priv)}")
    key = coincurve.PrivateKey(priv)
    pub = key.public_key.format(compressed=compressed)
    sha = hashlib.sha256(pub).digest()
    ripe = hashlib.new("ripemd160", sha).digest()
    return base58.b58encode_check(b"\x00" + ripe).decode()


# ---------------------------------------------------------------------------
# Brainwallet (SHA256)
# ---------------------------------------------------------------------------


def brainwallet_address(passphrase: str, compressed: bool = True) -> str:
    """The classic / Vasek-paper form: address = SHA256-then-encode.

    Historically catastrophic: ~1,800 of the ~18,000 known brain
    wallets were drained within minutes of being funded
    (Vasek et al., 2016, "The Bitcoin Brain Drain").
    """
    priv = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return _privkey_to_address(priv, compressed=compressed)


# ---------------------------------------------------------------------------
# WarpWallet (Castellucci 2014)
# ---------------------------------------------------------------------------


def warpwallet_address(
    passphrase: str,
    salt: str = "",
    compressed: bool = True,
    scrypt_n: int = 2 ** 18,
    scrypt_r: int = 8,
    scrypt_p: int = 1,
    pbkdf2_iters: int = 2 ** 16,
) -> str:
    """WarpWallet (Castellucci, 2014): scrypt XOR PBKDF2.

    Slow by design; default parameters cost ~5-30 seconds per
    derivation on a modern CPU. This is the design choice that makes
    brute-force dictionary attacks uneconomic for properly-saltable
    pass-phrases.
    """
    pwd = passphrase.encode("utf-8") + b"\x01"
    salt_b = salt.encode("utf-8") + b"\x01"
    s = hashlib.scrypt(
        pwd, salt=salt_b, n=scrypt_n, r=scrypt_r, p=scrypt_p, dklen=32,
    )
    pwd2 = passphrase.encode("utf-8") + b"\x02"
    salt_b2 = salt.encode("utf-8") + b"\x02"
    p = hashlib.pbkdf2_hmac("sha256", pwd2, salt_b2, pbkdf2_iters, dklen=32)
    priv = bytes(a ^ b for a, b in zip(s, p))
    return _privkey_to_address(priv, compressed=compressed)


# ---------------------------------------------------------------------------
# BIP-39 mnemonic
# ---------------------------------------------------------------------------


def _bip32_master(seed: bytes) -> tuple[bytes, bytes]:
    """BIP-32 master node: (k_master, c_master) from a seed via
    HMAC-SHA512 with key 'Bitcoin seed'."""
    h = hmac.new(b"Bitcoin seed", seed, hashlib.sha512).digest()
    return h[:32], h[32:]


def _ckd_priv(k_par: bytes, c_par: bytes, index: int) -> tuple[bytes, bytes]:
    """BIP-32 child key derivation, private parent -> private child.

    Handles both hardened (index >= 0x80000000) and unhardened.
    """
    SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    if index >= 0x80000000:
        data = b"\x00" + k_par + index.to_bytes(4, "big")
    else:
        pub = coincurve.PrivateKey(k_par).public_key.format(compressed=True)
        data = pub + index.to_bytes(4, "big")
    h = hmac.new(c_par, data, hashlib.sha512).digest()
    I_L, c_child = h[:32], h[32:]
    n_par = int.from_bytes(k_par, "big")
    n_IL = int.from_bytes(I_L, "big")
    if n_IL >= SECP256K1_N:
        raise ValueError("BIP-32 derivation produced invalid child key (rare)")
    n_child = (n_par + n_IL) % SECP256K1_N
    if n_child == 0:
        raise ValueError("BIP-32 derivation produced zero child key (rare)")
    return n_child.to_bytes(32, "big"), c_child


def bip39_address(
    mnemonic: str,
    passphrase: str = "",
    derivation_path: str = "m/44'/0'/0'/0/0",
    compressed: bool = True,
) -> str:
    """Derive a mainnet address from a BIP-39 mnemonic + optional passphrase.

    Default path is BIP-44 first-account / first-external-chain /
    first-address. This is what Trezor / Ledger / most wallets use by
    default. Wallet-internal change addresses live at
    `m/44'/0'/0'/1/k`.

    No validation of the mnemonic's checksum or wordlist is done.
    Pass any string; if it's a valid 12/24-word seed the output is
    the correct address.
    """
    seed = hashlib.pbkdf2_hmac(
        "sha512", mnemonic.encode("utf-8"),
        ("mnemonic" + passphrase).encode("utf-8"),
        iterations=2048, dklen=64,
    )
    k, c = _bip32_master(seed)
    if derivation_path.startswith("m/"):
        derivation_path = derivation_path[2:]
    for part in derivation_path.split("/"):
        if not part:
            continue
        hardened = part.endswith("'")
        idx = int(part.rstrip("'"))
        if hardened:
            idx += 0x80000000
        k, c = _ckd_priv(k, c, idx)
    return _privkey_to_address(k, compressed=compressed)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


@dataclass
class AddressDerivation:
    """Configurable address derivation for one of the supported schemes.

    Construct once per scheme + parameters, call `derive(passphrase)`
    in the audit loop. This pre-bundles options so the audit loop is
    a single function call per candidate.
    """
    scheme: Scheme = "brainwallet_sha256"
    compressed: bool = True
    # WarpWallet
    warpwallet_salt: str = ""
    # BIP-39
    bip39_passphrase: str = ""
    bip39_path: str = "m/44'/0'/0'/0/0"

    def derive(self, passphrase: str) -> str:
        if self.scheme == "brainwallet_sha256":
            return brainwallet_address(passphrase, compressed=self.compressed)
        if self.scheme == "warpwallet":
            return warpwallet_address(
                passphrase, salt=self.warpwallet_salt,
                compressed=self.compressed,
            )
        if self.scheme == "bip39":
            return bip39_address(
                passphrase, passphrase=self.bip39_passphrase,
                derivation_path=self.bip39_path,
                compressed=self.compressed,
            )
        raise ValueError(f"unknown scheme: {self.scheme!r}")
