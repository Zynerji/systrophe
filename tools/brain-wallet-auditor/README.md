# brain-wallet-auditor

**Passphrase strength auditor for Bitcoin brain wallets.**

Built on top of the Systrophe address-space λ₂ catcher — but the
catcher's role here is diagnostic, not search-accelerative. The
honest reference for the underlying attack model is

> Vasek, Bonneau, Castellucci, Keith, Moore (2016). "The Bitcoin
> Brain Drain: A Short Paper on the Use and Abuse of Bitcoin Brain
> Wallets." Financial Cryptography and Data Security.

The Vasek paper found ~18,000 brain wallets in the public blockchain,
~1,800 of which were drained within minutes of being funded — by
attackers running dictionary attacks against the trivial
`SHA256(passphrase)` derivation form.

This tool packages the attack model as a **passphrase strength
auditor** with three legitimate use cases:

1. **Personal-wallet recovery.** You remember most of a brain-wallet
   passphrase but not quite. Feed candidate variations. Examples:
   capitalisation, punctuation, leetspeak, accidental typos.
2. **Custodial / wallet-provider audit.** Verify that your accepted
   passphrases produce well-distributed addresses; verify your KDF
   choice (WarpWallet or BIP-39 with PBKDF2) is doing its job.
3. **Academic / red-team research.** Reproduce Vasek et al. on a
   current blockchain snapshot. Methodology open.

## What it is NOT

* It does NOT generate passphrases for sweeping the blockchain. You
  supply candidates; the tool derives addresses and reports.
* It does NOT crack BIP-39 mnemonics with proper entropy. PBKDF2 +
  secp256k1 keyspace ≈ 2^128 work; you don't get there with this
  tool.
* It does NOT crack secp256k1-with-CSPRNG keys. Bitcoin puzzles, raw
  CSPRNG-generated wallets, and properly-implemented modern wallets
  are out of scope.
* **It does NOT contain or imply Systrophe acceleration of brute force.**
  The catcher is spectrally flat against cryptographic hashes by
  design. The catcher's value-add here is reporting whether the
  *candidate pool you supply* has spectral clustering — useful for
  prioritising further variations, not for shortcutting key recovery.

## Ethical scope

This tool audits passphrases you supply. Using it against passphrases
you don't own to sweep funded wallets is theft and probably wire
fraud. The repo ships it as a personal-audit / research instrument.
Read the Vasek paper for the academic framing.

## Layout

```
brain-wallet-auditor/
├── README.md
├── brain_wallet_auditor/
│   ├── __init__.py
│   ├── derivation.py   brainwallet_sha256 / warpwallet / bip39 derivation
│   ├── snapshot.py     FundedAddressSet + minimal pure-Python Bloom filter
│   ├── catcher.py      diagnose_candidate_pool via systrophe.novelty_catcher
│   └── audit.py        audit_passphrases() main entry point
├── examples/
│   └── audit_personal_phrases.py
├── tests/
│   └── test_brain_wallet_auditor.py     19 tests, all offline
└── requirements.txt    coincurve + base58 (no GPU / Blackwell required)
```

## Quick start

```python
from brain_wallet_auditor import (
    FundedAddressSet, audit_passphrases, brainwallet_address,
)

# Suppose your real passphrase was "MySecretPassphrase"
truth_addr = brainwallet_address("MySecretPassphrase")
funded = FundedAddressSet.from_iterable([truth_addr])

candidates = [
    "my secret passphrase",
    "MySecretPassphrase",
    "my-secret-passphrase",
    "mysecretpassphrase",
    "my s3cret p4ssphr4se",
]

report = audit_passphrases(
    candidates,
    schemes=["brainwallet_sha256"],
    funded_set=funded,
)
print(report.n_hits)              # 1
for hit in report.hits:
    print(hit.passphrase, hit.address)
```

For BIP-39 audit (e.g. you wrote down a 12-word seed and you want to
verify it derives to a known address):

```python
from brain_wallet_auditor import bip39_address

mnemonic = ("abandon abandon abandon abandon abandon abandon "
            "abandon abandon abandon abandon abandon about")
addr = bip39_address(mnemonic, passphrase="", derivation_path="m/44'/0'/0'/0/0")
print(addr)   # 1LqBGSKuX5yYUonjxT5qGfpUsXKYYWeabA  (well-known test vector)
```

For WarpWallet:

```python
from brain_wallet_auditor import warpwallet_address

# Slow by design: ~5-30 seconds per derivation at default params.
addr = warpwallet_address("strong passphrase here", salt="your-salt")
```

## Performance

On a single laptop CPU thread (Windows 11, Python 3.12):

* `brainwallet_sha256`: ~1,200 derivations/sec (the Vasek-paper-target form)
* `bip39` (m/44'/0'/0'/0/0): ~80 derivations/sec (PBKDF2-SHA512 2048 iters dominates)
* `warpwallet`: ~0.05 derivations/sec at default params (scrypt N=2^18 is intentionally slow)

For 10K candidates of plain brainwallet_sha256, the audit completes
in 10 seconds. Larger pools scale linearly. Multi-core / GPU
parallelisation is straightforward (each derivation is independent).

### GPU kernels (optional, Blackwell)

For very large brainwallet sweeps (millions of candidates) the
`kernels/` subdir provides two independent GPU primitives:

* `kernels/triton_sha256.py` — Triton kernel for batched SHA-256 of
  L ≤ 55 byte inputs. ~136 M hashes/sec on RTX PRO 6000 Blackwell.
* `kernels/secp256k1_rs/` — Rust + cudarc (CUDA 13.x via NVRTC) kernel
  for batched scalar multiplication `pub = priv * G`. Sustained
  ~8.6 M keys/sec at batch ≥ 256 k. Verified bit-exact against
  libsecp256k1 on 1024 random keys.

The two are independent benchmark crates; the end-to-end Python
`audit_passphrases` API does not call them yet (PyO3 binding is the
remaining glue work).

## Funded-address snapshot

For real audits you'll want a snapshot of Bitcoin addresses that
currently have non-zero balance. Options:

1. **Bitcoin Core**: build a UTXO index from a full node. Export
   funded addresses to a text file.
2. **Block explorer APIs**: most expose a "rich list" or have bulk
   exports.
3. **Pre-built lists**: e.g. https://github.com/blockchair (public
   address dumps).

Load the list into `FundedAddressSet.from_bloom(addresses, expected=N)`
for memory-efficient lookup. The tool's pure-Python Bloom filter uses
~30 MB for 50M addresses at FPR 1e-4.

## Install

This tool ships as part of the Systrophe repo. Standalone install:

```bash
pip install -r tools/brain-wallet-auditor/requirements.txt
# coincurve, base58
PYTHONPATH=src:tools/brain-wallet-auditor python -c "from brain_wallet_auditor import audit_passphrases"
```

## License

MIT, inherited from the Systrophe parent package.
