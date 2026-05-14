"""brain-wallet-auditor: passphrase strength auditor for Bitcoin brain wallets.

Three legitimate use cases:

  1. **Personal-wallet recovery.** You remember most of a brain-wallet
     passphrase but not quite. Feed candidate variations; the tool
     checks each. You audit your own keys.
  2. **Custodial audit.** A wallet provider verifies that the
     passphrases / KDFs they accept don't admit weak derived
     addresses. The Systrophe catcher's verdict on the derived-address
     pool is the positive signal.
  3. **Academic / red-team research.** Reproduce the Vasek et al.
     (2016) "Bitcoin Brain Drain" methodology on a current blockchain
     snapshot. Methodology is open and published.

What it does NOT do:

  * It does not generate passphrases for sweeping the blockchain.
    You supply candidate passphrases (a dictionary, your own
    variations, a research wordlist); the tool derives addresses
    and reports.
  * It does not crack BIP-39 mnemonics with reasonable entropy.
    The KDF chain holds; a dictionary attack on 12-word seeds at
    full entropy finds nothing.
  * It does not crack secp256k1-with-CSPRNG keys (Bitcoin puzzles,
    raw-PRNG wallets). The keyspace is too large.

Provenance: the address-space catcher diagnostic uses
`systrophe.novelty_catcher.scan_novelty`. The cryptographic pipeline
(SHA256 -> secp256k1 -> hash160 -> base58check) is straightforward
Bitcoin reference math, implemented on top of `coincurve` and
`base58`. The Vasek et al. (2016) "Bitcoin Brain Drain" paper is the
academic reference for the attack model.

Ethical scope: this tool is for auditing passphrases YOU supply.
Operating it against passphrases you don't own to sweep funded
addresses is theft and probably wire fraud in your jurisdiction.
The repository ships it as a research / personal-audit instrument.
"""

from .backends import (
    CPUMultiprocessingBackend,
    CPUSingleBackend,
    CPUThreadsBackend,
    DerivationBackend,
    GPUCudaBackend,
    get_backend,
)
from .derivation import (
    AddressDerivation,
    bip39_address,
    brainwallet_address,
    warpwallet_address,
)
from .snapshot import FundedAddressSet
from .audit import (
    AuditReport,
    AuditResult,
    audit_passphrases,
)
from .catcher import (
    PoolDiagnostic,
    diagnose_candidate_pool,
)

__all__ = [
    "AddressDerivation",
    "AuditReport",
    "AuditResult",
    "CPUMultiprocessingBackend",
    "CPUSingleBackend",
    "CPUThreadsBackend",
    "DerivationBackend",
    "FundedAddressSet",
    "GPUCudaBackend",
    "PoolDiagnostic",
    "audit_passphrases",
    "bip39_address",
    "brainwallet_address",
    "warpwallet_address",
    "diagnose_candidate_pool",
    "get_backend",
]

__version__ = "0.1.0"
