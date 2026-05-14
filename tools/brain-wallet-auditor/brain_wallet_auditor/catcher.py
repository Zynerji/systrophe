"""Catcher diagnostic on a candidate-passphrase pool.

The Systrophe address-space lambda_2 catcher does NOT accelerate
brute-force dictionary attacks on hash functions. SHA256 + secp256k1
+ RIPEMD160 are spectrally flat by design.

What the catcher CAN do here:

  * On a pool of derived addresses from candidate passphrases, run
    `scan_novelty`. If the verdict is "uniform", the candidate pool
    has the expected spectral signature of random hash output --
    i.e. the KDF is doing its job and the dictionary attack is the
    only available approach.
  * If the verdict is "novel_structure", the candidate pool has
    spectral clustering. This means either (a) the candidate
    generator is producing correlated passphrases that map to
    related addresses (useful for prioritising further variations
    around the clusters), or (b) the KDF has a non-uniformity that's
    worth investigating (would be a security finding).

The diagnostic is a SANITY CHECK on the candidate pool, not a
search-acceleration technique. Treat its output as forensic
metadata, not as a key-recovery signal.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from systrophe.novelty_catcher import scan_novelty


@dataclass
class PoolDiagnostic:
    """Outcome of running the catcher on a passphrase-pool's derived addresses."""

    n_candidates: int
    verdict: str                            # "uniform" | "novel_structure" | "smooth"
    n_sharp_features: int
    sharp_clusters: list[dict]              # candidate-index ranges where the catcher flagged structure
    interpretation: str                     # plain-text guidance

    def __repr__(self) -> str:
        return (
            f"PoolDiagnostic(n={self.n_candidates}, verdict={self.verdict!r}, "
            f"sharp={self.n_sharp_features})"
        )


def _address_to_features(addr: str) -> np.ndarray:
    """Map a base58check Bitcoin address to a real-valued feature vector
    that the Systrophe catcher can hash.

    Uses SHA256(addr) and unpacks the 32 bytes to 32 floats. The
    catcher hashes these via its own rank-thermometer encoding.
    """
    h = hashlib.sha256(addr.encode("utf-8")).digest()
    return np.frombuffer(h, dtype=np.uint8).astype(float)


def diagnose_candidate_pool(
    passphrases: list[str],
    derive_fn,
    n_bits: int = 32,
    parameter_label: str = "candidate_idx",
) -> PoolDiagnostic:
    """Run the catcher across a pool of derived addresses.

    Parameters
    ----------
    passphrases
        Candidate passphrases (the audit pool).
    derive_fn
        Callable `passphrase -> address`. Pass e.g.
        `AddressDerivation(scheme="brainwallet_sha256").derive`.
    n_bits
        Catcher address width.
    """
    if len(passphrases) < 4:
        return PoolDiagnostic(
            n_candidates=len(passphrases),
            verdict="too_small",
            n_sharp_features=0,
            sharp_clusters=[],
            interpretation="Pool too small to run the catcher (need >= 4).",
        )

    addresses = [derive_fn(p) for p in passphrases]
    features = [_address_to_features(a) for a in addresses]
    indices = np.arange(len(passphrases), dtype=float)

    def fn(idx_float):
        i = int(round(idx_float))
        i = max(0, min(len(features) - 1, i))
        return features[i]

    scan = scan_novelty(
        indices, fn, n_bits=n_bits, parameter_label=parameter_label,
        data_adaptive=True,
    )

    clusters = []
    for s in scan.sharp_features:
        i_lo, i_hi = s["between_indices"]
        clusters.append({
            "between_candidate_indices": [int(i_lo), int(i_hi)],
            "passphrase_lo": passphrases[i_lo],
            "passphrase_hi": passphrases[i_hi],
            "address_lo": addresses[i_lo],
            "address_hi": addresses[i_hi],
            "hamming_step": int(s["hamming_step"]),
        })

    if scan.verdict == "uniform":
        interpretation = (
            "Candidate pool's derived addresses are spectrally uniform "
            "in address-space. This is the expected signature of a "
            "well-behaved KDF on random-looking passphrases. The "
            "catcher does NOT accelerate the dictionary attack here; "
            "the dictionary attack proceeds independent of this verdict."
        )
    elif scan.verdict == "novel_structure":
        interpretation = (
            "Candidate pool's derived addresses cluster in address-space. "
            "Either (a) the passphrase generator is producing correlated "
            "passphrases (in which case you can prioritise further "
            "variations around the flagged candidate-index neighbourhoods) "
            "or (b) you've found a KDF anomaly worth a separate look. "
            "Sharp clusters: see `sharp_clusters`."
        )
    else:
        interpretation = (
            f"Catcher verdict: {scan.verdict!r}. Pool likely too small or "
            f"too smooth to flag emergents."
        )

    return PoolDiagnostic(
        n_candidates=len(passphrases),
        verdict=scan.verdict,
        n_sharp_features=len(scan.sharp_features),
        sharp_clusters=clusters,
        interpretation=interpretation,
    )
