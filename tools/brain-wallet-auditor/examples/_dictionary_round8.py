"""Round 8 — maximal numeric + targeted phrase expansion (~10M candidates).

We've now established that the brain-wallet attack surface is
heavily saturated: 1.07M candidates produced only 2 new historical
finds totalling 0.0003 BTC. Round 8 tests whether ANY remaining
yield exists at the high-end of plausible candidate space:

  - 7-digit numerics: 1,000,000 .. 9,999,999  (~9M candidates)
  - Round-7 base
  - Specific bip39-like short mnemonic candidates

Local time estimate (5K addr/sec): ~67 min for 10M candidates
(20M derivations).
"""

from __future__ import annotations


def _seven_digits() -> list[str]:
    return [str(i) for i in range(1_000_000, 10_000_000)]


# Specific short bip-39 / mnemonic-like phrases that have shown up in
# brain-wallet research:
BIP39_LIKE = [
    "abandon", "ability", "able", "about", "above", "absent", "absorb",
    "abandon abandon", "abandon ability", "ability about",
    "abandon ability able about above absent absorb",
    "abandon abandon abandon",
    "abandon abandon abandon abandon",
    "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo",
    "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
    "abandon math mimic master filter",
    "all hat no cattle", "all bark no bite",
]


def build_round8_candidates() -> list[str]:
    from _dictionary_round7 import build_round7_candidates

    out: list[str] = []
    seen: set[str] = set()
    for p in build_round7_candidates():
        if p not in seen:
            seen.add(p)
            out.append(p)
    for s in BIP39_LIKE:
        if s not in seen:
            seen.add(s)
            out.append(s)
    for n in _seven_digits():
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


if __name__ == "__main__":
    cs = build_round8_candidates()
    print(f"round 8 candidate count: {len(cs):,}")
