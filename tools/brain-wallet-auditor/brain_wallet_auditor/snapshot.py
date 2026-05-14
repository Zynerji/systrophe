"""Funded-address snapshot store.

For research and audit work, we need to test whether a derived
address is in a (large) set of known-funded Bitcoin addresses. Two
backends:

  * **`set`** -- a plain Python set, fine for thousands of addresses.
  * **`bloom`** -- a Bloom filter, fine for tens of millions with a
    configurable false-positive rate. ~30 MB for 50M addresses at
    FPR 1e-4. Bloom-filter false positives must be confirmed via an
    exact-match second pass against an authoritative source (e.g.
    Bitcoin Core's `getaddressinfo` or a queryable index node).

In testing we use `set`. In production an analyst typically loads a
Bloom filter built from a blockchain snapshot.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable


class FundedAddressSet:
    """Container for the set of funded addresses to look candidates up in.

    Default backend is a Python `set`. Use `.from_bloom(...)` for a
    Bloom-filter-backed set.
    """

    def __init__(self, addresses: Iterable[str] = ()) -> None:
        self._addrs: set[str] = set(addresses)
        self._bloom: _Bloom | None = None
        self._bloom_only: bool = False  # True -> skip exact-set check
        self._bloom_only_count: int = 0

    def __contains__(self, address: str) -> bool:
        if self._bloom is not None:
            if self._bloom_only:
                # Bloom-only mode: may return false positives at the
                # configured fpr; verify externally before acting on hits.
                return address in self._bloom
            return address in self._bloom and address in self._addrs
        return address in self._addrs

    def __len__(self) -> int:
        if self._bloom_only:
            return self._bloom_only_count
        return len(self._addrs)

    def add(self, address: str) -> None:
        if not self._bloom_only:
            self._addrs.add(address)
        if self._bloom is not None:
            self._bloom.add(address)
            if self._bloom_only:
                self._bloom_only_count += 1

    def add_many(self, addresses: Iterable[str]) -> None:
        for a in addresses:
            self.add(a)

    @classmethod
    def from_bloom_only(cls, bloom: "_Bloom", n_added: int) -> "FundedAddressSet":
        """Wrap a pre-built _Bloom in Bloom-only mode.

        Use when the underlying snapshot is too large to hold in a
        Python set alongside. `__contains__` will then admit
        false positives at the Bloom's configured fpr; callers must
        verify hits externally (e.g. via a Bitcoin node) before
        acting on them.
        """
        s = cls()
        s._bloom = bloom
        s._bloom_only = True
        s._bloom_only_count = int(n_added)
        return s

    @classmethod
    def from_iterable(cls, addresses: Iterable[str]) -> "FundedAddressSet":
        return cls(addresses=addresses)

    @classmethod
    def from_bloom(
        cls,
        addresses: Iterable[str],
        expected: int,
        fpr: float = 1e-4,
    ) -> "FundedAddressSet":
        """Build a Bloom-filter-backed funded set.

        Parameters
        ----------
        addresses
            Iterable of addresses to add.
        expected
            Expected (or upper-bound) number of addresses you will add.
            Determines the Bloom-filter size.
        fpr
            Desired false-positive rate (e.g. 1e-4 means 0.01%).
        """
        s = cls()
        s._bloom = _Bloom(expected_n=expected, fpr=fpr)
        s.add_many(addresses)
        return s


# ---------------------------------------------------------------------------
# Minimal Bloom filter (no external deps)
# ---------------------------------------------------------------------------


@dataclass
class _Bloom:
    """Standard counting-free Bloom filter using k SHA-256 derived hashes.

    Memory: m bits, rounded up to a multiple of 8. m and k are
    computed to satisfy `fpr` at `expected_n` items.
    """
    expected_n: int
    fpr: float = 1e-4
    bits: bytearray = field(init=False)
    m: int = field(init=False)
    k: int = field(init=False)

    def __post_init__(self) -> None:
        if self.expected_n <= 0:
            raise ValueError("expected_n must be > 0")
        if not (0.0 < self.fpr < 1.0):
            raise ValueError("fpr must be in (0, 1)")
        # Optimal m, k for given n and fpr
        m = -self.expected_n * math.log(self.fpr) / (math.log(2) ** 2)
        m = max(8, int(math.ceil(m / 8)) * 8)
        self.m = m
        self.k = max(1, int(round((m / self.expected_n) * math.log(2))))
        self.bits = bytearray(self.m // 8)

    def _hash(self, item: str) -> Iterable[int]:
        # Generate k bit-indices via double-hashing of SHA-256.
        h = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = int.from_bytes(h[:16], "big")
        h2 = int.from_bytes(h[16:], "big")
        for i in range(self.k):
            yield (h1 + i * h2) % self.m

    def add(self, item: str) -> None:
        for idx in self._hash(item):
            self.bits[idx // 8] |= (1 << (idx % 8))

    def __contains__(self, item: str) -> bool:
        for idx in self._hash(item):
            if not (self.bits[idx // 8] & (1 << (idx % 8))):
                return False
        return True
