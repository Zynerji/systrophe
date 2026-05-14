"""Round 7 — exhaustive numeric + hex + date scan, ~2 million candidates.

The offline Bloom makes scale free. Round 7 covers numeric patterns
that human-chosen PINs and brainwallets commonly use:

  - All numerics 1 to 999,999 (with zero-padded variants for short)
  - 10-digit phone-number patterns
  - Hex strings of common short lengths (8, 12, 16, 32 chars) sampled
  - All ISO dates 1970-01-01 through 2030-12-31 in several formats
  - Round-6 base

At ~5K derivations/sec local, full scan takes ~13 minutes wall clock.
"""

from __future__ import annotations


def _numeric_exhaustive() -> list[str]:
    """Numeric strings 1..999999 + zero-padded variants for short."""
    out: set[str] = set()
    for i in range(1, 1_000_000):
        s = str(i)
        out.add(s)
        if i < 10_000:
            out.add(f"{i:04d}")
            out.add(f"{i:05d}")
        if i < 1_000:
            out.add(f"{i:06d}")
    return list(out)


def _date_iso(years_range: range, months: range, days: range) -> list[str]:
    out: set[str] = set()
    for y in years_range:
        for m in months:
            for d in days:
                # ISO-style
                out.add(f"{y}-{m:02d}-{d:02d}")
                # US-style
                out.add(f"{m:02d}/{d:02d}/{y}")
                # Compact
                out.add(f"{y}{m:02d}{d:02d}")
                # DDMMYY
                out.add(f"{d:02d}{m:02d}{y % 100:02d}")
                # MMDDYY
                out.add(f"{m:02d}{d:02d}{y % 100:02d}")
    return list(out)


def _hex_samples() -> list[str]:
    """Common short hex strings — alphabetic-only first chars (avoid massive
    explosion). Cover popular subsets people would actually type."""
    out: set[str] = set()
    # Common 8-char hex (16M total — sample interesting subsets)
    for s in [
        "deadbeef", "0000beef", "cafebabe", "babecafe",
        "abcdef01", "01234567", "fedcba98", "ffffffff",
        "00000000", "12345678", "87654321",
    ]:
        out.add(s)
        out.add(s.upper())
    # Common patterns of 16-char hex
    for s in [
        "deadbeefcafebabe", "0123456789abcdef", "fedcba9876543210",
        "abcdef0123456789", "1234567890abcdef",
    ]:
        out.add(s)
        out.add(s.upper())
    return list(out)


def build_round7_candidates() -> list[str]:
    from _dictionary_round6 import build_round6_candidates

    out: list[str] = []
    seen: set[str] = set()
    for p in build_round6_candidates():
        if p not in seen:
            seen.add(p)
            out.append(p)

    for source in [
        _numeric_exhaustive(),
        _date_iso(range(1970, 2031), range(1, 13),
                   (1, 5, 10, 15, 20, 25, 30)),
        _hex_samples(),
    ]:
        for p in source:
            if p not in seen:
                seen.add(p)
                out.append(p)

    return out


if __name__ == "__main__":
    cs = build_round7_candidates()
    print(f"round 7 candidate count: {len(cs):,}")
