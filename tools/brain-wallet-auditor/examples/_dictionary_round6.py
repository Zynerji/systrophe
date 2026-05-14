"""Round 6 — aggressive combinatorial expansion: target ~200K-500K candidates.

The offline-Bloom audit makes candidate count free. Optimise for
coverage of plausible-but-not-trivial passphrases:

  - Full crypto verb × noun × suffix combinatorial
  - Wider 2-word and 3-word combos
  - All capitalization variants of the base round-5 set
  - Comprehensive numeric (1-99999) + zero-padded forms
  - Word-plus-number-plus-symbol combinations
"""

from __future__ import annotations


WIDE_NOUNS = [
    "bitcoin", "btc", "satoshi", "wallet", "address", "coin", "coins",
    "money", "cash", "gold", "silver", "diamond", "treasure",
    "secret", "key", "lock", "vault", "safe", "bank",
    "love", "heart", "soul", "mind", "freedom", "dream",
    "moon", "star", "sun", "sky", "earth", "ocean",
    "king", "queen", "lord", "god", "angel", "devil",
    "fire", "water", "ice", "snow", "rain", "storm",
    "magic", "spell", "potion", "curse",
    "hodl", "lambo", "stack", "sats",
]

WIDE_VERBS = [
    "send", "buy", "sell", "trade", "save", "spend", "earn",
    "make", "take", "give", "get", "have", "hold", "keep",
    "love", "live", "die", "fight", "win", "lose",
    "run", "fly", "swim", "dance", "sing",
    "open", "close", "lock", "unlock", "find", "lost",
    "hodl", "stack", "mine", "stake", "burn",
]

WIDE_ADJS = [
    "good", "bad", "best", "worst", "first", "last", "new", "old",
    "big", "small", "fast", "slow", "free", "open", "secret",
    "lost", "found", "happy", "sad", "rich", "poor",
    "magic", "epic", "legendary", "rare", "common",
    "silver", "gold", "diamond", "platinum",
    "hot", "cold", "bright", "dark", "lucky", "unlucky",
]

WIDE_SUFFIXES = [
    "", "!", "1", "12", "123", "1234", "12345",
    "2020", "2021", "2022", "2023", "2024", "2025",
    "0", "00", "000", "999", "111", "777", "666",
    ".", "?", "!", "!!", "1!", "123!",
]

SEPARATORS = [" ", "", "_", "-"]


def _combos2(a_list, b_list, seps):
    return [f"{a}{sep}{b}" for a in a_list for sep in seps for b in b_list]


def _add_suffixes(words, suffixes):
    return [f"{w}{s}" for w in words for s in suffixes]


def build_round6_candidates() -> list[str]:
    from _dictionary_round5 import build_round5_candidates

    out: list[str] = []
    seen: set[str] = set()

    for p in build_round5_candidates():
        if p not in seen:
            seen.add(p)
            out.append(p)

    # 2-word combos (3 separator styles each)
    pools = [
        _combos2(WIDE_ADJS[:25], WIDE_NOUNS[:25], SEPARATORS[:3]),
        _combos2(WIDE_VERBS[:25], WIDE_NOUNS[:25], SEPARATORS[:3]),
        _combos2(WIDE_NOUNS[:25], WIDE_VERBS[:25], SEPARATORS[:3]),
    ]
    for pool in pools:
        for p in pool:
            if p not in seen:
                seen.add(p)
                out.append(p)
            # Title variant
            t = p.title()
            if t not in seen:
                seen.add(t)
                out.append(t)

    # WIDE base words + every suffix
    base_words = WIDE_NOUNS + WIDE_VERBS + WIDE_ADJS
    for p in _add_suffixes(base_words, WIDE_SUFFIXES):
        if p not in seen:
            seen.add(p)
            out.append(p)
        t = p.title()
        if t not in seen:
            seen.add(t)
            out.append(t)

    # Numeric 1-99999 (zero-padded common widths)
    for i in range(1, 100_000):
        s = str(i)
        if s not in seen:
            seen.add(s)
            out.append(s)
        if i < 10_000:
            s4 = f"{i:04d}"
            if s4 not in seen:
                seen.add(s4)
                out.append(s4)
        if i < 1000:
            s6 = f"{i:06d}"
            if s6 not in seen:
                seen.add(s6)
                out.append(s6)

    return out


if __name__ == "__main__":
    cs = build_round6_candidates()
    print(f"round 6 candidate count: {len(cs)}")
