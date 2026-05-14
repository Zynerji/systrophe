"""Round 5 — combinatorial expansion: target ~200K candidates.

Once the offline Bloom is built, candidate-count is no longer
API-bound. We can scan 100K+ passphrases entirely locally and
only API-verify the (tiny number of) Bloom hits.

Sources combined:
  - All round 1-4 entries (~8K)
  - 2-word combinations of top 100 nouns + top 100 verbs/adjectives
  - 3-word phrases of the form ADJ-NOUN-VERB / DETERMINER-ADJ-NOUN
  - Date patterns (MM/DD/YYYY, YYYY-MM-DD, DDMMYY) at moderate density
  - Numeric-only at high density (1-9999 + zero-padded variants)
  - Crypto-themed combinations (verb + crypto-noun + suffix)
"""

from __future__ import annotations


TOP_NOUNS = [
    "money", "wallet", "bank", "vault", "safe", "treasure", "gold",
    "silver", "diamond", "ruby", "pearl", "key", "lock", "door",
    "house", "home", "car", "boat", "plane", "rocket",
    "love", "heart", "soul", "mind", "body", "spirit",
    "hope", "dream", "wish", "fate", "destiny", "future", "past",
    "fire", "water", "earth", "wind", "rain", "snow", "sun", "moon",
    "star", "sky", "cloud", "ocean", "river", "mountain", "forest",
    "city", "country", "world", "universe", "galaxy", "planet",
    "secret", "mystery", "answer", "question", "truth", "lie",
    "friend", "enemy", "lover", "stranger", "family", "father",
    "mother", "brother", "sister", "son", "daughter", "baby",
    "king", "queen", "prince", "princess", "lord", "lady", "god",
    "angel", "devil", "demon", "ghost", "spirit",
    "magic", "spell", "potion", "wand", "sword", "shield", "armor",
    "code", "cipher", "puzzle", "riddle", "mystery", "clue",
    "bitcoin", "crypto", "blockchain", "satoshi", "token",
    "wallet", "address", "private", "public",
]

TOP_VERBS = [
    "love", "live", "die", "fight", "run", "walk", "fly", "swim",
    "eat", "drink", "sleep", "dream", "wake", "rise", "fall",
    "make", "take", "give", "send", "buy", "sell", "trade",
    "win", "lose", "stop", "start", "begin", "end", "finish",
    "open", "close", "lock", "unlock", "find", "lost", "seek",
    "keep", "hold", "drop", "throw", "catch", "save", "spend",
    "hodl", "mine", "stack", "earn", "spend",
]

TOP_ADJS = [
    "big", "small", "fast", "slow", "good", "bad", "great", "evil",
    "best", "worst", "first", "last", "new", "old", "young", "ancient",
    "hot", "cold", "warm", "cool", "bright", "dark", "shiny", "dull",
    "happy", "sad", "angry", "calm", "lucky", "unlucky", "rich", "poor",
    "free", "open", "secret", "hidden", "lost", "found", "broken", "fixed",
    "silver", "gold", "diamond", "platinum", "iron", "steel",
    "lazy", "smart", "stupid", "wise", "kind", "cruel",
    "magic", "magical", "epic", "legendary", "rare", "common",
    "secret", "private", "public", "official", "real", "fake",
]

DETERMINERS = ["the", "my", "your", "his", "her", "our", "their"]

CRYPTO_NOUNS = [
    "bitcoin", "btc", "satoshi", "wallet", "coin", "coins", "address",
    "block", "blockchain", "ether", "eth", "doge", "shiba",
    "lambo", "moon", "diamond", "hands", "ape",
    "private", "public", "key", "keys", "seed", "phrase",
]

CRYPTO_VERBS = [
    "send", "buy", "sell", "trade", "stack", "hodl", "mine",
    "spend", "save", "burn", "stake",
]

DATE_PATTERNS = []
# Common birthdays etc
for y in range(1950, 2027):
    for m in (1, 5, 7, 12):
        for d in (1, 15, 25):
            DATE_PATTERNS.append(f"{m:02d}/{d:02d}/{y}")
            DATE_PATTERNS.append(f"{y}-{m:02d}-{d:02d}")
            DATE_PATTERNS.append(f"{d:02d}{m:02d}{y % 100:02d}")
            DATE_PATTERNS.append(f"{y}{m:02d}{d:02d}")

NUMERIC_ZERO_PAD = []
for i in range(1, 10_000):
    NUMERIC_ZERO_PAD.append(str(i))
    if i < 100:
        NUMERIC_ZERO_PAD.append(f"{i:02d}")
        NUMERIC_ZERO_PAD.append(f"{i:04d}")
    elif i < 1000:
        NUMERIC_ZERO_PAD.append(f"{i:04d}")
        NUMERIC_ZERO_PAD.append(f"{i:06d}")

# Common keyboard patterns expanded
KEYBOARD_PATTERNS = [
    "qwertyuiop", "qwertyuiop[]", "qwerty1234", "qwerty123",
    "qwerty12345", "qwerty123456", "qwerty1234567890",
    "asdfghjkl", "asdfghjkl;", "asdf1234", "asdfasdfasdf",
    "zxcvbnm", "zxcvbnm,./", "1qaz2wsx", "1qaz2wsx3edc",
    "1qaz2wsx3edc4rfv", "qazwsx", "qazwsxedc",
    "qazwsxedcrfv", "qazwsxedcrfvtgb",
    "147258369", "159753", "13579", "24680",
    "abcdef", "abcdefg", "abcdefgh", "abcdefghi", "abcdefghij",
    "0987654321", "9876543210",
    "qwerty!", "qwerty@", "QWERTY",
    "1234abcd", "abcd1234", "1q2w3e4r5t",
]


def _two_word_combos(words_a: list[str], words_b: list[str],
                       seps: tuple[str, ...] = (" ", "")) -> list[str]:
    out = []
    for a in words_a:
        for b in words_b:
            for sep in seps:
                out.append(f"{a}{sep}{b}")
    return out


def _three_word_combos(a_list: list[str], b_list: list[str],
                         c_list: list[str], sep: str = " ") -> list[str]:
    out = []
    for a in a_list:
        for b in b_list:
            for c in c_list:
                out.append(f"{a}{sep}{b}{sep}{c}")
    return out


def _maybe_vary(base: str, light: bool = False) -> list[str]:
    """Light variants to avoid combinatorial explosion."""
    out = {base}
    out.add(base.lower())
    if light:
        return list(out)
    out.add(base.upper())
    out.add(base.title())
    if " " in base:
        out.add(base.replace(" ", ""))
        out.add(base.replace(" ", "_"))
    out.add(base + "1")
    return list(out)


def build_round5_candidates() -> list[str]:
    from _dictionary_round4 import build_round4_candidates

    out: list[str] = []
    seen: set[str] = set()
    for p in build_round4_candidates():
        if p not in seen:
            seen.add(p)
            out.append(p)

    # 2-word combos (sparse top set to avoid 10K×10K explosion)
    short_nouns = TOP_NOUNS[:40]
    short_verbs = TOP_VERBS[:30]
    short_adjs = TOP_ADJS[:30]

    for combo_src in [
        _two_word_combos(short_adjs, short_nouns, seps=(" ", "")),
        _two_word_combos(short_verbs, short_nouns, seps=(" ", "")),
        _two_word_combos(DETERMINERS, short_nouns, seps=(" ", "")),
        _two_word_combos(short_nouns, short_verbs, seps=(" ",)),
        _two_word_combos(CRYPTO_VERBS, CRYPTO_NOUNS, seps=(" ", "")),
    ]:
        for p in combo_src:
            if p not in seen:
                seen.add(p)
                out.append(p)
            # Single light variant (lowercase already in p typically)
            up = p.title()
            if up not in seen:
                seen.add(up)
                out.append(up)

    # 3-word "DET ADJ NOUN" combinations (smaller subsets to keep size sane)
    for p in _three_word_combos(
        DETERMINERS[:5], short_adjs[:15], short_nouns[:15],
    ):
        if p not in seen:
            seen.add(p)
            out.append(p)

    # Date patterns (already include variants)
    for p in DATE_PATTERNS:
        if p not in seen:
            seen.add(p)
            out.append(p)

    # Numeric strings (already in many forms)
    for p in NUMERIC_ZERO_PAD:
        if p not in seen:
            seen.add(p)
            out.append(p)

    # Keyboard patterns + variants
    for p in KEYBOARD_PATTERNS:
        for v in _maybe_vary(p):
            if v not in seen:
                seen.add(v)
                out.append(v)

    return out


if __name__ == "__main__":
    cs = build_round5_candidates()
    print(f"round 5 candidate count: {len(cs)}")
