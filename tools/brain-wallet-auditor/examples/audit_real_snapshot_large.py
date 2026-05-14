"""Large-scale live-blockchain audit: ~2500 weak passphrases, parallel queries.

Methodology — Vasek 2016 reproduction at scale
----------------------------------------------
1. Build a candidate dictionary from publicly-published weak-password
   research (top passwords, common phrases, Bitcoin-themed weak
   strings, capitalisation/punctuation variants).
2. Derive both compressed and uncompressed brainwallet addresses
   locally (cpu_threads backend; ~hundreds of derivations/sec
   wall-clock since this is the brainwallet_sha256 GIL-bound path).
3. Query blockstream.info's free public API for each derived
   address, IN PARALLEL with a small thread pool (4 workers) to stay
   well below abuse thresholds.
4. Tally: per-category hit rate, total lifetime BTC funneled, any
   currently-funded balances.

Ethical scope
-------------
Read-only blockchain queries against publicly-published weak
passphrases. No private keys exported, no transactions constructed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brain_wallet_auditor import brainwallet_address


# ---------------------------------------------------------------------------
# Candidate dictionary
# ---------------------------------------------------------------------------


# Tier 1: classic weak passwords (top of every leaked-password dump).
TOP_PASSWORDS = [
    "password", "123456", "12345678", "1234", "qwerty", "12345",
    "111111", "1234567", "abc123", "letmein", "iloveyou", "monkey",
    "trustno1", "1234567890", "dragon", "baseball", "football", "shadow",
    "master", "michael", "superman", "696969", "123123", "batman",
    "0", "1", "a", "test", "guest", "admin", "root", "secret",
    "password1", "password123", "qwerty123", "welcome", "login", "passw0rd",
    "starwars", "freedom", "whatever", "qazwsx", "mustang", "harley",
    "ranger", "hunter", "buster", "thomas", "robert", "soccer",
    "killer", "george", "sexy", "andrew", "charlie", "superman1",
    "asshole", "fuckyou", "dallas", "jessica", "panties", "pepper",
    "1111", "austin", "william", "daniel", "golfer", "summer",
    "heather", "hammer", "yankees", "joshua", "maggie", "biteme",
    "enter", "ashley", "thunder", "cowboy", "silver", "richard",
    "fucker", "orange", "merlin", "michelle", "corvette", "bigdog",
    "cheese", "matthew", "121212", "patrick", "martin", "freedom1",
    "ginger", "blowjob", "nicole", "sparky", "yellow", "camaro",
    "secret1", "dick", "falcon", "taylor", "111111a", "131313",
    "jordan", "jennifer", "zxcvbnm", "asdfgh", "gandalf", "computer",
    "qwertyuiop", "asdfghjkl", "zxcvbnm123", "qaz", "wsx", "edc",
    "rfv", "tgb", "yhn", "ujm", "ik", "ol",
]

# Tier 2: Bitcoin / crypto / blockchain-themed
CRYPTO_PHRASES = [
    "bitcoin", "satoshi", "blockchain", "ethereum", "litecoin",
    "to the moon", "hodl", "hodl hodl", "satoshi nakamoto",
    "bitcoin is awesome", "buy the dip", "diamond hands",
    "not your keys not your coins", "free bitcoin", "bitcoin for everyone",
    "mt gox", "mtgox", "silk road", "wallet", "private key",
    "crypto", "mining", "block reward", "proof of work", "halving",
    "bitcoin pizza", "10000 bitcoins", "lambo", "when moon",
    "btc", "eth", "ltc", "xrp", "doge", "shiba",
    "blockchain rocks", "decentralize", "decentralized", "trustless",
    "bitcoin to the moon", "stay humble stack sats", "stack sats",
    "i am satoshi", "we are all satoshi", "this is the way",
    "drop the gold", "fiat is dying", "bitcoin will save us",
    "1 bitcoin", "21 million", "21000000",
    "genesis block", "bitcoin core", "lightning network", "segwit",
    "21 million coins", "deflationary", "sound money",
]

# Tier 3: famous quotes / lyrics / culture (well-known + brief)
FAMOUS_QUOTES = [
    "to be or not to be",
    "to be or not to be, that is the question",
    "the quick brown fox jumps over the lazy dog",
    "i have a dream",
    "may the force be with you",
    "use the force",
    "do or do not there is no try",
    "say hello to my little friend",
    "i'll be back",
    "ill be back",
    "you talkin to me",
    "you talking to me",
    "frankly my dear i don't give a damn",
    "houston we have a problem",
    "show me the money",
    "life is like a box of chocolates",
    "with great power comes great responsibility",
    "i am your father",
    "luke i am your father",
    "rosebud",
    "here's looking at you kid",
    "go ahead make my day",
    "never gonna give you up",
    "never gonna let you down",
    "all your base are belong to us",
    "for the horde",
    "leeroy jenkins",
    "winter is coming",
    "valar morghulis",
    "you shall not pass",
    "one ring to rule them all",
    "live long and prosper",
    "beam me up scotty",
    "houston the eagle has landed",
    "thats one small step for man",
    "i think therefore i am",
    "cogito ergo sum",
    "let them eat cake",
    "et tu brute",
    "veni vidi vici",
    "carpe diem",
    "memento mori",
    "ad astra per aspera",
    "que sera sera",
    "c'est la vie",
    "the answer is 42",
    "42",
    "dont panic",
    "so long and thanks for all the fish",
    "the meaning of life the universe and everything",
]

# Tier 4: xkcd-936-style multi-word passphrases (4 common words)
# These are the FAMOUS published ones; not generative.
XKCD_STYLE = [
    "correct horse battery staple",
    "Tr0ub4dor&3",
    "common horse battery staple",
    "purple horse battery staple",
    "horse battery staple correct",
    "battery horse staple correct",
    "this is fun",
    "i love bitcoin",
    "give me your money",
    "my brain wallet",
    "brain wallet",
    "warp wallet",
    "trezor",
    "ledger",
    "electrum",
    "passphrase",
    "the wallet",
    "open sesame",
    "abracadabra",
    "abrakadabra",
    "alakazam",
    "shazam",
    "expecto patronum",
    "expelliarmus",
    "wingardium leviosa",
    "alohomora",
    "avada kedavra",
    "mischief managed",
    "i solemnly swear",
    "im a wizard harry",
]

# Tier 5: bip-39 reference test vectors + famously-tested mnemonics
BIP39_TESTS = [
    "abandon abandon abandon abandon abandon abandon abandon abandon "
    "abandon abandon abandon about",
    "legal winner thank year wave sausage worth useful legal winner "
    "thank yellow",
    "letter advice cage absurd amount doctor acoustic avoid letter "
    "advice cage above",
    "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
]


def _variants(base: str) -> list[str]:
    """Generate canonical capitalization + punctuation variants."""
    out: set[str] = set()
    out.add(base)
    out.add(base.lower())
    out.add(base.upper())
    out.add(base.capitalize())
    out.add(base.title())
    # Strip / add spaces and basic punctuation
    if " " in base:
        out.add(base.replace(" ", ""))
        out.add(base.replace(" ", "_"))
        out.add(base.replace(" ", "-"))
    # Common trailing
    out.add(base + "!")
    out.add(base + ".")
    out.add(base + "1")
    return list(out)


def build_candidates() -> list[str]:
    """Build a deduplicated candidate list ~2500 entries."""
    seen: set[str] = set()
    out: list[str] = []
    # Direct tiers
    for src in [TOP_PASSWORDS, CRYPTO_PHRASES, FAMOUS_QUOTES, XKCD_STYLE,
                BIP39_TESTS]:
        for p in src:
            if p not in seen:
                seen.add(p)
                out.append(p)
    # Variants of TOP_PASSWORDS + CRYPTO_PHRASES (the highest-hit-rate tiers).
    for p in TOP_PASSWORDS + CRYPTO_PHRASES + XKCD_STYLE:
        for v in _variants(p):
            if v not in seen:
                seen.add(v)
                out.append(v)
    # Numeric range
    for i in range(1, 200):
        s = str(i)
        if s not in seen:
            seen.add(s)
            out.append(s)
    # Years
    for y in range(1900, 2027):
        s = str(y)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Concurrent live blockchain query
# ---------------------------------------------------------------------------


_session_lock = threading.Lock()
_progress = {"done": 0, "n_funded": 0, "n_currently_funded": 0,
             "total_lifetime_sat": 0, "total_current_sat": 0}


def fetch_one(addr: str, timeout: float = 20.0) -> dict | None:
    req = urllib.request.Request(
        f"https://blockstream.info/api/address/{addr}",
        headers={"User-Agent": "systrophe-brain-wallet-auditor/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def fetch_one_stats(addr: str) -> dict:
    raw = fetch_one(addr)
    if raw is None:
        return {"addr": addr, "ok": False, "funded": 0, "spent": 0,
                "balance": 0, "n_tx": 0}
    chain = raw.get("chain_stats", {})
    mempool = raw.get("mempool_stats", {})
    funded = chain.get("funded_txo_sum", 0) + mempool.get("funded_txo_sum", 0)
    spent = chain.get("spent_txo_sum", 0) + mempool.get("spent_txo_sum", 0)
    return {
        "addr": addr, "ok": True,
        "funded": funded, "spent": spent, "balance": funded - spent,
        "n_tx": chain.get("tx_count", 0) + mempool.get("tx_count", 0),
    }


def process_passphrase(passphrase: str, total: int) -> list[dict]:
    """Derive compressed + uncompressed addresses and query both."""
    rows: list[dict] = []
    for compressed in (True, False):
        addr = brainwallet_address(passphrase, compressed=compressed)
        st = fetch_one_stats(addr)
        st["passphrase"] = passphrase
        st["compressed"] = compressed
        rows.append(st)
        with _session_lock:
            _progress["done"] += 1
            if st["funded"] > 0:
                _progress["n_funded"] += 1
                _progress["total_lifetime_sat"] += st["funded"]
            if st["balance"] > 0:
                _progress["n_currently_funded"] += 1
                _progress["total_current_sat"] += st["balance"]
            done = _progress["done"]
            if done % 50 == 0 or done == total:
                print(f"  [{done:>5}/{total}]  "
                      f"funded_ever={_progress['n_funded']}  "
                      f"currently_funded={_progress['n_currently_funded']}  "
                      f"total_lifetime={_progress['total_lifetime_sat']/1e8:.4f} BTC")
    return rows


def main():
    candidates = build_candidates()
    print(f"Brain-wallet large audit -- LIVE Bitcoin blockchain")
    print("=" * 78)
    print(f"  {len(candidates)} candidate passphrases")
    print(f"  Each produces 2 addresses (compressed + uncompressed)")
    print(f"  Total queries to blockstream.info: {2 * len(candidates)}")
    print(f"  Parallelism: 4 threads (polite)")
    print()
    print("Progress (every 50 addresses):")
    total = 2 * len(candidates)

    all_rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process_passphrase, p, total) for p in candidates]
        for fut in as_completed(futures):
            try:
                all_rows.extend(fut.result())
            except Exception as e:
                print(f"  !! worker error: {e}")
    elapsed = time.time() - t0

    # Summary
    n_total = len(all_rows)
    n_ok = sum(1 for r in all_rows if r["ok"])
    n_funded = sum(1 for r in all_rows if r["ok"] and r["funded"] > 0)
    n_curr = sum(1 for r in all_rows if r["ok"] and r["balance"] > 0)
    total_lifetime = sum(r["funded"] for r in all_rows if r["ok"])
    total_current = sum(r["balance"] for r in all_rows if r["ok"])
    print()
    print("=" * 78)
    print(f"  Wall clock:                      {elapsed:.1f}s "
          f"({n_total/elapsed:.1f} addresses/sec)")
    print(f"  Addresses queried:               {n_total}")
    print(f"  Successful API responses:        {n_ok}")
    print(f"  Addresses EVER funded:           {n_funded}")
    print(f"  Addresses CURRENTLY funded:      {n_curr}")
    print(f"  Total lifetime BTC funneled:     {total_lifetime/1e8:.4f} BTC "
          f"({total_lifetime:,} sat)")
    print(f"  Currently recoverable balance:   {total_current/1e8:.6f} BTC "
          f"({total_current:,} sat)")
    print()

    # Top 20 lifetime-funded
    funded_rows = sorted(
        [r for r in all_rows if r["ok"] and r["funded"] > 0],
        key=lambda r: r["funded"], reverse=True,
    )
    print(f"Top {min(20, len(funded_rows))} addresses by lifetime funded:")
    for r in funded_rows[:20]:
        c = "C" if r["compressed"] else "U"
        print(f"  [{c}] {r['addr']:<38} {r['funded']/1e8:>10.4f} BTC "
              f"({r['n_tx']:>5} tx)  <- {r['passphrase']!r}")
    print()

    # Currently-funded addresses (the recovery prizes if you own them)
    if n_curr > 0:
        print(f"CURRENTLY FUNDED ({n_curr}) -- private key is "
              f"SHA256(passphrase) for these:")
        for r in all_rows:
            if r["ok"] and r["balance"] > 0:
                c = "C" if r["compressed"] else "U"
                print(f"  [{c}] {r['addr']:<38} {r['balance']/1e8:>10.6f} BTC"
                      f"  <- {r['passphrase']!r}")

    out_path = pathlib.Path(__file__).parent / "audit_real_snapshot_large_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_passphrases": len(candidates),
            "n_addresses": n_total,
            "n_successful_queries": n_ok,
            "n_ever_funded": n_funded,
            "n_currently_funded": n_curr,
            "total_lifetime_funded_sat": total_lifetime,
            "total_currently_funded_sat": total_current,
            "wall_clock_seconds": elapsed,
            "results": all_rows,
        }, f, indent=2)
    print(f"\nFull JSON: {out_path}")


if __name__ == "__main__":
    main()
