"""Round-3 expanded passphrase dictionary.

Targets categories that hit BEST in round 1:
  - Crypto/bitcoin phrases (501 BTC from 'bitcoin is awesome' alone)
  - Numeric weak strings ('1', '0', '12345', '42')
  - Top passwords ('password', 'master', 'admin')
  - Dev/test phrases ('test', 'blockchain')
  - Single words

Adds new categories with high historical hit rates:
  - Numeric-keyboard walks
  - Common 4+ word phrases (xkcd-style)
  - Wikipedia "1000 most common words" subset
  - Inspirational / motivational phrases
  - Standard CTF / hacker-culture phrases
  - Date patterns
  - Crypto-asset names + variations
"""

from __future__ import annotations


CRYPTO_EXPANDED = [
    # Bitcoin direct
    "free bitcoin", "free bitcoins", "send bitcoin", "send btc",
    "my bitcoin", "my bitcoins", "your bitcoin", "his bitcoin",
    "her bitcoin", "buy bitcoin", "buy bitcoins", "sell bitcoin",
    "i love bitcoin", "we love bitcoin", "bitcoin forever",
    "bitcoin mining", "bitcoin miner", "bitcoin millionaire",
    "bitcoin billionaire", "bitcoin trillion", "bitcoin king",
    "bitcoin queen", "bitcoin god", "bitcoin lord",
    "bitcoin is dead", "bitcoin is the future", "bitcoin is money",
    "bitcoin is god", "bitcoin will win", "bitcoin to a million",
    "bitcoin to 100k", "bitcoin to 1m", "bitcoin moon",
    "bitcoin mooning", "bitcoin lambo", "bitcoin to lambo",
    "bitcoin retire", "bitcoin retirement", "bitcoin freedom",
    "bitcoin pizza day", "10000 bitcoin", "10000 bitcoins",
    # Crypto-adjacent
    "ethereum", "ether", "eth", "vitalik", "buterin",
    "litecoin", "doge", "dogecoin", "shiba", "shiba inu",
    "much wow", "such btc", "to the moon", "wen moon", "wen lambo",
    "diamond hands", "paper hands", "ape", "apes", "ape strong",
    "wagmi", "ngmi", "gm", "gn", "fud", "fomo",
    "hodler", "hodling", "buy the rip", "rekt", "rekt city",
    # Wallets / exchanges
    "coinbase", "binance", "kraken", "gemini", "okx", "bitfinex",
    "bittrex", "poloniex", "huobi", "kucoin", "ftx", "ftxus",
    "trezor", "ledger", "ledger nano", "trezor model t",
    "electrum", "armory", "samourai", "blue wallet", "muun",
    "bitcoin core", "bitcoin core wallet", "metamask", "phantom",
    "rainbow", "argent",
    # Famous phrases from the lit
    "satoshi vision", "bitcoin sv", "bcash", "bitcoin cash",
    "lightning", "lightning network", "taproot", "segwit",
    "schnorr", "merkle tree", "merkle root", "block hash",
    "block 0", "block one", "genesis", "genesis block",
    "satoshis dream", "satoshis vision",
    # Numbers
    "21 million", "21000000", "21,000,000", "twenty one million",
    "100000000", "one hundred million sat", "1 satoshi", "one satoshi",
]

NUMERIC_PATTERNS = [
    # Phone keypad patterns
    "147258369", "159753", "258", "369", "147", "741",
    "2580", "01010101", "1379", "7531",
    "246810", "13579", "10293847",
    # Year patterns
    *[str(y) for y in range(1900, 2027)],
    # Date patterns
    "01/01/2000", "12/31/1999", "01/01/2024", "12/25/2023",
    "010101", "121212", "010199", "311299",
    # Bank/PIN style
    "0000", "1111", "1234", "9999", "0007", "007",
    "1004", "1212", "4321", "0420", "0911", "911",
    "2222", "3333", "4444", "5555", "6666", "7777", "8888",
    # Famous numbers
    "31415926", "27182818", "16180339", "9999999",
    "8675309",  # Jenny's number
    "867-5309", "1234567890",
    "112358", "11235813", "1123581321",  # Fibonacci
    # Crypto numbers
    "21000000", "1000000", "100000", "10000", "1000",
]

DEV_TEST_PHRASES = [
    "hello world", "hello, world", "Hello, World!",
    "lorem ipsum", "lorem ipsum dolor sit amet", "foo", "bar",
    "foobar", "foo bar", "baz", "foobarbaz",
    "this is a test", "this is only a test", "do not use",
    "delete me", "remove this", "dummy", "placeholder",
    "todo", "todo: fix", "fixme", "wip",
    "asdf", "asdfasdf", "qwer", "qwerqwer",
    "zxcv", "zxcvzxcv", "test1234", "test123",
    "demo", "demo password", "demo wallet",
    "default password", "changeme", "change me",
    "use the source", "use the source luke", "may the source be with you",
]

INSPIRATIONAL = [
    "just do it", "yes we can", "no fear", "carpe diem",
    "live laugh love", "live love laugh", "love conquers all",
    "follow your dreams", "never give up", "stay strong",
    "be the change", "be yourself", "be kind",
    "dream big", "think big", "make it happen",
    "i can do this", "i believe in myself",
    "everything happens for a reason",
    "what doesn't kill you makes you stronger",
    "this too shall pass", "keep calm and carry on",
    "keep calm", "keep going",
    "the best is yet to come", "im a survivor",
    "good vibes only", "good vibes", "positive vibes",
]

HACKER_CULTURE = [
    "1337", "leet", "h4x0r", "haxor", "hacker",
    "hack the planet", "hackthegibson", "hack the gibson",
    "there is no spoon", "follow the white rabbit",
    "wake up neo", "the matrix has you", "knock knock neo",
    "the cake is a lie", "all your base are belong to us",
    "0wned", "pwned", "pwn3d",
    "rm -rf /", "format c:", "DROP TABLE users;",
    "alert(1)", "console.log", "print('hello')",
    "ssh root@localhost", "sudo make me a sandwich",
    "iam root", "iamroot", "iam admin", "iamadmin",
    "give me root", "uid 0", "root access",
]

XKCD_4_WORD = [
    # The famous one plus variations on the same template
    "correct horse battery staple",
    "horse correct battery staple",
    "battery staple correct horse",
    "staple battery horse correct",
    "horse staple battery correct",
    "yellow purple horse battery",
    "purple yellow horse battery",
    "blue green dog table",
    "red orange cat chair",
    "common english random word",
    "passphrase like this is",
    "this is not my passphrase",
    "this is my passphrase",
    "my passphrase is secret",
    "open this wallet please",
    "open the box please",
    "give me my money back",
    "where is my money",
    "i lost my password",
    "i forgot my password",
    "what is my password",
    "this is my secret",
    "this is the secret",
    "i need my coins back",
]

RELIGIOUS_SPIRITUAL = [
    "god is good", "god is great", "god is love",
    "praise jesus", "praise the lord", "amen",
    "alleluia", "hallelujah", "shalom", "peace",
    "namaste", "om", "om mani padme hum",
    "allah", "allahu akbar", "inshallah",
    "blessed be", "blessed", "thank god",
    "thank god its friday", "tgif",
    "love and light", "love is the answer",
    "we are one", "i am one", "we are all one",
    "let it be", "go in peace",
]

FAMOUS_NAMES = [
    "albert einstein", "einstein", "isaac newton", "newton",
    "richard feynman", "feynman", "stephen hawking", "hawking",
    "nikola tesla", "tesla", "thomas edison", "edison",
    "leonardo da vinci", "da vinci", "michelangelo",
    "william shakespeare", "shakespeare",
    "abraham lincoln", "lincoln", "george washington",
    "barack obama", "obama", "donald trump", "trump",
    "elon musk", "musk", "steve jobs", "jobs",
    "bill gates", "gates", "mark zuckerberg", "zuckerberg",
    "jeff bezos", "bezos", "warren buffett", "buffett",
    "michael jordan", "kobe bryant", "lebron james",
    "tom brady", "lionel messi", "cristiano ronaldo",
    "taylor swift", "beyonce", "rihanna", "madonna",
    "elvis", "elvis presley", "michael jackson",
    "the beatles", "john lennon", "paul mccartney",
    "bob marley", "bob dylan",
]


def _vary(base: str) -> list[str]:
    out: set[str] = set()
    out.add(base)
    out.add(base.lower())
    out.add(base.upper())
    out.add(base.title())
    if " " in base:
        out.add(base.replace(" ", ""))
        out.add(base.replace(" ", "_"))
    out.add(base + "!")
    out.add(base + "1")
    return list(out)


def build_round3_candidates() -> list[str]:
    """Build round-3 dictionary including round-1 + round-2 + new tiers."""
    from _dictionary_round2 import build_round2_candidates

    out: list[str] = []
    seen: set[str] = set()
    # Start with round-2 base (which includes round-1)
    for p in build_round2_candidates():
        if p not in seen:
            seen.add(p)
            out.append(p)
    # Add round-3 tiers + variants
    new_sources = [
        CRYPTO_EXPANDED, NUMERIC_PATTERNS, DEV_TEST_PHRASES,
        INSPIRATIONAL, HACKER_CULTURE, XKCD_4_WORD,
        RELIGIOUS_SPIRITUAL, FAMOUS_NAMES,
    ]
    for src in new_sources:
        for p in src:
            if p not in seen:
                seen.add(p)
                out.append(p)
        for p in src:
            for v in _vary(p):
                if v not in seen:
                    seen.add(v)
                    out.append(v)
    return out


if __name__ == "__main__":
    cs = build_round3_candidates()
    print(f"round 3 candidate count: {len(cs)}")
