"""Round-2 expanded passphrase dictionary.

Adds, on top of round-1 categories:
  - Extended top-passwords list (out to ~500)
  - Names + names+year combinations (common pattern)
  - Movie / book titles
  - Song titles
  - Reversed common words
  - l33t-speak transforms of top words
  - Polyglot common phrases (Spanish/French/German basics)
  - Famous Bitcoin/crypto personalities
  - Common keyboard walks

All from publicly-published weak-password research.
"""

from __future__ import annotations


EXTENDED_PASSWORDS = [
    "shadow1", "letmein1", "abc1234", "iloveyou1", "monkey1", "trustno11",
    "admin123", "qwerty1", "password!", "qwerty!", "letmein!",
    "Password1", "Password123", "Password!", "Password!1",
    "ncc1701", "ncc-1701", "starwars1", "darkside",
    "112233", "147258369", "654321", "159753", "246810",
    "iloveyou!", "myspace1", "amanda", "buster1", "cookie",
    "summer1", "winter1", "spring1", "autumn1",
    "Internet", "internet", "Welcome1", "welcome1",
    "trustme", "letmein2", "passw0rd1", "passw0rd!", "p@ssw0rd",
    "P@ssw0rd", "P@ssword", "P@ssword1", "P@ssword!", "Admin123",
    "qazwsxedc", "1q2w3e4r", "1q2w3e", "1qaz2wsx", "1qazxsw2",
    "qwertyui", "asdfasdf", "qweqwe", "qwer1234", "asdf1234",
    "zaq12wsx", "zaq1xsw2", "zxcvbnm1", "zxcvbn",
    "qwert", "asdfg", "qwerty12345", "asdfghj", "1234qwer",
    "password12", "Password12", "password!@#", "p@$$w0rd",
]

NAMES = [
    "john", "mary", "david", "robert", "michael", "william", "james",
    "joseph", "richard", "thomas", "charles", "christopher", "daniel",
    "matthew", "anthony", "donald", "mark", "paul", "steven", "andrew",
    "kenneth", "george", "joshua", "kevin", "brian", "edward", "ronald",
    "timothy", "jason", "jeffrey", "ryan", "jacob", "gary", "nicholas",
    "eric", "jonathan", "stephen", "larry", "justin", "scott", "brandon",
    "frank", "benjamin", "gregory", "samuel", "raymond", "patrick",
    "alexander", "jack", "dennis", "jerry", "tyler", "aaron", "henry",
    "douglas", "peter", "adam", "noah", "zachary", "kyle", "walter",
    "elizabeth", "patricia", "jennifer", "linda", "barbara", "susan",
    "jessica", "sarah", "karen", "nancy", "lisa", "betty", "helen",
    "sandra", "donna", "carol", "ruth", "sharon", "michelle", "laura",
    "kimberly", "deborah", "dorothy", "amy", "angela", "ashley", "brenda",
    "emma", "olivia", "ava", "isabella", "sophia", "charlotte",
]

NAME_YEAR_PATTERN = [
    f"{n}{y}" for n in [
        "john", "mary", "david", "michael", "robert", "jennifer",
        "ashley", "jessica", "matt", "sarah", "chris", "kevin",
    ] for y in ("1980", "1985", "1990", "1995", "2000", "2010")
]

MOVIE_TITLES = [
    "star wars", "the godfather", "the matrix", "inception",
    "fight club", "pulp fiction", "the dark knight",
    "forrest gump", "the shawshank redemption", "schindlers list",
    "back to the future", "raiders of the lost ark", "jaws",
    "the lord of the rings", "the fellowship of the ring",
    "the two towers", "the return of the king", "the hobbit",
    "the godfather part ii", "the avengers", "iron man",
    "spider man", "captain america", "thor", "black panther",
    "wonder woman", "the lion king", "frozen", "toy story",
    "finding nemo", "the incredibles", "monsters inc",
    "harry potter", "harry potter and the sorcerers stone",
    "harry potter and the chamber of secrets", "interstellar",
    "the prestige", "memento", "the social network", "joker",
    "parasite", "the silence of the lambs", "casablanca",
    "citizen kane", "gone with the wind",
]

SONG_TITLES = [
    "bohemian rhapsody", "stairway to heaven", "hotel california",
    "imagine", "let it be", "yesterday", "hey jude", "smells like teen spirit",
    "billie jean", "thriller", "beat it", "purple haze",
    "sweet child o mine", "november rain", "back in black",
    "born to run", "thunderstruck", "highway to hell",
    "another brick in the wall", "comfortably numb",
    "wonderwall", "dont stop believin", "livin on a prayer",
    "shake it off", "rolling in the deep", "single ladies",
    "uptown funk", "happy", "shape of you", "old town road",
    "blinding lights", "bad guy", "watermelon sugar",
    "i will always love you", "my heart will go on",
    "i wanna dance with somebody", "respect",
    "what's going on", "imagine", "let it go",
]

L33T_TRANSFORMS = []  # filled by transform of TOP_PASSWORDS
for base in [
    "password", "secret", "money", "bitcoin", "wallet", "letmein",
    "trustno1", "iloveyou", "monkey", "satoshi", "blockchain",
]:
    L33T_TRANSFORMS.append(base
        .replace("o", "0").replace("e", "3").replace("a", "4")
        .replace("i", "1").replace("s", "5"))
    L33T_TRANSFORMS.append(base
        .replace("o", "0").replace("i", "1"))
    L33T_TRANSFORMS.append(base.replace("a", "@"))

REVERSED = [
    "drowssap", "654321", "tercep", "yenom", "niocitb", "tellaw",
    "niemtel", "1ontsurt", "uoyevoli", "yeknom", "ihsotas",
]

POLYGLOT = [
    "contrasena", "contraseña", "mot de passe", "passwort",
    "kennwort", "geheim", "wachtwoord", "senha",
    "пароль", "密码", "パスワード", "암호",
    "hola", "bonjour", "guten tag", "gracias", "merci",
    "te amo", "je t'aime", "ich liebe dich", "tu eres", "te quiero",
    "amor", "amour", "liebe", "amore", "namor",
    "open sesame", "ouvre toi sesame", "abrete sesamo",
    "freedom", "liberte", "libertad", "freiheit",
]

CRYPTO_PEOPLE = [
    "vitalik", "vitalik buterin", "satoshi", "satoshi nakamoto",
    "hal finney", "nick szabo", "wei dai", "adam back",
    "andreas antonopoulos", "roger ver", "craig wright",
    "winklevoss", "winklevii", "cz binance", "sam bankman fried",
    "michael saylor", "elon musk", "jack dorsey",
    "max keiser", "stacy herbert",
]

KEYBOARD_WALKS = [
    "qwertyuiopasdfghjklzxcvbnm", "1qaz2wsx3edc", "1qaz!QAZ",
    "qweasd", "qweasdzxc", "asdzxc", "qwertasdfgzxcvb",
    "!@#$%^&*()", "1qazxsw23edc", "qwerty1234",
    "qwerty12345", "qwertyqwerty", "asdfasdf",
]


def build_round2_candidates() -> list[str]:
    """Round-2 dictionary, in addition to round-1 base."""
    from audit_real_snapshot_large import (
        TOP_PASSWORDS, CRYPTO_PHRASES, FAMOUS_QUOTES, XKCD_STYLE,
        BIP39_TESTS, _variants, build_candidates as build_round1,
    )

    out: list[str] = []
    seen: set[str] = set()

    # Round-1 base
    for p in build_round1():
        if p not in seen:
            seen.add(p)
            out.append(p)

    # Extended tiers
    new_sources = [
        EXTENDED_PASSWORDS, NAMES, NAME_YEAR_PATTERN,
        MOVIE_TITLES, SONG_TITLES, L33T_TRANSFORMS,
        REVERSED, POLYGLOT, CRYPTO_PEOPLE, KEYBOARD_WALKS,
    ]
    for src in new_sources:
        for p in src:
            if p not in seen:
                seen.add(p)
                out.append(p)
        # Variants of every new source
        for p in src:
            for v in _variants(p):
                if v not in seen:
                    seen.add(v)
                    out.append(v)

    return out


if __name__ == "__main__":
    cs = build_round2_candidates()
    print(f"round 2 candidate count: {len(cs)}")
