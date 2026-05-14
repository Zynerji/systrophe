"""Round 4 — longer/obscurer phrases sweepers might not have run.

Round 1+2+3 found 123 ever-funded but 0 currently-funded. That's
exactly as Vasek 2016 documented: weak-passphrase brain wallets are
monitored 24/7 by sweepers and drained within minutes of funding.
The "trivial" dictionary space is saturated.

Round 4 expands into:
  - Long famous quotes (40+ characters — beyond typical sweeper dicts)
  - Major foreign-language phrases (Spanish/French/German/Italian)
  - Niche subculture references (Star Trek, LOTR, Hitchhiker's Guide)
  - Common short combinations (top words + year)
  - Common Bitcoin-era phrases from blogs / forums (2010-2014)
"""

from __future__ import annotations


LONG_QUOTES = [
    "the only way out is through",
    "the only thing we have to fear is fear itself",
    "ask not what your country can do for you ask what you can do for your country",
    "i came i saw i conquered",
    "to err is human to forgive divine",
    "give me liberty or give me death",
    "we shall fight on the beaches",
    "this is the end of the beginning",
    "we have nothing to fear but fear itself",
    "the buck stops here",
    "the only easy day was yesterday",
    "what does not kill me makes me stronger",
    "elementary my dear watson",
    "houston we have a problem",
    "i could have been a contender",
    "go ahead make my day",
    "may the road rise up to meet you",
    "all that glitters is not gold",
    "to thine own self be true",
    "the only thing necessary for the triumph of evil",
    "out damned spot out i say",
    "what fools these mortals be",
    "let me count the ways",
    "shall i compare thee to a summers day",
    "im going to make him an offer he cant refuse",
    "you cant handle the truth",
    "round up the usual suspects",
    "the stuff that dreams are made of",
    "the force is strong with this one",
    "do or do not there is no try",
    "size matters not",
    "great kid dont get cocky",
    "i have a bad feeling about this",
    "these aren't the droids you're looking for",
    "i am one with the force the force is with me",
    "live long and prosper",
    "space the final frontier",
    "make it so",
    "engage",
    "resistance is futile",
    "the needs of the many outweigh the needs of the few",
    "klaatu barada nikto",
    "klaatu barada necktie",
    "klaatu barada necktie nikto",
    "fly you fools",
    "even the smallest person can change the course of the future",
    "not all those who wander are lost",
    "all we have to decide is what to do with the time that is given us",
    "my precious",
    "the ring is mine",
    "they have a cave troll",
    "the road goes ever on and on",
    "it is a far far better thing that i do",
    "it was the best of times it was the worst of times",
    "call me ishmael",
    "in the beginning",
    "in the beginning was the word",
    "this is the dawning of the age of aquarius",
    "imagine all the people living life in peace",
    "all we are saying is give peace a chance",
    "thank you for the music",
    "we will rock you",
    "we are the champions",
    "another one bites the dust",
    "fly me to the moon",
    "my way",
    "i did it my way",
    "i love rock and roll",
    "born in the usa",
    "born to run",
    "bohemian rhapsody is no big deal",
    "is this the real life is this just fantasy",
    "mama just killed a man",
    "galileo galileo galileo figaro",
    "we will we will rock you",
]

FOREIGN_PHRASES = [
    # Spanish — top common phrases
    "hola mundo", "buenos dias", "buenas tardes", "buenas noches",
    "como estas", "como te llamas", "me llamo", "encantado",
    "por favor", "muchas gracias", "de nada", "lo siento",
    "te amo", "te quiero", "te extrano", "te extraño",
    "feliz cumpleanos", "feliz cumpleaños", "feliz navidad",
    "feliz ano nuevo", "feliz año nuevo", "felicidades",
    "que tal", "que pasa", "que hay", "no se", "no lo se",
    "no entiendo", "lo entiendo", "perdon", "perdón",
    "tengo hambre", "tengo sed", "tengo sueno", "tengo sueño",
    "me gusta", "no me gusta", "esta bien", "está bien",
    "vamos a la playa", "vamos a la fiesta",
    # French
    "bonjour", "bonsoir", "salut", "au revoir", "merci",
    "merci beaucoup", "de rien", "sil vous plait", "s'il vous plait",
    "comment allez vous", "comment ca va", "comment ça va",
    "ca va", "ça va", "tres bien", "très bien",
    "je t'aime", "je vous aime", "mon amour", "ma cherie",
    "voulez vous coucher avec moi", "je ne sais pas",
    "joyeux anniversaire", "joyeux noel", "joyeux noël",
    "bonne annee", "bonne année", "felicitations",
    "c'est la vie", "c'est magnifique", "vive la france",
    # German
    "guten morgen", "guten tag", "guten abend", "gute nacht",
    "wie geht es dir", "wie heisst du", "wie heißt du",
    "ich heisse", "ich heiße", "danke", "bitte", "entschuldigung",
    "ja", "nein", "auf wiedersehen", "tschuss", "tschüss",
    "ich liebe dich", "ich vermisse dich", "mein schatz",
    "frohe weihnachten", "frohes neues jahr",
    # Italian
    "ciao", "buongiorno", "buona sera", "buonanotte",
    "come stai", "come va", "tutto bene", "grazie",
    "prego", "scusa", "mi dispiace", "ti amo", "amore mio",
    "buon natale", "buon anno", "auguri",
    # Portuguese / Brazilian
    "oi tudo bem", "bom dia", "boa tarde", "boa noite",
    "obrigado", "obrigada", "de nada", "por favor",
    "eu te amo", "te amo", "saudades", "feliz natal",
]

NICHE_SUBCULTURE = [
    # Star Trek
    "make it so number one", "set phasers to stun",
    "set phasers to kill", "warp speed", "warp factor 9",
    "engage warp drive", "the prime directive",
    "computer end program", "tea earl grey hot",
    "to boldly go where no man has gone before",
    "to boldly go where no one has gone before",
    "highly illogical", "fascinating", "i'm a doctor not a",
    "damn it jim", "beam me up", "scotty beam me up",
    # LOTR/Tolkien
    "ash nazg durbatuluk", "one ring to rule them all",
    "you shall not pass gandalf",
    "you cannot pass", "the eye of sauron", "the dark lord",
    "frodo of the shire", "mr frodo", "the precious",
    "we wants it we needs it", "stupid fat hobbit",
    "second breakfast", "elevenses",
    # Hitchhiker's
    "dont panic", "don't panic", "42 is the answer",
    "the answer to the great question",
    "the answer to life the universe and everything",
    "mostly harmless", "so long and thanks",
    "time is an illusion lunchtime doubly so",
    "the great green arkleseizure",
    # Sci-fi
    "the spice must flow", "fear is the mind killer",
    "i must not fear", "litany against fear",
    "shai hulud", "muad dib", "lisan al gaib",
    # Pop sci-fi
    "open the pod bay doors hal", "im sorry dave",
    "i'm sorry dave i'm afraid i can't do that",
    "this conversation can serve no purpose",
    "klaatu barada nikto",
    # Anime
    "kamehameha", "its over 9000", "it's over 9000",
    "im going to be the pirate king", "i'm going to be the pirate king",
    "ill never give up", "i'll never give up",
    "tatakae", "the rumbling",
    "san", "kun", "chan", "sensei", "senpai",
    # Internet meme classics
    "what is love baby don't hurt me",
    "rickrolled", "rick rolled",
    "all your base", "all your base are belong to us",
    "in soviet russia", "yo dawg i heard you like",
    "shoop da whoop", "leeroy jenkins",
    "this is sparta", "you shall not pass",
    "i can has cheezburger", "lol cats",
    "doge wow such bitcoin", "much wow very bitcoin",
    "do a barrel roll",
]

BITCOIN_ERA_2010_2014 = [
    "free bitcoin", "free btc", "send me bitcoin",
    "send me btc", "donate btc", "btc donations",
    "buy me a beer", "buy me a coffee",
    "thanks for the bitcoin", "thanks for the btc",
    "bitcoin tip", "tip jar",
    "satoshi tips", "satoshi tips bot",
    "bitcoin faucet", "faucet", "bitcoin faucet site",
    "free coins", "free crypto", "free money",
    "bitcoin pizza", "two pizzas",
    "ten thousand bitcoins for two pizzas",
    "laszlo hanyecz", "papa john",
    "the white paper", "bitcoin white paper",
    "peer to peer electronic cash", "p2p electronic cash",
    "bitcointalk", "bitcoin talk",
    "satoshi forum", "the cypherpunks",
    "cypherpunks write code",
    "be your own bank",
    "vires in numeris", "strength in numbers",
    "bitcoin will be money", "bitcoin will replace fiat",
    "bitcoin standard", "the bitcoin standard",
    "mastering bitcoin",
    "andreas antonopoulos", "saifedean ammous",
]


COMMON_WORD_PLUS_NUMBER = []
for base in [
    "password", "secret", "wallet", "money", "bitcoin", "crypto",
    "satoshi", "blockchain", "test", "admin", "root", "user",
    "love", "freedom", "monkey", "dragon",
]:
    for suffix in ["", "1", "12", "123", "1234", "12345", "2020",
                    "2021", "2022", "2023", "2024", "2025"]:
        COMMON_WORD_PLUS_NUMBER.append(f"{base}{suffix}")


def _vary(base: str) -> list[str]:
    out: set[str] = set()
    out.add(base)
    out.add(base.lower())
    out.add(base.title())
    if " " in base:
        out.add(base.replace(" ", ""))
        out.add(base.replace(" ", "_"))
    return list(out)


def build_round4_candidates() -> list[str]:
    from _dictionary_round3 import build_round3_candidates

    out: list[str] = []
    seen: set[str] = set()
    for p in build_round3_candidates():
        if p not in seen:
            seen.add(p)
            out.append(p)
    new_sources = [
        LONG_QUOTES, FOREIGN_PHRASES, NICHE_SUBCULTURE,
        BITCOIN_ERA_2010_2014, COMMON_WORD_PLUS_NUMBER,
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
    cs = build_round4_candidates()
    print(f"round 4 candidate count: {len(cs)}")
