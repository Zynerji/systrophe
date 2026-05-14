"""Run the brain-wallet auditor against the LIVE Bitcoin blockchain.

Methodology
-----------
Iterates a small set of publicly-known weak passphrases (the
canonical Vasek-paper-style test set). For each passphrase, derives
both the compressed and uncompressed mainnet P2PKH address, then
queries blockstream.info's free public API for that address's
LIFETIME funded_total and CURRENT balance.

The "snapshot" here is built live, per-address, from the actual
chain. No bulk download required.

Ethical scope
-------------
This script ONLY READS public blockchain data. It does not
construct transactions, does not move funds, does not export
private keys. The passphrases hard-coded below are the canonical
publicly-published weak-brain-wallet test set used in Vasek et al.
(2016) "The Bitcoin Brain Drain" and reproduced in dozens of
follow-on papers. Their derived addresses have been drained for
over a decade; this script just confirms that.

Output meaning
--------------
* funded_total > 0  →  someone, at some point, sent BTC to that
                       address via this weak passphrase.
* balance == 0      →  the wallet has been drained (typical).
* balance > 0       →  there's currently unspent BTC at that address.
                       For passphrases YOU own, that's recoverable.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brain_wallet_auditor import brainwallet_address


# Canonical publicly-known weak-brain-wallet passphrases.
# All from Vasek et al. 2016 + follow-on lit + xkcd 936.
PASSPHRASES = [
    "correct horse battery staple",  # xkcd 936; the most famous case
    "password",
    "satoshi nakamoto",
    "to be or not to be, that is the question",
    "1",
    "bitcoin is awesome",
    "iloveyou",
    "letmein",
    "the quick brown fox jumps over the lazy dog",
    "abandon abandon",
    "monkey",
    "qwerty",
    "secret",
    "passphrase",
    "blockchain",
    "say hello to my little friend",
    "test",
    "trustno1",
    "Never gonna give you up",
    "do or do not there is no try",
]


def fetch_address_stats(addr: str, timeout: float = 20.0) -> dict | None:
    """Return blockstream.info chain stats for `addr`, or None on error."""
    req = urllib.request.Request(
        f"https://blockstream.info/api/address/{addr}",
        headers={
            "User-Agent": "systrophe-brain-wallet-auditor/0.1 "
                          "(research; passphrase-audit)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"    !! API error for {addr}: {e}")
        return None


def derive_and_query(passphrase: str, compressed: bool) -> dict:
    addr = brainwallet_address(passphrase, compressed=compressed)
    stats = fetch_address_stats(addr)
    if stats is None:
        return {"addr": addr, "ok": False}
    chain = stats.get("chain_stats", {})
    mempool = stats.get("mempool_stats", {})
    funded = chain.get("funded_txo_sum", 0) + mempool.get("funded_txo_sum", 0)
    spent = chain.get("spent_txo_sum", 0) + mempool.get("spent_txo_sum", 0)
    n_tx = chain.get("tx_count", 0) + mempool.get("tx_count", 0)
    return {
        "addr": addr,
        "ok": True,
        "funded_total_sat": funded,
        "spent_total_sat": spent,
        "balance_sat": funded - spent,
        "tx_count": n_tx,
    }


def fmt_btc(sats: int) -> str:
    return f"{sats:>15,} sat ({sats / 1e8:>10.4f} BTC)"


def main():
    print("Brain-wallet auditor vs LIVE Bitcoin blockchain")
    print("=" * 78)
    print(f"  Testing {len(PASSPHRASES)} publicly-known weak passphrases.")
    print(f"  For each: compute SHA256-brainwallet address (both compressed and")
    print(f"  uncompressed forms), query blockstream.info for current balance.")
    print()

    total_funded_lifetime = 0
    total_current_balance = 0
    n_ever_funded = 0
    n_currently_funded = 0
    results = []

    for i, p in enumerate(PASSPHRASES):
        print(f"[{i + 1:>2}/{len(PASSPHRASES)}] '{p}'")
        for compressed_flag, label in [(True, "compressed"),
                                          (False, "uncompressed")]:
            r = derive_and_query(p, compressed=compressed_flag)
            if not r.get("ok"):
                continue
            funded = r["funded_total_sat"]
            balance = r["balance_sat"]
            n_tx = r["tx_count"]
            marker = "  "
            if balance > 0:
                marker = "**"  # currently has BTC
                n_currently_funded += 1
                total_current_balance += balance
            if funded > 0:
                n_ever_funded += 1
                total_funded_lifetime += funded
            print(f"    {marker}{label:<13} {r['addr']:<38} "
                  f"funded={fmt_btc(funded)}  bal={fmt_btc(balance)}  "
                  f"txs={n_tx}")
            results.append({"passphrase": p, "compressed": compressed_flag,
                              **r})
            # be polite to the public API
            time.sleep(0.1)

    print()
    print("=" * 78)
    print(f"  Total addresses tested:          {2 * len(PASSPHRASES)}")
    print(f"  Addresses EVER funded:           {n_ever_funded}")
    print(f"  Addresses CURRENTLY funded:      {n_currently_funded}")
    print(f"  Lifetime funded across all:      {fmt_btc(total_funded_lifetime)}")
    print(f"  Currently recoverable balance:   {fmt_btc(total_current_balance)}")
    print()
    print("Interpretation: lifetime funded > 0 confirms these passphrases were")
    print("ACTUAL wallet seeds used by humans. balance == 0 confirms they have")
    print("been (re)drained, typically within minutes of being funded -- exactly")
    print("the Vasek 2016 'Bitcoin Brain Drain' phenomenon.")

    # Save raw JSON for inspection
    out_path = pathlib.Path(__file__).parent / "audit_real_snapshot_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results,
                   "summary": {
                       "n_passphrases": len(PASSPHRASES),
                       "n_addresses": 2 * len(PASSPHRASES),
                       "n_ever_funded": n_ever_funded,
                       "n_currently_funded": n_currently_funded,
                       "total_funded_lifetime_sat": total_funded_lifetime,
                       "total_current_balance_sat": total_current_balance,
                   }}, f, indent=2)
    print(f"\nRaw JSON: {out_path}")


if __name__ == "__main__":
    main()
