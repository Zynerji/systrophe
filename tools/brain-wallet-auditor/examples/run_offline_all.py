"""Run the offline Bloom audit against rounds 1 through 6 in one pass.

Combines all candidate dictionaries (de-duplicated automatically by
the rounds builders' chain), derives every address locally,
Bloom-checks against the funded-set snapshot, and API-verifies any
hits.

Output: audit_offline_combined_results.json + an entry in BTC.md
via the consolidator.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from brain_wallet_auditor import brainwallet_address
from brain_wallet_auditor.snapshot import FundedAddressSet, _Bloom
from live_chain import fetch_balance


SNAPSHOT_BIN = HERE / "snapshot_funded.bloom"
OUT_JSON = HERE / "audit_offline_combined_results.json"


def main():
    if not SNAPSHOT_BIN.exists():
        print(f"!! {SNAPSHOT_BIN} not found. Run fetch_snapshot.py first.")
        sys.exit(1)
    with open(SNAPSHOT_BIN, "rb") as f:
        payload = pickle.load(f)
    bloom = payload["bloom"] if isinstance(payload, dict) else payload
    n_added = payload.get("n_added", 0) if isinstance(payload, dict) else 0
    funded = FundedAddressSet.from_bloom_only(bloom, n_added=int(n_added))
    print(f"Loaded Bloom: claims {len(funded):,} funded addresses")
    print()

    # Round 8 chains 1-7 + 7-digit numerics
    from _dictionary_round8 import build_round8_candidates
    pps = build_round8_candidates()
    print(f"Combined candidate count (rounds 1-8): {len(pps):,}")
    print(f"Addresses to derive: {2 * len(pps):,}")
    print()

    bloom_hits: list[dict] = []
    n_derived = 0
    t0 = time.time()
    for p in pps:
        for compressed in (True, False):
            addr = brainwallet_address(p, compressed=compressed)
            n_derived += 1
            if addr in funded:
                bloom_hits.append({
                    "passphrase": p, "compressed": compressed, "addr": addr,
                })
        if n_derived % 20_000 == 0:
            el = time.time() - t0
            print(f"  [{n_derived:>8,} derived in {el:6.1f}s]  "
                  f"rate={n_derived/el:>7.0f}/s  bloom_hits={len(bloom_hits)}")
    elapsed = time.time() - t0
    print()
    print(f"  total derived: {n_derived:,} in {elapsed:.1f}s "
          f"({n_derived/elapsed:.0f}/sec)")
    print(f"  Bloom hits: {len(bloom_hits)}")
    print()

    # API-verify the Bloom hits
    verified: list[dict] = []
    if bloom_hits:
        print(f"API-verifying {len(bloom_hits)} Bloom hits...")
        for i, h in enumerate(bloom_hits):
            r = fetch_balance(h["addr"])
            r.update({"passphrase": h["passphrase"],
                       "compressed": h["compressed"], "bloom_hit": True})
            verified.append(r)
            tag = ""
            if r.get("ok") and r.get("balance", 0) > 0:
                tag = "  *** CURRENTLY FUNDED ***"
            elif r.get("ok") and r.get("funded", 0) > 0:
                tag = "  (drained)"
            elif r.get("ok"):
                tag = "  (false positive)"
            else:
                tag = "  (API error)"
            print(f"  [{i + 1}/{len(bloom_hits)}]  "
                  f"funded={r.get('funded', 0)/1e8:.6f} BTC  "
                  f"bal={r.get('balance', 0)/1e8:.8f} BTC  "
                  f"src={r.get('source', '-')}{tag}  "
                  f"<- {h['passphrase']!r}")

    # Summary
    n_real_funded = sum(1 for r in verified if r.get("ok")
                         and r.get("funded", 0) > 0)
    n_currently_funded = sum(1 for r in verified if r.get("ok")
                              and r.get("balance", 0) > 0)
    sat_lifetime = sum(r.get("funded", 0) for r in verified if r.get("ok"))
    sat_current = sum(r.get("balance", 0) for r in verified if r.get("ok"))
    n_false_positives = sum(1 for r in verified if r.get("ok")
                             and r.get("funded", 0) == 0)

    print()
    print("=" * 78)
    print(f"  candidates scanned: {len(pps):,}")
    print(f"  addresses derived: {n_derived:,}")
    print(f"  Bloom hits: {len(bloom_hits)}")
    print(f"  API-verified ever-funded: {n_real_funded}")
    print(f"  API-verified currently-funded: {n_currently_funded}")
    print(f"  Bloom false-positives (verified empty): {n_false_positives}")
    print(f"  total lifetime: {sat_lifetime/1e8:.4f} BTC")
    print(f"  total currently recoverable: {sat_current/1e8:.6f} BTC")

    if n_currently_funded > 0:
        print()
        print(f"  *** {n_currently_funded} CURRENTLY-FUNDED ADDRESSES ***")
        for r in verified:
            if r.get("ok") and r.get("balance", 0) > 0:
                c = "C" if r["compressed"] else "U"
                print(f"      [{c}] {r['addr']:<38} "
                      f"{r['balance']/1e8:.8f} BTC <- {r['passphrase']!r}")

    OUT_JSON.write_text(json.dumps({
        "n_passphrases": len(pps),
        "n_addresses_derived": n_derived,
        "n_bloom_hits": len(bloom_hits),
        "n_ever_funded": n_real_funded,
        "n_currently_funded": n_currently_funded,
        "total_lifetime_funded_sat": sat_lifetime,
        "total_currently_funded_sat": sat_current,
        "wall_clock_local_seconds": elapsed,
        "results": verified,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
