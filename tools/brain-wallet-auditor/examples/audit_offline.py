"""Offline audit: derive locally, Bloom-check, API-verify hits only.

Once `fetch_snapshot.py` has built `snapshot_funded.bloom`, this
script runs an arbitrary candidate dictionary against it without
any per-address API calls. Bloom hits get API-verified before being
reported (Bloom false-positive rate is ~1e-4).

Throughput: derivation runs at ~9 K/sec on the local machine
(cpu_single brainwallet_sha256). Bloom-checking is microseconds.
So 1 million candidates = ~2 million addresses (compressed +
uncompressed) = ~3-4 minutes wall clock for the local part.
API verification only fires on the (rare) Bloom hits, so even with
the rate-limit dance the API workload is tiny.
"""

from __future__ import annotations

import argparse
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


def load_snapshot() -> FundedAddressSet:
    if not SNAPSHOT_BIN.exists():
        raise FileNotFoundError(
            f"{SNAPSHOT_BIN} missing. Run fetch_snapshot.py first.",
        )
    with open(SNAPSHOT_BIN, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "bloom" in payload:
        bloom = payload["bloom"]
        n_added = payload.get("n_added", 0)
    else:
        bloom = payload
        n_added = 0
    return FundedAddressSet.from_bloom_only(bloom, n_added=int(n_added))


def _load_candidates(which: str) -> list[str]:
    """Build the requested candidate list."""
    if which == "round1":
        from audit_real_snapshot_large import build_candidates
        return build_candidates()
    if which == "round2":
        from _dictionary_round2 import build_round2_candidates
        return build_round2_candidates()
    if which == "round3":
        from _dictionary_round3 import build_round3_candidates
        return build_round3_candidates()
    if which == "round4":
        from _dictionary_round4 import build_round4_candidates
        return build_round4_candidates()
    if which == "round5":
        from _dictionary_round5 import build_round5_candidates
        return build_round5_candidates()
    if which == "round6":
        from _dictionary_round6 import build_round6_candidates
        return build_round6_candidates()
    raise ValueError(f"unknown candidate set {which!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="round4",
                    help="round1 / round2 / round3 / round4")
    ap.add_argument("--verify-hits", action="store_true",
                    help="API-verify every Bloom hit (default: True)",
                    default=True)
    ap.add_argument("--out", default=str(HERE / "audit_offline_results.json"))
    args = ap.parse_args()

    print(f"Loading snapshot Bloom from {SNAPSHOT_BIN} ...")
    funded = load_snapshot()
    print(f"  bloom-only mode, claims {len(funded):,} funded addresses")

    print(f"Building candidates ({args.candidates}) ...")
    pps = _load_candidates(args.candidates)
    print(f"  {len(pps):,} candidate passphrases "
          f"({2 * len(pps):,} addresses to derive)")

    print(f"Deriving and Bloom-checking locally...")
    t0 = time.time()
    bloom_hits: list[dict] = []
    n_derived = 0
    for p in pps:
        for compressed in (True, False):
            addr = brainwallet_address(p, compressed=compressed)
            n_derived += 1
            if addr in funded:
                bloom_hits.append({
                    "passphrase": p, "compressed": compressed, "addr": addr,
                })
        if n_derived % 10_000 == 0:
            elapsed = time.time() - t0
            rate = n_derived / elapsed
            print(f"  [{n_derived:>8,} derived in {elapsed:6.1f}s]  "
                  f"rate={rate:>7.0f}/s  bloom_hits={len(bloom_hits)}")
    elapsed = time.time() - t0
    print()
    print(f"  total derived: {n_derived:,} in {elapsed:.1f}s "
          f"({n_derived/elapsed:.0f}/sec)")
    print(f"  Bloom hits: {len(bloom_hits)}")
    print()

    # API-verify each Bloom hit
    if args.verify_hits and bloom_hits:
        print(f"Verifying {len(bloom_hits)} Bloom hits via live API...")
        verified: list[dict] = []
        for i, h in enumerate(bloom_hits):
            r = fetch_balance(h["addr"])
            r["passphrase"] = h["passphrase"]
            r["compressed"] = h["compressed"]
            r["bloom_hit"] = True
            verified.append(r)
            print(f"  [{i + 1}/{len(bloom_hits)}]  ok={r.get('ok')}  "
                  f"funded={r.get('funded', 0)/1e8:.4f} BTC  "
                  f"bal={r.get('balance', 0)/1e8:.6f} BTC  "
                  f"src={r.get('source')}  <- {h['passphrase']!r}")
        n_real = sum(1 for r in verified if r.get("ok")
                      and r.get("funded", 0) > 0)
        n_curr = sum(1 for r in verified if r.get("ok")
                      and r.get("balance", 0) > 0)
        sat_lifetime = sum(r.get("funded", 0) for r in verified
                            if r.get("ok"))
        sat_curr = sum(r.get("balance", 0) for r in verified
                        if r.get("ok"))
        print()
        print("=" * 78)
        print(f"  Bloom hits: {len(bloom_hits)}")
        print(f"  API-confirmed ever-funded: {n_real}")
        print(f"  API-confirmed currently-funded: {n_curr}")
        print(f"  total lifetime: {sat_lifetime/1e8:.4f} BTC")
        print(f"  total currently recoverable: {sat_curr/1e8:.6f} BTC")
    else:
        verified = bloom_hits

    out = pathlib.Path(args.out)
    out.write_text(json.dumps({
        "candidate_set": args.candidates,
        "n_passphrases": len(pps),
        "n_addresses_derived": n_derived,
        "n_bloom_hits": len(bloom_hits),
        "wall_clock_local_seconds": elapsed,
        "verified_hits": verified,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
