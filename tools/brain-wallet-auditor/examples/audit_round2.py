"""Round 2 — resilient fetcher + retries + expanded dictionary.

Drives:
  1. Retries every (addr, passphrase) from round 1 that didn't get a
     successful API response.
  2. Adds the round-2 expansion dictionary on top.

Uses the multi-API fallback fetcher (`live_chain.fetch_balance`) so a
single API rate-limit doesn't drop the whole audit.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from brain_wallet_auditor import brainwallet_address
from live_chain import fetch_balance
from _dictionary_round2 import build_round2_candidates


_lock = threading.Lock()
_p = {"done": 0, "n_funded": 0, "n_current": 0,
      "sat_lifetime": 0, "sat_current": 0}


def derive_addrs(passphrase: str) -> list[tuple[bool, str]]:
    return [
        (True, brainwallet_address(passphrase, compressed=True)),
        (False, brainwallet_address(passphrase, compressed=False)),
    ]


def process(addr: str, passphrase: str, compressed: bool, total: int) -> dict:
    r = fetch_balance(addr)
    r["passphrase"] = passphrase
    r["compressed"] = compressed
    with _lock:
        _p["done"] += 1
        if r.get("ok"):
            if r["funded"] > 0:
                _p["n_funded"] += 1
                _p["sat_lifetime"] += r["funded"]
            if r["balance"] > 0:
                _p["n_current"] += 1
                _p["sat_current"] += r["balance"]
        d = _p["done"]
        if d % 50 == 0 or d == total:
            print(f"  [{d:>5}/{total}]  "
                  f"funded_ever={_p['n_funded']}  "
                  f"currently={_p['n_current']}  "
                  f"lifetime={_p['sat_lifetime']/1e8:.4f} BTC  "
                  f"recoverable={_p['sat_current']/1e8:.6f} BTC")
    return r


def load_already_queried() -> set[str]:
    """Set of address strings successfully fetched in any prior round."""
    seen: set[str] = set()
    for p in HERE.glob("audit_*results.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            for row in d.get("results", []):
                if row.get("ok"):
                    seen.add(row["addr"])
        except Exception:
            continue
    return seen


def main():
    # Round-2 dictionary
    candidates = build_round2_candidates()
    print(f"Round 2 dictionary: {len(candidates)} candidates")

    # Compute (addr, passphrase, compressed) tasks; filter out
    # addresses that have already been successfully queried.
    already = load_already_queried()
    print(f"  already-queried (any round, success): {len(already)} addrs")

    tasks: list[tuple[str, str, bool]] = []
    for p in candidates:
        for compressed, addr in derive_addrs(p):
            if addr not in already:
                tasks.append((addr, p, compressed))

    print(f"  new tasks (not yet queried): {len(tasks)}")
    if not tasks:
        print("  nothing to do.")
        return

    print(f"  parallelism: 2 threads, multi-API fallback")
    print(f"  estimated wall clock: ~{len(tasks) * 0.5 / 60:.1f} minutes")
    print()

    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(process, addr, p, c, len(tasks))
                   for (addr, p, c) in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"  !! worker error: {e}")
    elapsed = time.time() - t0

    n_ok = sum(1 for r in rows if r["ok"])
    n_funded = sum(1 for r in rows if r["ok"] and r["funded"] > 0)
    n_curr = sum(1 for r in rows if r["ok"] and r["balance"] > 0)
    sat_lifetime = sum(r["funded"] for r in rows if r["ok"])
    sat_current = sum(r["balance"] for r in rows if r["ok"])

    print()
    print("=" * 78)
    print(f"  wall clock: {elapsed:.1f}s  ({len(tasks)/elapsed:.1f} addrs/sec)")
    print(f"  successful API responses: {n_ok}/{len(tasks)}")
    print(f"  ever-funded: {n_funded}")
    print(f"  currently-funded: {n_curr}")
    print(f"  total lifetime: {sat_lifetime/1e8:.4f} BTC")
    print(f"  total currently recoverable: {sat_current/1e8:.6f} BTC")

    if n_curr > 0:
        print()
        print(f"CURRENTLY FUNDED ({n_curr}):")
        for r in rows:
            if r["ok"] and r["balance"] > 0:
                c = "C" if r["compressed"] else "U"
                print(f"  [{c}] {r['addr']:<38} {r['balance']/1e8:.8f} BTC"
                      f"  <- {r['passphrase']!r}  via {r['source']}")

    out_path = HERE / "audit_round2_results.json"
    out_path.write_text(json.dumps({
        "n_passphrases": len(candidates),
        "n_addresses": len(tasks),
        "n_successful_queries": n_ok,
        "n_ever_funded": n_funded,
        "n_currently_funded": n_curr,
        "total_lifetime_funded_sat": sat_lifetime,
        "total_currently_funded_sat": sat_current,
        "wall_clock_seconds": elapsed,
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nFull JSON: {out_path}")


if __name__ == "__main__":
    main()
