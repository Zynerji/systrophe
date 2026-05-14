"""Round 2: expanded dictionary, live blockchain audit.

Identical wire protocol to round-1 (audit_real_snapshot_large.py) but
draws candidates from _dictionary_round2.build_round2_candidates().
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # for _dictionary_round2

from audit_real_snapshot_large import fetch_one_stats
from brain_wallet_auditor import brainwallet_address
from _dictionary_round2 import build_round2_candidates


_progress_lock = threading.Lock()
_p = {"done": 0, "n_funded": 0, "n_currently_funded": 0,
      "total_lifetime_sat": 0, "total_current_sat": 0}


def process_passphrase(p: str, total: int) -> list[dict]:
    rows = []
    for compressed in (True, False):
        addr = brainwallet_address(p, compressed=compressed)
        st = fetch_one_stats(addr)
        st["passphrase"] = p
        st["compressed"] = compressed
        rows.append(st)
        with _progress_lock:
            _p["done"] += 1
            if st.get("funded", 0) > 0:
                _p["n_funded"] += 1
                _p["total_lifetime_sat"] += st["funded"]
            if st.get("balance", 0) > 0:
                _p["n_currently_funded"] += 1
                _p["total_current_sat"] += st["balance"]
            d = _p["done"]
            if d % 100 == 0 or d == total:
                print(f"  [{d:>5}/{total}]  "
                      f"funded_ever={_p['n_funded']}  "
                      f"currently={_p['n_currently_funded']}  "
                      f"lifetime={_p['total_lifetime_sat']/1e8:.4f} BTC")
    return rows


def main():
    candidates = build_round2_candidates()
    total = 2 * len(candidates)
    print(f"Round 2: {len(candidates)} candidates, {total} queries to blockstream.info")
    print(f"Parallelism: 4 threads (polite)")
    print()
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

    n_ok = sum(1 for r in all_rows if r["ok"])
    n_funded = sum(1 for r in all_rows if r["ok"] and r["funded"] > 0)
    n_curr = sum(1 for r in all_rows if r["ok"] and r["balance"] > 0)
    total_lifetime = sum(r["funded"] for r in all_rows if r["ok"])
    total_current = sum(r["balance"] for r in all_rows if r["ok"])

    print()
    print("=" * 78)
    print(f"  Wall clock:                {elapsed:.1f}s ({total/elapsed:.1f}/s)")
    print(f"  Successful API responses:  {n_ok}/{total}")
    print(f"  Addresses EVER funded:     {n_funded}")
    print(f"  Addresses CURRENTLY funded: {n_curr}")
    print(f"  Total lifetime BTC funneled: {total_lifetime/1e8:.4f} BTC")
    print(f"  Currently recoverable:       {total_current/1e8:.6f} BTC")
    print()

    funded_rows = sorted(
        [r for r in all_rows if r["ok"] and r["funded"] > 0],
        key=lambda r: r["funded"], reverse=True,
    )
    print(f"Top {min(40, len(funded_rows))} addresses by lifetime funded:")
    for r in funded_rows[:40]:
        c = "C" if r["compressed"] else "U"
        print(f"  [{c}] {r['addr']:<38} {r['funded']/1e8:>12.6f} BTC "
              f"({r['n_tx']:>6} tx)  <- {r['passphrase']!r}")

    if n_curr > 0:
        print()
        print(f"CURRENTLY FUNDED ({n_curr}):")
        for r in all_rows:
            if r["ok"] and r["balance"] > 0:
                c = "C" if r["compressed"] else "U"
                print(f"  [{c}] {r['addr']:<38} {r['balance']/1e8:>12.8f} BTC"
                      f"  <- {r['passphrase']!r}")

    out_path = pathlib.Path(__file__).parent / "audit_round2_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_passphrases": len(candidates),
            "n_addresses": total,
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
