"""Stream-download the blockchair daily Bitcoin funded-address dump
and build a Bloom filter of currently-funded addresses in one pass.

Output: snapshot_funded.bloom (pickled FundedAddressSet) + a
snapshot_meta.json with row counts / source URL / timestamp.

Streaming approach: never decompresses the full TSV to disk; reads
gzipped chunks from HTTP, decompresses on the fly, parses one line
at a time, inserts only currently-funded addresses (balance > 0)
into the Bloom.

The dump format is one address per row, TSV-separated. Common
columns include `address`, `balance`, `received`, `spent` (varies
by exact format).
"""

from __future__ import annotations

import gzip
import io
import json
import pathlib
import pickle
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brain_wallet_auditor import FundedAddressSet
from brain_wallet_auditor.snapshot import _Bloom


URL = "https://gz.blockchair.com/bitcoin/addresses/blockchair_bitcoin_addresses_latest.tsv.gz"
HERE = pathlib.Path(__file__).resolve().parent
SNAPSHOT_BIN = HERE / "snapshot_funded.bloom"
SNAPSHOT_META = HERE / "snapshot_meta.json"


def stream_addresses(url: str = URL):
    """Yield (address, balance_sat) tuples by stream-decompressing the HTTP body."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "systrophe-brain-wallet-auditor/0.3"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        # Stream-decompress on the fly
        with gzip.GzipFile(fileobj=resp) as gz:
            # Read line by line via TextIOWrapper
            reader = io.TextIOWrapper(gz, encoding="utf-8", errors="replace",
                                        newline="")
            header_line = reader.readline().strip()
            cols = header_line.split("\t")
            try:
                i_addr = cols.index("address")
            except ValueError:
                i_addr = 0
            balance_idx = None
            for cand in ("balance", "current_balance", "balance_sat"):
                if cand in cols:
                    balance_idx = cols.index(cand)
                    break
            if balance_idx is None:
                # Try to use received - spent if balance not present
                try:
                    i_recv = cols.index("received")
                    i_spent = cols.index("spent")
                except ValueError:
                    raise RuntimeError(
                        f"can't find balance column in {cols!r}",
                    )
                for line in reader:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) <= max(i_addr, i_recv, i_spent):
                        continue
                    try:
                        bal = int(parts[i_recv]) - int(parts[i_spent])
                    except ValueError:
                        continue
                    yield (parts[i_addr], bal)
                return
            for line in reader:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(i_addr, balance_idx):
                    continue
                try:
                    bal = int(parts[balance_idx])
                except ValueError:
                    continue
                yield (parts[i_addr], bal)


def main():
    print(f"Stream-downloading {URL}")
    print(f"Building Bloom of CURRENTLY-FUNDED addresses only...")
    print()
    n_total = 0
    n_funded = 0
    total_balance_sat = 0
    # Pre-size: BTC has ~1B distinct addresses, but only ~50M have non-zero
    # balance. Bloom at 50M with fpr=1e-4 ~ 120 MB.
    bloom = _Bloom(expected_n=50_000_000, fpr=1e-4)
    t0 = time.time()
    last_log = t0
    try:
        for addr, bal in stream_addresses():
            n_total += 1
            if bal > 0:
                bloom.add(addr)
                n_funded += 1
                total_balance_sat += bal
            now = time.time()
            if now - last_log >= 5:
                el = now - t0
                rate = n_total / el if el > 0 else 0
                print(f"  [{n_total:>10,} rows, {el:7.1f}s]  "
                      f"funded={n_funded:>10,}  "
                      f"bal_total={total_balance_sat/1e8:>14.2f} BTC  "
                      f"rate={rate:>7.0f}/s")
                last_log = now
    except Exception as e:
        print(f"  !! stream error after {n_total:,} rows: "
              f"{type(e).__name__}: {e}")
        print(f"  Saving partial Bloom...")

    elapsed = time.time() - t0
    print()
    print(f"  total rows: {n_total:,}")
    print(f"  funded addresses: {n_funded:,}")
    print(f"  total balance (sat): {total_balance_sat:,}")
    print(f"  total balance (BTC): {total_balance_sat/1e8:,.2f}")
    print(f"  wall clock: {elapsed:.1f}s")

    with open(SNAPSHOT_BIN, "wb") as f:
        pickle.dump({"bloom": bloom, "n_added": n_funded}, f)
    SNAPSHOT_META.write_text(json.dumps({
        "url": URL,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "total_rows": n_total,
        "n_currently_funded": n_funded,
        "total_balance_sat": total_balance_sat,
        "total_balance_btc": total_balance_sat / 1e8,
        "fetch_elapsed_seconds": elapsed,
        "bloom_m_bits": getattr(bloom, "m", None),
        "bloom_k_hashes": getattr(bloom, "k", None),
        "bloom_size_bytes": len(bloom.bits) if hasattr(bloom, "bits") else None,
    }, indent=2), encoding="utf-8")
    print()
    print(f"Wrote Bloom: {SNAPSHOT_BIN}")
    print(f"Wrote meta:  {SNAPSHOT_META}")


if __name__ == "__main__":
    main()
