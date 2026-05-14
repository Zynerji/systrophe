"""Resilient live-blockchain address-balance fetcher.

Wraps three public APIs in fallback order:
  1. blockstream.info     (Esplora REST)
  2. mempool.space        (compatible Esplora REST)
  3. blockchain.info       (legacy q/addressbalance endpoint, simpler)

Retry policy: per-API exponential backoff on connection errors.
After all three APIs fail, returns ok=False.

Returns a uniform dict:
  {addr, ok, funded, spent, balance, n_tx, source}
"""

from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request


_UA = "systrophe-brain-wallet-auditor/0.2 (research)"
_global_lock = threading.Lock()
_last_request_at: dict[str, float] = {}     # per-host last-request unix-time
_MIN_GAP_S = 0.15                            # min gap between calls to one host

# Circuit breaker: count consecutive failures per host; trip after N.
_consecutive_failures: dict[str, int] = {}
_TRIP_AT = 6
_dead_hosts: set[str] = set()


def _note_success(host: str) -> None:
    with _global_lock:
        _consecutive_failures[host] = 0


def _note_failure(host: str) -> None:
    with _global_lock:
        n = _consecutive_failures.get(host, 0) + 1
        _consecutive_failures[host] = n
        if n >= _TRIP_AT and host not in _dead_hosts:
            _dead_hosts.add(host)
            print(f"  [circuit-breaker] {host} tripped after "
                  f"{_TRIP_AT} consecutive failures; skipping for rest of run")


def _is_dead(host: str) -> bool:
    with _global_lock:
        return host in _dead_hosts


def _polite_gap(host: str) -> None:
    """Sleep enough so we don't exceed 1/_MIN_GAP_S req/sec per host (global)."""
    with _global_lock:
        last = _last_request_at.get(host, 0.0)
        now = time.time()
        gap = now - last
        if gap < _MIN_GAP_S:
            time.sleep(_MIN_GAP_S - gap + random.uniform(0, 0.05))
        _last_request_at[host] = time.time()


def _try_esplora(addr: str, host: str, timeout: float) -> dict | None:
    """Generic Esplora-REST-compatible API call (blockstream / mempool)."""
    if _is_dead(host):
        return None
    _polite_gap(host)
    req = urllib.request.Request(
        f"https://{host}/api/address/{addr}",
        headers={"User-Agent": _UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read())
        chain = raw.get("chain_stats", {})
        mempool = raw.get("mempool_stats", {})
        funded = chain.get("funded_txo_sum", 0) + mempool.get("funded_txo_sum", 0)
        spent = chain.get("spent_txo_sum", 0) + mempool.get("spent_txo_sum", 0)
        _note_success(host)
        return {
            "addr": addr, "ok": True,
            "funded": funded, "spent": spent, "balance": funded - spent,
            "n_tx": chain.get("tx_count", 0) + mempool.get("tx_count", 0),
            "source": host,
        }
    except Exception:
        _note_failure(host)
        return None


def _try_blockstream(addr: str, timeout: float) -> dict | None:
    return _try_esplora(addr, "blockstream.info", timeout)


def _try_mempool(addr: str, timeout: float) -> dict | None:
    return _try_esplora(addr, "mempool.space", timeout)


def _try_blockchain_info(addr: str, timeout: float) -> dict | None:
    """blockchain.info gives 'final_balance' and 'total_received'."""
    host = "blockchain.info"
    if _is_dead(host):
        return None
    _polite_gap(host)
    req = urllib.request.Request(
        f"https://{host}/rawaddr/{addr}?limit=0",
        headers={"User-Agent": _UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = json.loads(r.read())
        funded = int(raw.get("total_received", 0))
        balance = int(raw.get("final_balance", 0))
        spent = funded - balance
        _note_success(host)
        return {
            "addr": addr, "ok": True,
            "funded": funded, "spent": spent, "balance": balance,
            "n_tx": int(raw.get("n_tx", 0)),
            "source": host,
        }
    except Exception:
        _note_failure(host)
        return None


def fetch_balance(addr: str, timeout: float = 15.0,
                    max_attempts_per_api: int = 2) -> dict:
    """Try the three APIs in order, with circuit-breaker + short backoff.

    Order: mempool.space, blockstream.info, blockchain.info. APIs that
    have hit _TRIP_AT consecutive failures get skipped for the rest of
    the session.
    """
    APIs = [_try_mempool, _try_blockstream, _try_blockchain_info]
    for api in APIs:
        for attempt in range(max_attempts_per_api):
            r = api(addr, timeout=timeout)
            if r is not None and r.get("ok"):
                return r
            # If this host got tripped mid-attempt, don't retry it
            if r is None and attempt == 0:
                # Short single retry only if not tripped yet
                time.sleep(0.3 + random.uniform(0, 0.2))
            else:
                break
    return {"addr": addr, "ok": False, "funded": 0, "spent": 0,
            "balance": 0, "n_tx": 0, "source": "none"}
