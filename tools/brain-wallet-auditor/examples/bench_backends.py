"""Honest benchmark of cpu_single / cpu_threads / cpu_mp on this machine.

Tests both fast (brainwallet_sha256) and slow (bip39) schemes so the
output reflects the actually-meaningful speedups:

  * brainwallet_sha256: GIL-bound in coincurve; threads = no speedup;
    multiprocessing = SLOWER on Windows-spawn hosts.
  * bip39: PBKDF2-SHA512 dominates and releases the GIL; threads
    deliver near-linear scaling with core count.

The bench prints per-backend rates AND the cross-backend
address-equality check so a regression would be visible at runtime.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brain_wallet_auditor import (
    FundedAddressSet,
    audit_passphrases,
    brainwallet_address,
)


def run_scheme(scheme: str, pps: list[str], target: str | None,
                 verify: bool = True) -> None:
    funded = FundedAddressSet.from_iterable([target]) if target else FundedAddressSet()
    backends = ["cpu_single", "cpu_threads", "cpu_mp"]
    rates: dict[str, float] = {}
    addrs_by_backend: dict[str, list[str]] = {}

    print(f"  scheme={scheme}  n={len(pps)}")
    for backend in backends:
        t0 = time.perf_counter()
        r = audit_passphrases(
            pps, schemes=[scheme], funded_set=funded,
            run_diagnostic=False, backend=backend,
        )
        el = time.perf_counter() - t0
        rate = len(pps) / el
        rates[backend] = rate
        addrs_by_backend[backend] = [x.address for x in r.results]
        print(f"    {backend:>12}: {el:6.2f}s   {rate:>8.1f} derivations/sec"
              f"   hits={r.n_hits}")

    # Cross-backend equality
    if verify:
        ref = addrs_by_backend["cpu_single"]
        for backend, addrs in addrs_by_backend.items():
            if addrs != ref:
                print(f"    DISAGREEMENT: {backend} produces different addresses!")
                return
    ref_rate = rates["cpu_single"]
    print(f"    speedup vs cpu_single: "
          f"threads={rates['cpu_threads']/ref_rate:.2f}x  "
          f"mp={rates['cpu_mp']/ref_rate:.2f}x")
    print()


def main():
    print(f"machine: {os.cpu_count()} CPU cores")
    print()

    # ---- Fast scheme: brainwallet_sha256 ----
    print("FAST scheme (brainwallet_sha256: GIL-bound coincurve scalar mul)")
    pps_fast = [f"bench_phrase_{i:05d}" for i in range(10_000)]
    target_fast = brainwallet_address("planted_hit_secret")
    pps_fast[5000] = "planted_hit_secret"
    run_scheme("brainwallet_sha256", pps_fast, target_fast)

    # ---- Slow scheme: bip39 ----
    print("SLOW scheme (bip39: PBKDF2-SHA512 dominates, releases GIL)")
    # 256 BIP-39 mnemonics — enough to amortise pool startup but not
    # so many that the test takes minutes.
    base = ("abandon abandon abandon abandon abandon abandon "
             "abandon abandon abandon abandon abandon")
    pps_bip = [f"{base} abou{c}" for c in "abcdefgh"] * 32
    run_scheme("bip39", pps_bip, None)

    print("Use cpu_threads for bip39/warpwallet; cpu_single is fine for")
    print("brainwallet_sha256 (the fast path).")


if __name__ == "__main__":
    main()
