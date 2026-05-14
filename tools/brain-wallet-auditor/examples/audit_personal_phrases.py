"""Personal-wallet recovery example.

Suppose you suspect that one of a small list of variations of a
passphrase you used years ago funds a brain-wallet address. The
auditor derives the address for each variation and (optionally)
checks against a known-funded set.

This example uses a synthetic funded set so it runs offline and
without network access. For real audits, populate `funded_set` from
a blockchain snapshot.
"""

from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from brain_wallet_auditor import (
    AddressDerivation,
    FundedAddressSet,
    audit_passphrases,
    brainwallet_address,
)


def main():
    # Suppose your "secret" passphrase was one of these spelling
    # variations -- we don't know which.
    candidates = [
        "my secret passphrase",
        "My secret passphrase",
        "my Secret Passphrase",
        "my secret passphrase!",
        "my-secret-passphrase",
        "MySecretPassphrase",
        "my_secret_passphrase",
        "mysecretpassphrase",
        # plus some leet variants
        "my s3cret p4ssphr4se",
        "M¥ secret passphrase",
    ]

    # Plant the truth: suppose the actual passphrase was "MySecretPassphrase"
    truth = "MySecretPassphrase"
    funded = FundedAddressSet.from_iterable([
        brainwallet_address(truth),
    ])

    print(f"Auditing {len(candidates)} candidate passphrases against a "
          f"funded set of {len(funded)} target address(es)...")
    print()

    t0 = time.time()
    report = audit_passphrases(
        candidates, schemes=["brainwallet_sha256"],
        funded_set=funded, run_diagnostic=True,
    )
    elapsed = time.time() - t0

    print(f"  {report.n_passphrases} passphrases x {report.n_schemes} schemes"
          f" = {report.n_passphrases * report.n_schemes} derivations in "
          f"{elapsed:.3f}s")
    print(f"  derivations/sec: {report.derivations_per_second:.0f}")
    print(f"  funded-set hits: {report.n_hits}")
    print()

    if report.n_hits > 0:
        print("Hits:")
        for h in report.hits:
            print(f"  passphrase='{h.passphrase}'  scheme={h.scheme}  "
                  f"address={h.address}")
        print()
        print("If a hit is YOUR passphrase, recovery: derive the private")
        print("key as SHA256(passphrase) and import as a P2PKH key.")
    else:
        print("No hits. None of the candidates produce a funded address.")
    print()

    if report.pool_diagnostic is not None:
        d = report.pool_diagnostic
        print(f"Catcher pool diagnostic: verdict='{d.verdict}'")
        print(f"  {d.interpretation}")


if __name__ == "__main__":
    main()
