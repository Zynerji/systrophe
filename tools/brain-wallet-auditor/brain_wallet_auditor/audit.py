"""Main audit entry point: passphrases -> addresses -> funded-set check."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from .catcher import PoolDiagnostic, diagnose_candidate_pool
from .derivation import AddressDerivation, Scheme
from .snapshot import FundedAddressSet


@dataclass
class AuditResult:
    """Per-passphrase audit outcome."""
    passphrase: str
    scheme: str
    address: str
    found_in_funded_set: bool


@dataclass
class AuditReport:
    """End-to-end audit report."""

    n_passphrases: int
    n_schemes: int
    n_hits: int
    results: list[AuditResult]
    pool_diagnostic: PoolDiagnostic | None = None
    elapsed_seconds: float = 0.0
    derivations_per_second: float = 0.0

    @property
    def hits(self) -> list[AuditResult]:
        """Just the passphrases that hit the funded set."""
        return [r for r in self.results if r.found_in_funded_set]

    def __repr__(self) -> str:
        return (
            f"AuditReport(n_passphrases={self.n_passphrases}, "
            f"n_schemes={self.n_schemes}, n_hits={self.n_hits}, "
            f"derivations/s={self.derivations_per_second:.1f})"
        )


def audit_passphrases(
    passphrases: Iterable[str],
    schemes: Iterable[Scheme] = ("brainwallet_sha256",),
    funded_set: FundedAddressSet | None = None,
    derivation_kwargs: dict | None = None,
    run_diagnostic: bool = True,
) -> AuditReport:
    """Audit a list of candidate passphrases against a funded-address set.

    Parameters
    ----------
    passphrases
        The candidate passphrases you supply. The tool does NOT
        generate passphrases for you.
    schemes
        Which derivation schemes to test. Each passphrase is derived
        via every requested scheme; one result row per (passphrase,
        scheme) pair.
    funded_set
        Optional `FundedAddressSet`. If omitted, all `found_in_funded_set`
        flags are False but the derivation is still done (useful for
        the diagnostic-only mode).
    derivation_kwargs
        Forwarded to `AddressDerivation`. Use this to set, e.g.,
        `warpwallet_salt` or `bip39_path`.
    run_diagnostic
        If True (default), run the catcher pool diagnostic on the
        first scheme's derived addresses. The diagnostic is purely
        informative -- it does not accelerate the audit.
    """
    pps = list(passphrases)
    schemes = list(schemes)
    derivation_kwargs = derivation_kwargs or {}
    funded_set = funded_set if funded_set is not None else FundedAddressSet()

    t0 = time.time()
    results: list[AuditResult] = []
    pool_diag: PoolDiagnostic | None = None

    for scheme in schemes:
        deriv = AddressDerivation(scheme=scheme, **derivation_kwargs)
        scheme_addresses = []
        for p in pps:
            addr = deriv.derive(p)
            scheme_addresses.append(addr)
            results.append(AuditResult(
                passphrase=p, scheme=scheme, address=addr,
                found_in_funded_set=(addr in funded_set),
            ))
        if run_diagnostic and pool_diag is None:
            try:
                pool_diag = diagnose_candidate_pool(pps, deriv.derive)
            except Exception:
                pool_diag = None

    elapsed = time.time() - t0
    n_derivations = len(pps) * len(schemes)
    return AuditReport(
        n_passphrases=len(pps),
        n_schemes=len(schemes),
        n_hits=sum(1 for r in results if r.found_in_funded_set),
        results=results,
        pool_diagnostic=pool_diag,
        elapsed_seconds=elapsed,
        derivations_per_second=(
            float(n_derivations / elapsed) if elapsed > 0 else float("inf")
        ),
    )
