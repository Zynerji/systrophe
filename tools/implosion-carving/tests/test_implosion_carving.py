"""Tests for implosion-carving."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from implosion_carving import (
    ImplosionCarver,
    ImplosionSummary,
    PocketGeometry,
    Z3MonodromySignature,
    carve_photon_pocket,
    closure_residual,
    compute_z3_signature,
)
from implosion_carving.monodromy import closure_on_cover_residual
from systrophe.vanstockum import VanStockumInterior


# ---------------------------------------------------------------------------
# Z_3 monodromy signature (independent of spacetime — topological)
# ---------------------------------------------------------------------------


def test_z3_signature_converges_to_continuum_triplet():
    """Lowest distinct rescaled triplet → {0, 1/9, 4/9} as N grows."""
    sig_64 = compute_z3_signature(N=64)
    sig_512 = compute_z3_signature(N=512)
    assert isinstance(sig_64, Z3MonodromySignature)
    # Convergence is ~ 1/N^2 — coarser grid is worse, finer is better.
    assert sig_512.triplet_convergence_error < sig_64.triplet_convergence_error
    assert sig_512.triplet_convergence_error < 1e-3


def test_z3_continuum_triplet_values():
    """The continuum triplet is the canonical {0, 1/9, 4/9}."""
    sig = compute_z3_signature(N=1024)
    np.testing.assert_allclose(sig.continuum_triplet,
                                 [0.0, 1.0 / 9.0, 4.0 / 9.0])


def test_z3_closure_phase_is_zero():
    """Σ ω^k for k=0,1,2 = 0 for primitive cube root of unity."""
    sig = compute_z3_signature(N=128)
    res = closure_on_cover_residual(sig)
    assert res < 1e-12


def test_z3_three_branches_lowest_modes_aliased():
    """Branch 0's lowest is 0; branches 1 and 2 both alias to 1/9.

    The aliasing arises because the discrete spectrum is
    ``(n + branch/3)²`` with n ranging over N values that wrap, so for
    branch=2 the wrapped index n=N-1 gives effective ``(2/3 - 1) = -1/3``,
    which is closer to 0 than n=0's ``2/3``. The distinct continuum
    triplet ``{0, 1/9, 4/9}`` only emerges when ALL branches are merged
    (which `lowest_distinct_triplets` does)."""
    sig = compute_z3_signature(N=512)
    first_per_branch = [b[0] for b in sig.branch_eigenvalues]
    assert first_per_branch[0] == pytest.approx(0.0, abs=1e-6)
    assert first_per_branch[1] == pytest.approx(1.0 / 9.0, abs=2e-3)
    assert first_per_branch[2] == pytest.approx(1.0 / 9.0, abs=2e-3)
    # And the merged-across-branches triplet recovers {0, 1/9, 4/9}.
    np.testing.assert_allclose(
        sig.triplet_eigenvalues, [0.0, 1.0 / 9.0, 4.0 / 9.0], atol=2e-3,
    )


# ---------------------------------------------------------------------------
# carve_photon_pocket: Schwarzschild-limit sanity
# ---------------------------------------------------------------------------


def test_carve_succeeds_on_canonical_vs_fixture():
    """The canonical fixture (omega=1, R=1) admits an engineered M at r=1.5.

    Mirrors the existing photon-sphere test: r_target=1.5 → M≈0.617.
    """
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(vs, r_target=1.5)
    assert pocket.M_engineered is not None
    assert 0.4 < pocket.M_engineered < 0.8


def test_carve_produces_tight_dbdr_residual():
    """A successfully-carved pocket has db/dr ≈ 0 at r_target."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(vs, r_target=1.5)
    assert pocket.M_engineered is not None
    # Brent solver tolerance — should be tight.
    assert abs(pocket.closure_residual_dbdr) < 1e-3


def test_carve_is_stable_pocket():
    """The hybrid photon sphere is stable (d²b/dr² > 0) — trapped."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(vs, r_target=1.5)
    assert pocket.is_stable


def test_carve_is_carved_flag():
    """The is_carved aggregate flag is True for a successful carve."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(vs, r_target=1.5)
    assert pocket.is_carved


def test_carve_records_schwarzschild_limit():
    """Schwarzschild-limit reference M = r_target / 3 is always reported."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(vs, r_target=3.0)
    assert pocket.schwarzschild_limit_M == pytest.approx(1.0)


def test_carve_unattainable_target_returns_none():
    """A target outside the M search range returns M_engineered=None."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(
        vs, r_target=1.5, M_range=(10.0, 11.0),  # too large
    )
    assert pocket.M_engineered is None
    assert not pocket.is_carved
    assert np.isnan(pocket.closure_residual_dbdr)


# ---------------------------------------------------------------------------
# closure_residual: direct API
# ---------------------------------------------------------------------------


def test_closure_residual_tight_at_engineered_M():
    """closure_residual(M_engineered) → dbdr ≈ 0."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(vs, r_target=1.5)
    assert pocket.M_engineered is not None
    res = closure_residual(vs, r_target=1.5, M=pocket.M_engineered)
    assert abs(res["dbdr"]) < 1e-3
    assert res["stability"]


def test_closure_residual_loose_at_wrong_M():
    """At an off-engineered M, the dbdr residual is large."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    pocket = carve_photon_pocket(vs, r_target=1.5)
    assert pocket.M_engineered is not None
    M_wrong = pocket.M_engineered * 0.5
    res = closure_residual(vs, r_target=1.5, M=M_wrong)
    assert abs(res["dbdr"]) > abs(pocket.closure_residual_dbdr)


# ---------------------------------------------------------------------------
# ImplosionCarver end-to-end
# ---------------------------------------------------------------------------


def test_carver_carves_pocket():
    car = ImplosionCarver(omega=1.0, R=1.0)
    p = car.carve(r_target=1.5)
    assert p.is_carved
    assert p.is_stable


def test_carver_z3_signature_is_cached():
    car = ImplosionCarver(omega=1.0, R=1.0)
    s1 = car.z3_signature(N=128)
    s2 = car.z3_signature(N=128)
    assert s1 is s2  # identity, not just equal
    s3 = car.z3_signature(N=256)
    assert s3 is not s1  # different N → different signature


def test_carver_summary_one_shot():
    car = ImplosionCarver(omega=1.0, R=1.0)
    summ = car.summary(r_target=1.5)
    assert isinstance(summ, ImplosionSummary)
    assert summ.is_carved
    assert summ.is_stable
    assert summ.z3_triplet_convergence_error < 1e-3
    assert summ.z3_closure_phase_residual < 1e-12
    assert summ.M_engineered is not None
    assert summ.schwarzschild_limit_M == pytest.approx(0.5)


def test_carver_summary_handles_unattainable_target():
    car = ImplosionCarver(omega=1.0, R=1.0, M_range=(10.0, 11.0))
    summ = car.summary(r_target=1.5)
    assert not summ.is_carved
    assert summ.M_engineered is None
    # Z_3 part still works.
    assert summ.z3_triplet_convergence_error < 1e-3


def test_carver_retrograde_branch():
    """Retrograde branch should also admit an engineered carve."""
    car = ImplosionCarver(omega=1.0, R=1.0)
    p = car.carve(r_target=1.5, branch="retrograde")
    # Retrograde may have a different M, may or may not carve at the
    # canonical fixture; we just check the API is exercised.
    assert isinstance(p, PocketGeometry)


def test_carver_M_engineered_distinct_from_schwarzschild_limit():
    """For omega=1, R=1 (non-Schwarzschild background), the engineered
    M deviates from the pure-Schwarzschild limit r_target/3."""
    car = ImplosionCarver(omega=1.0, R=1.0)
    p = car.carve(r_target=1.5)
    assert p.M_engineered is not None
    # r_target/3 = 0.5; engineered ~0.617 from existing test
    assert abs(p.M_engineered - p.schwarzschild_limit_M) > 0.05
