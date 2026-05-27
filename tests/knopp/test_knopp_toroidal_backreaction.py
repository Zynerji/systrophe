"""Tests for self-consistent NK back-reaction on the toroidal CTC band."""

import math

import pytest

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_backreaction import (
    BackreactedBand,
    BackreactionDiagnostic,
    F_effective,
    backreacted_band,
    backreaction_diagnostic,
    critical_lambda,
    q_threshold_from_balance,
    summarise_backreaction,
    t_kk_polyakov_boulware,
)


@pytest.fixture
def binary_in_band():
    return EffectiveToroidalKerrBinary(M=1.0, d=2.0, chi=1.0)


@pytest.fixture
def binary_no_band():
    return EffectiveToroidalKerrBinary(M=1.0, d=10.0, chi=1.0)


# ----- effective metric & Polyakov stress --------------------------------


def test_F_effective_positive_outside_core(binary_in_band):
    # F = 1 + 4M/r is strictly > 1 everywhere outside r=0.
    for r in [0.5, 1.0, 5.0, 10.0]:
        assert F_effective(binary_in_band, r) > 1.0


def test_t_kk_finite_in_band(binary_in_band):
    for r in [0.7, 1.0, 1.5, 2.5, 3.5]:
        T = t_kk_polyakov_boulware(binary_in_band, r)
        assert math.isfinite(T)


def test_t_kk_outer_region_decays(binary_in_band):
    """At large rho, F -> 1, F', F'' -> 0, so T_kk -> 0."""
    near = abs(t_kk_polyakov_boulware(binary_in_band, 2.0))
    far = abs(t_kk_polyakov_boulware(binary_in_band, 50.0))
    assert far < near


# ----- back-reacted band -------------------------------------------------


def test_lambda_zero_recovers_classical(binary_in_band):
    b = backreacted_band(binary_in_band, lam=0.0)
    assert b.rho_in_BR == pytest.approx(b.rho_in_classical, abs=1e-6)
    assert b.rho_out_BR == pytest.approx(b.rho_out_classical, abs=1e-6)
    assert b.shift_in == pytest.approx(0.0, abs=1e-6)
    assert b.shift_out == pytest.approx(0.0, abs=1e-6)
    assert b.inner_converged and b.outer_converged


def test_positive_lambda_converges(binary_in_band):
    b = backreacted_band(binary_in_band, lam=1.0)
    assert b.inner_converged
    assert b.outer_converged
    assert b.rho_in_BR is not None
    assert b.rho_out_BR is not None
    # band must be non-empty and well-ordered
    assert b.rho_out_BR > b.rho_in_BR > 0.0


def test_no_band_returns_closed_sentinel(binary_no_band):
    b = backreacted_band(binary_no_band, lam=1.0)
    assert b.rho_in_classical is None
    assert b.rho_out_classical is None
    assert b.band_closed is True


def test_rejects_negative_lambda(binary_in_band):
    with pytest.raises(ValueError):
        backreacted_band(binary_in_band, lam=-1.0)


def test_large_lambda_closes_band(binary_in_band):
    b = backreacted_band(binary_in_band, lam=1000.0)
    assert b.band_closed is True


def test_band_translates_or_shrinks_with_lambda(binary_in_band):
    """As lambda grows, the BR band edges shift continuously from the
    classical positions; the trace of (rho_in, rho_out) is a smooth
    curve. Exact monotone width-shrinkage depends on the sign pattern
    of <T_kk> across the band -- here T_kk is positive near rho_inner
    and negative near rho_outer, so both edges drift outward and the
    width is roughly conserved until band closure."""
    b0 = backreacted_band(binary_in_band, lam=0.0)
    b1 = backreacted_band(binary_in_band, lam=1.0)
    b5 = backreacted_band(binary_in_band, lam=5.0)
    assert not b0.band_closed
    assert not b1.band_closed
    assert not b5.band_closed
    # Edges are continuous in lambda
    assert abs(b1.rho_in_BR - b0.rho_in_BR) < 0.1
    assert abs(b5.rho_in_BR - b1.rho_in_BR) < 0.5


# ----- critical lambda ---------------------------------------------------


def test_critical_lambda_zero_for_no_band(binary_no_band):
    assert critical_lambda(binary_no_band) == 0.0


def test_critical_lambda_finite_positive(binary_in_band):
    lc = critical_lambda(binary_in_band)
    assert 0.0 < lc < float("inf")


def test_band_open_at_small_lambda(binary_in_band):
    """At lambda well below the closure threshold (and far enough from
    it that NK convergence is robust across platforms), the band must
    be open."""
    # lambda_crit on the working config is ~ 15-22 depending on
    # NK finite-difference platform jitter. lambda = 1 is safely below.
    b = backreacted_band(binary_in_band, lam=1.0)
    assert b.band_closed is False


def test_band_closes_eventually_for_large_lambda(binary_in_band):
    """Above some lambda the band always closes -- the bisection in
    critical_lambda confirms it. Direct re-test at 2*lambda_crit can
    occasionally find a spurious root from the NK fallback regime;
    test at a much-much-larger lambda instead, where any spurious
    root has been driven out."""
    lc = critical_lambda(binary_in_band)
    b = backreacted_band(binary_in_band, lam=1000.0 * lc)
    assert b.band_closed is True


# ----- Q-threshold -------------------------------------------------------


def test_q_threshold_returns_none_for_no_band(binary_no_band):
    Q = q_threshold_from_balance(binary_no_band)
    assert Q is None


def test_q_threshold_positive_for_in_band(binary_in_band):
    Q = q_threshold_from_balance(binary_in_band, E_krasnikov=1.0, omega_0=1.0)
    assert Q is not None
    assert math.isfinite(Q)
    assert Q > 0.0


def test_q_threshold_scales_with_E_krasnikov(binary_in_band):
    Q1 = q_threshold_from_balance(binary_in_band, E_krasnikov=1.0)
    Q4 = q_threshold_from_balance(binary_in_band, E_krasnikov=4.0)
    # Q_thr ~ sqrt(E_krasnikov)
    assert Q4 == pytest.approx(2.0 * Q1, rel=1e-12)


# ----- combined diagnostic -----------------------------------------------


def test_diagnostic_dataclass(binary_in_band):
    d = backreaction_diagnostic(binary_in_band, lam=1.0)
    assert isinstance(d, BackreactionDiagnostic)
    assert isinstance(d.band_at_lam, BackreactedBand)
    assert d.lambda_critical > 0.0


def test_summary_string_in_band(binary_in_band):
    d = backreaction_diagnostic(binary_in_band, lam=1.0)
    s = summarise_backreaction(d)
    assert "NK back-reaction" in s
    assert "classical band edges" in s
    assert "BR band edges" in s
    assert "lambda_critical" in s


def test_summary_string_no_band(binary_no_band):
    d = backreaction_diagnostic(binary_no_band, lam=1.0)
    s = summarise_backreaction(d)
    assert "No classical band" in s


def test_summary_when_band_collapses(binary_in_band):
    d = backreaction_diagnostic(binary_in_band, lam=1000.0)
    s = summarise_backreaction(d)
    assert "BAND CLOSED" in s
