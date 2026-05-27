"""Tests for back-reaction self-consistency module."""

import numpy as np
import pytest

from systrophe.qftcs.back_reaction import (
    BackReactionLandscape,
    back_reaction_landscape,
    chronology_favoring_deltas,
    compare_to_ctc_extinction,
    optimal_delta_by_NK,
    pair_back_reaction_residual,
    pair_back_reaction_residual_at_r,
)
from systrophe.geometry.pair import SystrophePair
from systrophe.geometry.sinusoid import TiplerSinusoid


@pytest.fixture
def pair():
    """Standard SystrophePair for testing."""
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=np.pi)
    return SystrophePair(s1=s1, s2=s2)


# ----- residual at single point -----------------------------------------

def test_residual_at_r_finite_and_positive(pair):
    r_val = pair_back_reaction_residual_at_r(pair, r=2.0)
    assert np.isfinite(r_val)
    assert r_val >= 0


def test_residual_sums_over_samples(pair):
    """Composite residual: T_part + L_part across samples."""
    rs = np.array([1.5, 2.0, 2.5])
    total = pair_back_reaction_residual(pair, r_samples=rs)
    T_components = [pair_back_reaction_residual_at_r(pair, float(r)) for r in rs]
    L_components = [abs(float(pair.L(float(r)))) for r in rs]
    expected = sum(T_components) + sum(L_components)
    assert total == pytest.approx(expected, rel=1e-12)


def test_residual_with_L_only_weight(pair):
    """L_weight only: residual equals total |L_pair| across samples."""
    rs = np.array([1.5, 2.0, 2.5])
    total = pair_back_reaction_residual(pair, r_samples=rs,
                                         L_weight=1.0, T_weight=0.0)
    L_components = [abs(float(pair.L(float(r)))) for r in rs]
    assert total == pytest.approx(sum(L_components), rel=1e-12)


# ----- landscape sweep --------------------------------------------------

def test_landscape_returns_correct_shape():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    deltas = np.linspace(0, 2 * np.pi, 12)
    rs = np.array([1.5, 2.0])
    lndscp = back_reaction_landscape(s1, s2, deltas, rs)
    assert lndscp.deltas.shape == deltas.shape
    assert lndscp.residuals.shape == deltas.shape
    assert lndscp.min_residual <= lndscp.max_residual


def test_landscape_residuals_finite():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    lndscp = back_reaction_landscape(
        s1, s2, np.linspace(0, np.pi, 6), np.array([1.5, 2.0]),
    )
    assert np.all(np.isfinite(lndscp.residuals))


# ----- chronology-favouring -------------------------------------------

def test_chronology_favoring_returns_subset():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    deltas = np.linspace(0, 2 * np.pi, 20)
    rs = np.array([1.5, 2.0])
    lndscp = back_reaction_landscape(s1, s2, deltas, rs)
    favoring = chronology_favoring_deltas(lndscp, quantile=0.2)
    # bottom 20%, so ~4 of 20
    assert 1 <= len(favoring) <= 6


def test_chronology_favoring_quantile_validation():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    deltas = np.array([0.0, np.pi])
    lndscp = back_reaction_landscape(s1, s2, deltas, np.array([2.0]))
    with pytest.raises(ValueError):
        chronology_favoring_deltas(lndscp, quantile=1.5)


# ----- comparison to CTC extinction -----------------------------------

def test_compare_to_ctc_extinction_basic():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    deltas = np.linspace(0, 2 * np.pi, 10)
    cmp = compare_to_ctc_extinction(s1, s2, deltas)
    assert "landscape" in cmp
    assert "br_min_delta" in cmp
    assert "ctc_min_delta" in cmp
    assert cmp["alignment_delta"] >= 0.0


def test_ctc_min_delta_is_pi():
    """The CTC log-measure minimum for matched-pair is delta = pi
    (anti-phase extinction). This is the existing v0.11 verified result."""
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    deltas = np.linspace(0, 2 * np.pi, 21)  # finer grid; includes pi
    cmp = compare_to_ctc_extinction(s1, s2, deltas)
    # delta = pi should be in the 0.2 quantile of CTC measure
    ctc_min = cmp["ctc_min_delta"]
    # Allow wrap (the minimum could be near 0 or 2*pi due to wrapping)
    # On this grid it should be ~ pi
    assert abs(ctc_min - np.pi) < 0.5
