"""CTC region detector tests."""

import numpy as np
import pytest

from systrophe.ctc import find_ctc_intervals, has_ctc
from systrophe.pair import SystrophePair
from systrophe.sinusoid import TiplerSinusoid


def test_purely_positive_function_has_no_ctc():
    L = lambda r: np.ones_like(np.asarray(r, dtype=float))
    assert not has_ctc(L, 1.0, 10.0)
    assert find_ctc_intervals(L, 1.0, 10.0) == []


def test_purely_negative_function_one_interval():
    L = lambda r: -np.ones_like(np.asarray(r, dtype=float))
    assert has_ctc(L, 1.0, 10.0)
    intervals = find_ctc_intervals(L, 1.0, 10.0)
    assert len(intervals) == 1
    a, b = intervals[0]
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(10.0)


def test_supercritical_sinusoid_has_ctcs():
    """A Tipler sinusoid with cosine sign change must produce CTC intervals."""
    s = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.0, p=0.0)
    L = lambda r: np.asarray(s.L(r))
    assert has_ctc(L, 1.0, 100.0)
    intervals = find_ctc_intervals(L, 1.0, 100.0)
    assert len(intervals) >= 2  # multiple log-periodic CTC bands
    # Each interval must be strictly inside [r_min, r_max] and non-degenerate
    for a, b in intervals:
        assert b > a
        # Function must indeed be negative on the interior of each interval
        mid = 0.5 * (a + b)
        assert s.L(mid) < 0


def test_zero_locations_match_sinusoid_zeros():
    """CTC interval endpoints coincide with sinusoid zeros (cosine roots)."""
    s = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.0, p=0.0)
    expected_zeros = s.zeros(1.0, 50.0)
    L = lambda r: np.asarray(s.L(r))
    intervals = find_ctc_intervals(L, 1.0, 50.0)
    found_zeros = sorted(z for ab in intervals for z in ab)
    # Every found endpoint should be one of the predicted zeros (within bisection tol)
    for z in found_zeros:
        if z in (1.0, 50.0):
            continue  # boundary, not an interior zero
        assert np.min(np.abs(expected_zeros - z)) < 1e-6


def test_destructive_interference_kills_ctcs():
    """Anti-phase pair with matched parameters has L = 0; no CTC intervals."""
    s1 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.0, p=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=np.pi, p=0.0)
    pair = SystrophePair(s1=s1, s2=s2)
    L = lambda r: np.asarray(pair.L(r))
    assert not has_ctc(L, 1.05, 50.0)


def test_offset_pair_produces_modulated_ctcs():
    """Phase-offset pair (delta_2 != delta_1) has a non-trivial CTC pattern."""
    s1 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.0, p=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=np.pi / 4, p=0.0)
    pair = SystrophePair(s1=s1, s2=s2)
    L = lambda r: np.asarray(pair.L(r))
    intervals = find_ctc_intervals(L, 1.05, 50.0)
    assert len(intervals) >= 2
    # Combined sinusoid is itself co-frequency, so the collapsed phase should match
    collapsed = pair.to_single_sinusoid()
    # Pattern of intervals must match the collapsed single sinusoid's CTC pattern
    L_c = lambda r: np.asarray(collapsed.L(r))
    intervals_c = find_ctc_intervals(L_c, 1.05, 50.0)
    assert len(intervals) == len(intervals_c)
