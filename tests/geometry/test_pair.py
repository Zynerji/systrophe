"""Two-cylinder superposition tests."""

import numpy as np
import pytest

from systrophe.geometry.pair import SystrophePair
from systrophe.geometry.sinusoid import TiplerSinusoid


def test_pair_reduces_to_single_when_amplitude_zero():
    """Setting s2 amplitude to zero recovers s1 exactly."""
    s1 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.3)
    s2 = TiplerSinusoid(R=1.0, a=1.5, A=0.0, delta=0.7)  # zero amplitude
    pair = SystrophePair(s1=s1, s2=s2)
    rs = np.linspace(1.5, 5.0, 50)
    np.testing.assert_allclose(pair.L(rs), s1.L(rs))


def test_co_frequency_pair_collapses_to_single_sinusoid():
    """When alpha_beat = 0, pair == single sinusoid via phasor sum."""
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=np.pi / 3)
    pair = SystrophePair(s1=s1, s2=s2)
    collapsed = pair.to_single_sinusoid()
    rs = np.linspace(1.5, 5.0, 50)
    np.testing.assert_allclose(pair.L(rs), collapsed.L(rs), rtol=1e-12, atol=1e-14)


def test_anti_phase_destructive_interference():
    """delta_2 = delta_1 + pi with identical (R, a, A, p) -> exact cancellation."""
    s1 = TiplerSinusoid(R=1.0, a=1.7, A=0.5, delta=0.2, p=1.0)
    s2 = TiplerSinusoid(R=1.0, a=1.7, A=0.5, delta=0.2 + np.pi, p=1.0)
    pair = SystrophePair(s1=s1, s2=s2)
    rs = np.linspace(1.5, 10.0, 50)
    np.testing.assert_allclose(pair.L(rs), 0.0, atol=1e-13)


def test_phase_offset_wraps_to_principal_range():
    s1 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=3 * np.pi)  # equivalent to pi
    pair = SystrophePair(s1=s1, s2=s2)
    assert -np.pi < pair.phase_offset <= np.pi
    assert pair.phase_offset == pytest.approx(np.pi)


def test_alpha_beat_nonzero_for_unequal_a():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)  # alpha = sqrt(3)
    s2 = TiplerSinusoid(R=1.0, a=2.0, A=1.0, delta=0.0)  # alpha = sqrt(15)
    pair = SystrophePair(s1=s1, s2=s2)
    assert pair.alpha_beat == pytest.approx(np.sqrt(15.0) - np.sqrt(3.0))


def test_collapse_refuses_when_alpha_beat_nonzero():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=2.0, A=1.0, delta=0.0)
    pair = SystrophePair(s1=s1, s2=s2)
    with pytest.raises(ValueError):
        pair.to_single_sinusoid()


def test_subcritical_source_rejected():
    s_super = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.0)
    # cannot even construct a subcritical TiplerSinusoid; use a contrived bypass:
    # here we just confirm construction guard via direct supercritical-only TiplerSinusoid
    with pytest.raises(ValueError):
        TiplerSinusoid(R=1.0, a=0.3, A=1.0, delta=0.0)
    # And SystrophePair only accepts supercritical inputs by construction; covered above.
    _ = SystrophePair(s1=s_super, s2=s_super)  # smoke test
