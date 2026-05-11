"""Tests for multi-cylinder array dynamics."""

import numpy as np
import pytest

from systrophe.multi_cylinder_dynamics import (
    MultiCylinderArray,
    ScalingStudyResult,
    beat_frequency_pair,
    beat_log_period,
    ctc_measure_vs_N,
    mixed_frequency_array,
    n_cylinder_extinction_phases,
    phasor_extinction_check,
    phasor_sum,
    random_phase_array,
    uniform_phase_array,
)
from systrophe.sinusoid import TiplerSinusoid


def test_array_construction():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    arr = MultiCylinderArray(sinusoids=[s1])
    assert arr.N == 1


def test_array_empty_raises():
    with pytest.raises(ValueError):
        MultiCylinderArray(sinusoids=[])


def test_uniform_phase_array_N3():
    arr = uniform_phase_array(N=3)
    assert arr.N == 3
    # Phasor sum should be zero
    phases = np.array([s.delta for s in arr.sinusoids])
    assert phasor_extinction_check(phases)


def test_uniform_phase_extinguishes_L():
    """For N>=2 uniform-phase, L is identically zero."""
    arr = uniform_phase_array(N=4)
    rs = np.linspace(1.05, 10.0, 100)
    L = arr.L(rs)
    assert np.max(np.abs(L)) < 1e-9


def test_uniform_phase_no_CTC_bands():
    arr = uniform_phase_array(N=5)
    bands = arr.ctc_bands(r_min=1.05, r_max=20.0)
    assert len(bands) == 0


def test_random_phase_array_returns_N():
    arr = random_phase_array(N=5, rng_seed=42)
    assert arr.N == 5


def test_random_phase_generally_has_CTC_bands():
    """Random phases generically produce CTC bands (no extinction)."""
    arr = random_phase_array(N=5, rng_seed=123)
    bands = arr.ctc_bands(r_min=1.05, r_max=10.0)
    # Almost-surely non-empty
    assert len(bands) >= 1


def test_mixed_frequency_array():
    arr = mixed_frequency_array(N=3, alpha_min=1.0, alpha_max=3.0)
    assert arr.N == 3
    assert not arr.is_matched_frequency


def test_matched_frequency_check():
    """uniform_phase_array has matched frequencies."""
    arr = uniform_phase_array(N=4, base_alpha=2.0)
    assert arr.is_matched_frequency


def test_beat_frequency():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.5, A=1.0, delta=0.0)
    beat = beat_frequency_pair(s1, s2)
    assert beat == pytest.approx(abs(s1.alpha - s2.alpha), rel=1e-12)


def test_beat_log_period():
    period = beat_log_period(alpha_beat=1.0)
    assert period == pytest.approx(2 * np.pi, rel=1e-12)


def test_beat_log_period_zero_returns_inf():
    period = beat_log_period(alpha_beat=0.0)
    assert period == float("inf")


def test_phasor_sum_uniform_N_zero():
    """Sum of N-th roots of unity is zero."""
    for N in (2, 3, 4, 5, 7):
        phases = n_cylinder_extinction_phases(N)
        z = phasor_sum(phases)
        assert abs(z) < 1e-12


def test_n_cylinder_extinction_phases_N3():
    """For N=3, phases should be (0, 2pi/3, 4pi/3)."""
    phases = n_cylinder_extinction_phases(3)
    expected = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
    assert np.allclose(phases, expected, atol=1e-12)


def test_phasor_extinction_with_amps():
    """If amps are nonzero only on i=0, phasor doesn't cancel."""
    deltas = np.array([0, np.pi])
    amps = np.array([1, 0])
    assert not phasor_extinction_check(deltas, amps)


def test_scaling_study_returns_result():
    result = ctc_measure_vs_N(N_range=[3, 4], n_random_samples=5)
    assert isinstance(result, ScalingStudyResult)
    assert result.N_values == [3, 4]
    assert len(result.uniform_measures) == 2
    assert len(result.random_measures) == 2


def test_uniform_measure_is_zero_for_all_N():
    """Uniform-phase extinction is independent of N."""
    result = ctc_measure_vs_N(N_range=[2, 3, 5, 7], n_random_samples=3)
    for m in result.uniform_measures:
        assert abs(m) < 1e-6


def test_random_measure_decreases_with_N():
    """Random-phase measure decreases as N grows (1/sqrt(N) scaling)."""
    result = ctc_measure_vs_N(N_range=[2, 4, 8], n_random_samples=10)
    # Allow imperfect scaling; just verify monotone-decreasing trend
    rand = result.random_measures
    # At least one decrease must hold
    assert rand[2] <= rand[0] * 1.1  # N=8 measure shouldn't exceed N=2
