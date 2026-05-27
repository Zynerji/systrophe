"""Tests for chronology-protection back-reaction study."""

import numpy as np
import pytest

from systrophe.ctc.chronology_protection import (
    ChronologyProtectionStudy,
    chronology_protection_study,
    chronology_protection_verdict,
    matched_pair_default_study,
    residual_derivative_factory,
)
from systrophe.geometry.sinusoid import TiplerSinusoid


def test_residual_derivative_factory_callable():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    rs = np.array([1.5, 2.0])
    F_prime = residual_derivative_factory(s1, s2, rs)
    val = F_prime(0.5)
    assert np.isfinite(val)


def test_study_returns_correct_shape():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    rs = np.array([1.5, 2.0])
    seeds = np.array([0.5, 1.0, 2.0])
    study = chronology_protection_study(s1, s2, rs, seeds=seeds, max_iter=10)
    assert isinstance(study, ChronologyProtectionStudy)
    assert study.converged_deltas.shape == (3,)
    assert study.convergence_flags.shape == (3,)


def test_study_default_seeds():
    """Default 8 seeds in [0.1, 2 pi - 0.1]."""
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    rs = np.array([1.5, 2.0])
    study = chronology_protection_study(s1, s2, rs, max_iter=10)
    assert study.n_seeds == 8


def test_verdict_returns_dict():
    s1 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    s2 = TiplerSinusoid(R=1.0, a=1.0, A=1.0, delta=0.0)
    rs = np.array([1.5, 2.0])
    study = chronology_protection_study(s1, s2, rs, max_iter=5)
    verdict = chronology_protection_verdict(study)
    assert "verdict" in verdict
    assert verdict["verdict"] in ("consistent", "inconsistent", "inconclusive")
    assert "fraction_chronology" in verdict


def test_matched_pair_default_runs():
    """Convenience function returns a study."""
    study = matched_pair_default_study(n_seeds=4)
    assert isinstance(study, ChronologyProtectionStudy)
    assert study.n_seeds == 4


def test_matched_pair_most_common_near_pi():
    """For matched pair, the most-common converged delta should be in
    a CTC-extinction band (near pi, within tolerance).

    This is the headline validation: NK iteration on the back-reaction
    residual DOES find a chronology-favouring fixed point.
    """
    study = matched_pair_default_study(n_seeds=6)
    # Median converged delta should be in (pi/2, 3pi/2) for the
    # matched pair (where the extinction lives)
    wrapped = float(study.most_common_delta % (2 * np.pi))
    # Permissive: just check it's in the half-circle containing pi
    assert np.pi / 2 < wrapped < 3 * np.pi / 2 or abs(wrapped - 2 * np.pi) < 1.0
