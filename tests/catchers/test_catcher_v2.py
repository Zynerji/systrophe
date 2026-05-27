"""Unit tests for catcher_v2 (coherence + consensus + multi-run)."""

from __future__ import annotations

import numpy as np
import pytest

from systrophe.catchers.catcher_v2 import (
    scan_novelty_coherent,
    scan_novelty_consensus,
    scan_novelty_multirun,
)


# -------- coherence --------

def test_coherence_detects_sweeping_chirp():
    """A signal that excites adjacent freq bands in sequence (chirp-like)
    should produce high sign-correlated rank shifts → high coherence."""
    p = np.linspace(0, 1, 30)
    rng = np.random.default_rng(7)

    def fn(t):
        # 8-component vector: at t=0.5, a "chirp" pulses through bands
        baseline = rng.normal(0, 0.1, 8)
        if 0.45 < t < 0.55:
            chirp = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 2
            return baseline + chirp
        return baseline

    res = scan_novelty_coherent(p, fn, sharp_threshold=0.10)
    assert res.verdict == "novel_structure", res.verdict
    # Sharp feature should be near t=0.5
    assert any(abs(s["parameter_value"] - 0.5) < 0.1 for s in res.sharp_features)


def test_coherence_quiet_returns_smooth():
    """Pure noise across components should give coherence within noise
    floor. With M=8 components, 7 adjacent pairs, random sign match
    rate has std ~ 1/sqrt(7) ≈ 0.19. Threshold 0.35 is well above the
    noise floor."""
    p = np.linspace(0, 1, 30)
    rng = np.random.default_rng(11)

    def fn(t):
        return rng.normal(0, 1, 8)

    res = scan_novelty_coherent(p, fn, sharp_threshold=0.50)
    assert res.verdict == "smooth"


def test_coherence_min_components_guard():
    """Too few components → no_signal (not a real test of the detector)."""
    p = np.linspace(0, 1, 10)

    def fn(t):
        return np.array([t, t * 2])

    res = scan_novelty_coherent(p, fn, min_components=4)
    assert res.verdict == "no_signal"


# -------- consensus --------

def test_consensus_detects_multi_band_burst():
    """When many components simultaneously exceed 3σ above their median,
    the consensus detector should fire."""
    p = np.linspace(0, 1, 30)
    rng = np.random.default_rng(13)

    def fn(t):
        x = rng.normal(0, 1, 16)
        if 0.45 < t < 0.55:
            x += 5  # all 16 components jump simultaneously
        return x

    res = scan_novelty_consensus(p, fn, z_threshold=3.0, consensus_fraction=0.50)
    assert res.verdict == "novel_structure"
    assert any(abs(s["parameter_value"] - 0.5) < 0.1 for s in res.sharp_features)


def test_consensus_single_component_doesnt_fire():
    """A spike in just ONE component out of 16 should NOT fire
    consensus at 30% fraction."""
    p = np.linspace(0, 1, 30)
    rng = np.random.default_rng(17)

    def fn(t):
        x = rng.normal(0, 1, 16)
        if 0.45 < t < 0.55:
            x[0] += 10  # only one component spikes
        return x

    res = scan_novelty_consensus(p, fn, z_threshold=3.0, consensus_fraction=0.30)
    assert res.verdict == "smooth"


def test_consensus_quiet_noise():
    """Pure noise should not fire consensus."""
    p = np.linspace(0, 1, 30)
    rng = np.random.default_rng(19)

    def fn(t):
        return rng.normal(0, 1, 16)

    res = scan_novelty_consensus(p, fn, z_threshold=3.0, consensus_fraction=0.30)
    assert res.verdict == "smooth"


# -------- multi-run --------

def test_multirun_detects_repeatable_emergent():
    """An emergent at the SAME parameter location across multiple runs
    (different random seeds) should be flagged."""
    p = np.linspace(0, 1, 30)
    fns = []
    for seed in range(5):
        rng = np.random.default_rng(seed)

        def fn(t, _rng=rng):
            x = _rng.normal(0, 1, 16)
            if 0.45 < t < 0.55:
                x += 5
            return x
        fns.append(fn)

    res = scan_novelty_multirun(p, fns, per_run_z_threshold=3.0,
                                  coincidence_fraction=0.6)
    assert res.verdict == "novel_structure"
    assert res.max_coincidence >= 0.6
    assert abs(res.max_coincidence_parameter - 0.5) < 0.1


def test_multirun_random_outliers_dont_fire():
    """If outliers appear at DIFFERENT parameter locations in each run,
    multi-run coincidence at high threshold should NOT fire.

    With 5 runs, parameter_tolerance=1 → each run covers ~3 of 30
    slots randomly. The probability of any single slot being in 4+ of
    5 runs is low; coincidence_fraction=0.8 (4 of 5 runs required)
    rules out random overlap."""
    p = np.linspace(0, 1, 30)
    fns = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        outlier_t = rng.uniform(0.1, 0.9)

        def fn(t, _rng=rng, _outlier=outlier_t):
            x = _rng.normal(0, 1, 16)
            if abs(t - _outlier) < 0.02:
                x += 5
            return x
        fns.append(fn)

    res = scan_novelty_multirun(p, fns, per_run_z_threshold=3.0,
                                  coincidence_fraction=0.8,
                                  parameter_tolerance=0)
    assert res.verdict == "smooth"


def test_multirun_single_run_returns_no_signal():
    """Need at least 2 runs."""
    p = np.linspace(0, 1, 30)

    def fn(t):
        return np.array([t, t * 2])

    res = scan_novelty_multirun(p, [fn])
    assert res.verdict == "no_signal"
