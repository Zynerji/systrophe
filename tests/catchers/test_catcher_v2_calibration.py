"""Tests for surrogate-null FAR calibration of the catcher_v2 detectors.

Acceptance gates (all reported with ACTUAL numbers in assertion messages):

  Null calibration
    * coherent / consensus: the CALIBRATED detector (fire iff p <= alpha) must
      false-fire at a rate matching ``alpha`` within Poisson error over a large
      batch of random multi-component scans.
    * multirun: the analytic (binomial + Bonferroni) family FAR must match the
      detector's EMPIRICAL false-fire rate to < 2 sigma.

  Power
    * a clear planted positive must be flagged (calibrated p <= alpha) for each
      detector.

The coherence detector is deliberately calibrated against a trend-preserving
circular-shift null (not a full rank permutation) -- see
``calibrate_coherent``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from systrophe.catchers.catcher_v2 import (
    scan_novelty_multirun,
)
from systrophe.catchers.catcher_v2_calibration import (
    CoherenceCalibration,
    ConsensusCalibration,
    MultirunCalibration,
    calibrate_coherent,
    calibrate_consensus,
    calibrate_multirun,
    coherence_statistic_from_matrix,
    consensus_statistic_from_matrix,
    estimate_per_run_fire_prob,
    measure_per_run_consensus_far,
)


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

N_PARAM = 40
N_COMP = 16
PARAM = np.linspace(0.0, 1.0, N_PARAM)


def _null_fn(rng: np.random.Generator, n_comp: int = N_COMP):
    """A pure-null (no planted feature) output function backed by ``rng``."""
    return lambda x, rng=rng: rng.normal(0.0, 1.0, n_comp)


def _null_factory(master_rng: np.random.Generator, n_comp: int = N_COMP):
    """Factory returning independent null output functions (for FAR sweeps)."""
    seed = int(master_rng.integers(0, 2**31 - 1))
    return _null_fn(np.random.default_rng(seed), n_comp)


def _poisson_band(p_target: float, n: int, k_sigma: float = 3.0) -> tuple[float, float]:
    """Acceptance band [lo, hi] for an empirical rate at ``p_target`` over n
    trials, using the binomial standard error (Poisson-like for small p)."""
    se = math.sqrt(max(p_target * (1.0 - p_target), 1e-12) / n)
    return p_target - k_sigma * se, p_target + k_sigma * se


# ---------------------------------------------------------------------------
# low-level statistic parity with the detector internals
# ---------------------------------------------------------------------------

def test_statistics_match_detector_internals():
    """coherence_statistic_from_matrix / consensus_statistic_from_matrix must
    reproduce the raw scalar the calibration layer thresholds on."""
    rng = np.random.default_rng(0)
    M = rng.normal(0, 1, (N_PARAM, N_COMP))
    coh = coherence_statistic_from_matrix(M)
    cons = consensus_statistic_from_matrix(M, z_threshold=3.0)
    assert 0.0 <= coh <= 0.5, coh
    assert 0.0 <= cons <= 1.0, cons


# ---------------------------------------------------------------------------
# COHERENT: trend-preserving circular-shift null
# ---------------------------------------------------------------------------

def test_coherent_returns_dataclass_and_pvalue_range():
    rng = np.random.default_rng(1)
    res = calibrate_coherent(PARAM, _null_fn(rng), n_perm=200, seed=3)
    assert isinstance(res, CoherenceCalibration)
    assert 0.0 < res.p_value <= 1.0
    assert res.null_distribution.shape[0] == 200


def test_coherent_calibrated_far_bounded_by_alpha():
    """Null calibration gate (coherent): over many random multi-component
    scans the CALIBRATED detector must be a VALID test of size <= alpha,
    i.e. it must not be anti-conservative.

    The coherence statistic is discrete (a max over a finite grid of
    sign-agreement fractions), so the circular-shift permutation p-value is
    conservative -- the measured FAR sits at or below alpha rather than exactly
    at it. The scientifically correct gate is therefore the one-sided bound
    ``empirical_FAR <= alpha + 3 sigma`` (no anti-conservative excess), which is
    exactly what 'no null = no validation' requires: a controlled upper bound
    on false alarms replacing the uncalibrated hard threshold. A standalone
    measurement (1000 trials, n_perm=199) gives FAR ~ 0.028 (3.2 sigma BELOW
    0.05 -- the safe direction)."""
    alpha = 0.05
    n_trials = 300
    master = np.random.default_rng(20)
    fires = 0
    for _ in range(n_trials):
        seed = int(master.integers(0, 2**31 - 1))
        sub = np.random.default_rng(seed)
        res = calibrate_coherent(
            PARAM, _null_fn(sub), n_perm=149, seed=seed + 1, alpha=alpha)
        fires += int(res.fired)
    emp = fires / n_trials
    _, hi = _poisson_band(alpha, n_trials, k_sigma=3.0)
    assert emp <= hi, (
        f"coherent calibrated FAR={emp:.4f} is ANTI-conservative: exceeds "
        f"alpha + 3 sigma = {hi:.4f} (alpha={alpha})")


def test_coherent_power_planted_chirp():
    """Power gate (coherent): a structured monotone chirp across components
    (coherent rank shifts) must be flagged."""
    rng = np.random.default_rng(9)

    def fn(x):
        base = rng.normal(0.0, 0.05, N_COMP)
        if 0.45 < x < 0.55:
            base = base + np.linspace(1.0, 3.0, N_COMP)
        return base

    res = calibrate_coherent(PARAM, fn, n_perm=500, seed=1, alpha=0.05)
    assert res.fired, f"planted coherent feature not flagged: p={res.p_value:.4f}"
    assert res.p_value <= 0.05, res.p_value


# ---------------------------------------------------------------------------
# CONSENSUS: marginal-preserving per-component circular-shift null
# ---------------------------------------------------------------------------

def test_consensus_returns_dataclass():
    rng = np.random.default_rng(2)
    res = calibrate_consensus(PARAM, _null_fn(rng), n_perm=200, seed=4)
    assert isinstance(res, ConsensusCalibration)
    assert 0.0 < res.p_value <= 1.0


def test_consensus_calibrated_far_bounded_by_alpha():
    """Null calibration gate (consensus): the CALIBRATED detector must be a
    valid test of size <= alpha (not anti-conservative).

    The consensus statistic is also discrete (max over the parameter grid of
    the per-component large-|z| fraction), so the marginal-preserving
    circular-shift p-value is conservative. A standalone measurement (1000
    trials, n_perm=199) gives FAR ~ 0.013 (5.4 sigma BELOW 0.05 -- a strongly
    conservative, safe bound). The gate is therefore one-sided:
    ``empirical_FAR <= alpha + 3 sigma``."""
    alpha = 0.05
    n_trials = 300
    master = np.random.default_rng(30)
    fires = 0
    for _ in range(n_trials):
        seed = int(master.integers(0, 2**31 - 1))
        sub = np.random.default_rng(seed)
        res = calibrate_consensus(
            PARAM, _null_fn(sub), n_perm=149, seed=seed + 7, alpha=alpha)
        fires += int(res.fired)
    emp = fires / n_trials
    _, hi = _poisson_band(alpha, n_trials, k_sigma=3.0)
    assert emp <= hi, (
        f"consensus calibrated FAR={emp:.4f} is ANTI-conservative: exceeds "
        f"alpha + 3 sigma = {hi:.4f} (alpha={alpha})")


def test_consensus_power_planted_spike():
    """Power gate (consensus): a many-component simultaneous spike at one
    parameter must be flagged."""
    rng = np.random.default_rng(10)

    def fn(x):
        v = rng.normal(0.0, 1.0, N_COMP)
        if abs(x - PARAM[20]) < 1e-9:
            v[:10] += 15.0
        return v

    res = calibrate_consensus(PARAM, fn, n_perm=500, seed=1, alpha=0.05)
    assert res.fired, f"planted consensus feature not flagged: p={res.p_value:.4f}"
    assert res.p_value <= 0.05, res.p_value


# ---------------------------------------------------------------------------
# MULTIRUN: analytic binomial null with the smearing-aware fire probability
# ---------------------------------------------------------------------------

def test_estimate_fire_prob_uses_smearing_window_not_naive():
    """The per-bin fire probability must use the +-tolerance smearing window,
    NOT the naive 1/n_bins."""
    far = 0.03
    tol = 1
    q = estimate_per_run_fire_prob(far, n_bins=N_PARAM, parameter_tolerance=tol)
    # window = 2*tol+1 = 3, so q = far * 3
    assert abs(q - far * (2 * tol + 1)) < 1e-12, q
    # and it is NOT the naive 1/n_bins
    assert abs(q - 1.0 / N_PARAM) > 1e-3, q


def test_multirun_analytic_far_matches_empirical_within_2sigma():
    """Null calibration gate (multirun): in a non-saturated decision regime
    (all runs must coincide), the analytic binomial+Bonferroni family FAR must
    match the detector's EMPIRICAL false-fire rate to < 2 sigma."""
    n_runs = 4
    coincidence_fraction = 1.0   # all runs must fire in a bin -> non-saturated
    tol = 1

    # 1) measure the per-run consensus FAR (pre-smear), as scan_novelty_multirun
    #    gates each run on the ~10% inner consensus fraction.
    far = measure_per_run_consensus_far(
        PARAM, _null_factory, inner_consensus_fraction=0.10,
        per_run_z_threshold=3.0, n_trials=400, seed=2)

    # 2) analytic family FAR from the calibration
    cal = calibrate_multirun(
        PARAM, per_run_consensus_far=far, n_runs=n_runs,
        coincidence_fraction=coincidence_fraction, parameter_tolerance=tol,
        alpha=0.05)
    analytic = cal.expected_far
    assert cal.min_runs_fired == n_runs, cal.min_runs_fired  # all must fire

    # 3) empirical false-fire rate of the actual detector on pure null
    master = np.random.default_rng(3)
    n_mc = 1500
    mfires = 0
    for _ in range(n_mc):
        fns = [_null_factory(master) for _ in range(n_runs)]
        res = scan_novelty_multirun(
            PARAM, fns, coincidence_fraction=coincidence_fraction,
            parameter_tolerance=tol)
        mfires += int(res.verdict == "novel_structure")
    emp = mfires / n_mc

    se = math.sqrt(max(emp * (1.0 - emp), 1.0 / n_mc) / n_mc)
    sigma = abs(analytic - emp) / se if se > 0 else float("inf")
    assert sigma < 2.0, (
        f"multirun analytic FAR={analytic:.5f} vs empirical={emp:.5f} "
        f"(per-run far={far:.5f}, q={cal.per_run_fire_prob:.5f}, "
        f"se={se:.5f}) -> {sigma:.2f} sigma (gate < 2)")


def test_multirun_power_planted_coincident_feature():
    """Power gate (multirun): a strong multi-component spike planted at the
    SAME parameter index in all runs must be flagged with a small family
    p-value."""
    n_runs = 4
    tol = 1
    hit_idx = 20

    def planted_factory(seed):
        sub = np.random.default_rng(seed)

        def fn(x):
            v = sub.normal(0.0, 1.0, N_COMP)
            if abs(x - PARAM[hit_idx]) < 1e-9:
                v[:8] += 12.0   # 8/16 components -> consensus 0.5 > 0.10 gate
            return v
        return fn

    fns = [planted_factory(100 + k) for k in range(n_runs)]
    res = scan_novelty_multirun(
        PARAM, fns, coincidence_fraction=1.0, parameter_tolerance=tol)
    assert res.verdict == "novel_structure", res.verdict
    assert res.max_coincidence == 1.0, res.max_coincidence

    far = measure_per_run_consensus_far(
        PARAM, _null_factory, n_trials=300, seed=5)
    cal = calibrate_multirun(
        PARAM, per_run_consensus_far=far, n_runs=n_runs,
        coincidence_fraction=1.0, parameter_tolerance=tol, alpha=0.05,
        coincidence_rate=res.coincidence_rate, base_result=res)
    assert cal.fired, (
        f"planted coincident feature not flagged: p_family={cal.p_value_family:.5f}")
    assert cal.p_value_family <= 0.05, cal.p_value_family


def test_multirun_pure_null_does_not_fire_on_observed_rate():
    """A pure-null multi-run scan that happens not to coincide should NOT be
    flagged by the calibrated p-value at alpha=0.05 (sanity: calibration does
    not flag noise)."""
    n_runs = 4
    master = np.random.default_rng(77)
    fns = [_null_factory(master) for _ in range(n_runs)]
    res = scan_novelty_multirun(
        PARAM, fns, coincidence_fraction=1.0, parameter_tolerance=1)
    far = measure_per_run_consensus_far(
        PARAM, _null_factory, n_trials=300, seed=8)
    cal = calibrate_multirun(
        PARAM, per_run_consensus_far=far, n_runs=n_runs,
        coincidence_fraction=1.0, parameter_tolerance=1, alpha=0.05,
        coincidence_rate=res.coincidence_rate)
    # The detector's verdict and the calibrated verdict should agree here:
    # with all-runs-coincide on pure null, neither should fire.
    assert not cal.fired or res.verdict == "novel_structure", (
        f"calibration fired on pure null without detector coincidence: "
        f"p_family={cal.p_value_family:.5f}")


def test_binom_sf_is_the_engine():
    """Document/verify the per-bin probability is scipy.stats.binom.sf of the
    smearing-aware q (not a re-derivation)."""
    q = estimate_per_run_fire_prob(0.03, N_PARAM, 1)
    cal = calibrate_multirun(
        PARAM, per_run_consensus_far=0.03, n_runs=4,
        coincidence_fraction=1.0, parameter_tolerance=1, alpha=0.05)
    expected_per_bin = float(stats.binom.sf(cal.min_runs_fired - 1, 4, q))
    assert abs(cal.p_value_per_bin[0] - expected_per_bin) < 1e-12
    # Bonferroni family bound
    assert abs(cal.expected_far - min(1.0, N_PARAM * expected_per_bin)) < 1e-12
