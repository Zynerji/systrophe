"""Surrogate-null false-alarm-rate calibration for the catcher_v2 consensus
family (Systrophe, 2026-06-01).

`catcher_v2` ships three consensus-style detectors that emit a binary
``novel_structure`` / ``smooth`` verdict from a raw statistic compared against a
hard-coded threshold:

  * ``scan_novelty_coherent``   -- cross-coordinate coherence in [-0.5, +0.5];
  * ``scan_novelty_consensus``  -- per-parameter consensus fraction in [0, 1];
  * ``scan_novelty_multirun``   -- multi-run coincidence rate in [0, 1].

None of those thresholds carries a calibrated false-alarm rate (FAR), which
violates the repository's standing "no null = no validation" / novelty-catcher
discipline rule. This module supplies the missing null model: it turns each raw
statistic into a calibrated p-value so a fire/no-fire decision can be made at a
controlled FAR instead of an arbitrary threshold.

Null models (each preserves the relevant marginal so only the *structure*
between components / runs is destroyed):

  coherent  -> trend-preserving **circular-shift** null. Each component column
               is independently circularly shifted (``np.roll``) by a random
               lag. This preserves each column's value distribution AND its
               local autocorrelation / trend (unlike a full rank permutation,
               which would also destroy within-column smoothness and inflate
               coherence variance). It destroys only the *cross-column*
               alignment that the coherence statistic measures.

  consensus -> per-component **circular-shift** null preserving marginals. Each
               component column is circularly shifted independently, so the
               per-component baseline / MAD and the count of large-|z| samples
               per column are preserved, but their parameter-axis coincidence
               (the thing the consensus fraction measures) is destroyed.

  multirun  -> **analytic** binomial null. Each run independently fires in a
               bin with probability ``q`` estimated from the *per-run consensus
               FAR*, smeared over the +-``parameter_tolerance`` window the
               detector applies and gated by the ~10% inner consensus fraction.
               The expected fired-slots-per-run is therefore NOT the naive
               ``1/n_bins``; we estimate ``q = (expected fired slots per run) /
               n_bins`` and feed it to ``scipy.stats.binom.sf`` with a
               Bonferroni correction over the ``n_bins`` simultaneous tests.

Every public entry point returns a small dataclass carrying the raw statistic,
the calibrated p-value, the surrogate-null distribution (where applicable) and a
``fired`` boolean at the requested ``alpha``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from systrophe.catchers.catcher_v2 import (
    scan_novelty_coherent,
    scan_novelty_consensus,
    scan_novelty_multirun,
    _build_output_matrix,
    _stable_rank,
)

__all__ = [
    "CoherenceCalibration",
    "ConsensusCalibration",
    "MultirunCalibration",
    "calibrate_coherent",
    "calibrate_consensus",
    "calibrate_multirun",
    "coherence_statistic_from_matrix",
    "consensus_statistic_from_matrix",
]


# ============================================================
# Low-level statistics computed directly from an output matrix.
#
# These reproduce the raw scalar each detector thresholds on, but operate on a
# pre-built (n_param, n_components) matrix so the surrogate generator can call
# them thousands of times without re-running the (possibly expensive)
# output_fn.
# ============================================================

def coherence_statistic_from_matrix(M: np.ndarray) -> float:
    """Max |coherence| over the parameter axis for matrix ``M``.

    Mirrors ``scan_novelty_coherent``: rank each component down the parameter
    axis, take adjacent-pair sign agreement of the rank deltas, subtract 0.5,
    and return the maximum absolute value. Range [0, 0.5].
    """
    M = np.asarray(M, dtype=float)
    n_param, n_components = M.shape
    if n_param < 2 or n_components < 2:
        return 0.0
    ranks = _stable_rank(M, axis=0)
    delta_ranks = np.diff(ranks, axis=0)
    sign_a = np.sign(delta_ranks[:, :-1])
    sign_b = np.sign(delta_ranks[:, 1:])
    same_sign = ((sign_a * sign_b) > 0).astype(np.float64)
    coherence = same_sign.mean(axis=1) - 0.5
    if coherence.size == 0:
        return 0.0
    return float(np.max(np.abs(coherence)))


def consensus_statistic_from_matrix(
    M: np.ndarray,
    z_threshold: float = 3.0,
    baseline_quartiles: tuple = (0.25, 0.25),
) -> float:
    """Max consensus fraction over the parameter axis for matrix ``M``.

    Mirrors ``scan_novelty_consensus``: per-component robust z-score against a
    median/MAD baseline taken from the outer ``baseline_quartiles`` of the
    parameter axis, then the per-parameter fraction of components exceeding
    ``z_threshold``; returns the maximum. Range [0, 1].
    """
    M = np.asarray(M, dtype=float)
    n_param, n_components = M.shape
    if n_param < 4:
        return 0.0
    front_q, back_q = baseline_quartiles
    n_front = max(1, int(front_q * n_param))
    n_back = max(1, int(back_q * n_param))
    baseline = np.concatenate([M[:n_front], M[-n_back:]], axis=0)
    med = np.median(baseline, axis=0)
    mad = np.median(np.abs(baseline - med), axis=0) + 1e-12
    sigma = 1.4826 * mad
    z = (M - med) / sigma
    consensus = (np.abs(z) > z_threshold).mean(axis=1)
    if consensus.size == 0:
        return 0.0
    return float(np.max(consensus))


# ============================================================
# Surrogate generators (preserve marginals, destroy structure).
# ============================================================

def _circular_shift_columns(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently circularly shift each column by a random non-trivial lag.

    Preserves each column's exact value multiset and its local
    autocorrelation/trend (a circular shift is a relabelling of the parameter
    axis). Destroys the *between-column* phase alignment that the coherence /
    consensus statistics measure. A zero lag would leave the column unchanged,
    so we draw lags in [1, n_param-1] when possible.
    """
    M = np.asarray(M, dtype=float)
    n_param, n_components = M.shape
    out = np.empty_like(M)
    if n_param <= 1:
        return M.copy()
    lags = rng.integers(1, n_param, size=n_components)
    for c in range(n_components):
        out[:, c] = np.roll(M[:, c], int(lags[c]))
    return out


def _rank_permute_columns(M: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently permute each column (full rank permutation null).

    Provided for comparison / completeness. Destroys both cross-column
    alignment AND within-column trend, so it is the more aggressive null. The
    coherence detector is calibrated against the circular-shift null instead
    (see module docstring), but this is exposed for the consensus marginal
    null where within-column trend is irrelevant.
    """
    M = np.asarray(M, dtype=float)
    n_param, n_components = M.shape
    out = np.empty_like(M)
    for c in range(n_components):
        out[:, c] = M[rng.permutation(n_param), c]
    return out


# ============================================================
# Variant 1: coherence calibration (trend-preserving circular-shift null)
# ============================================================

@dataclass
class CoherenceCalibration:
    raw_statistic: float
    p_value: float
    null_distribution: np.ndarray
    n_perm: int
    alpha: float
    fired: bool
    verdict: str = "smooth"
    base_result: object = field(default=None, repr=False)


def calibrate_coherent(
    parameter_values: np.ndarray,
    output_fn: Callable[[float], np.ndarray],
    *,
    alpha: float = 0.05,
    n_perm: int = 1000,
    min_components: int = 4,
    seed: int | None = 0,
) -> CoherenceCalibration:
    """Calibrate the cross-coordinate coherence detector against a
    trend-preserving circular-shift null.

    The raw statistic is ``max |coherence|`` (range [0, 0.5]). Under the null,
    each component column is independently circularly shifted so per-column
    trend is preserved but cross-column alignment is destroyed. The p-value is
    the right-tail rank of the observed statistic among ``n_perm`` surrogates
    (add-one smoothed).
    """
    p = np.asarray(parameter_values, dtype=float)
    M, _ = _build_output_matrix(p, output_fn)
    n_param, n_components = M.shape

    base = scan_novelty_coherent(p, output_fn, min_components=min_components)

    if n_components < min_components or n_param < 2:
        return CoherenceCalibration(
            raw_statistic=0.0,
            p_value=1.0,
            null_distribution=np.zeros(0),
            n_perm=n_perm,
            alpha=alpha,
            fired=False,
            verdict="no_signal",
            base_result=base,
        )

    raw = coherence_statistic_from_matrix(M)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        surr = _circular_shift_columns(M, rng)
        null[i] = coherence_statistic_from_matrix(surr)

    # right-tail, add-one smoothed (never reports p=0)
    p_value = float((np.count_nonzero(null >= raw) + 1) / (n_perm + 1))
    fired = p_value <= alpha
    return CoherenceCalibration(
        raw_statistic=raw,
        p_value=p_value,
        null_distribution=null,
        n_perm=n_perm,
        alpha=alpha,
        fired=fired,
        verdict="novel_structure" if fired else "smooth",
        base_result=base,
    )


# ============================================================
# Variant 2: consensus calibration (marginal-preserving circular-shift null)
# ============================================================

@dataclass
class ConsensusCalibration:
    raw_statistic: float
    p_value: float
    null_distribution: np.ndarray
    n_perm: int
    alpha: float
    fired: bool
    z_threshold: float
    verdict: str = "smooth"
    base_result: object = field(default=None, repr=False)


def calibrate_consensus(
    parameter_values: np.ndarray,
    output_fn: Callable[[float], np.ndarray],
    *,
    z_threshold: float = 3.0,
    baseline_quartiles: tuple = (0.25, 0.25),
    alpha: float = 0.05,
    n_perm: int = 1000,
    seed: int | None = 0,
) -> ConsensusCalibration:
    """Calibrate the per-component consensus detector against a
    marginal-preserving per-component circular-shift null.

    The raw statistic is ``max consensus fraction`` (range [0, 1]). Under the
    null each component column is circularly shifted independently, preserving
    the per-component baseline / MAD and the number of large-|z| samples per
    column, but destroying their parameter-axis coincidence. The p-value is the
    right-tail rank of the observed statistic among ``n_perm`` surrogates
    (add-one smoothed).
    """
    p = np.asarray(parameter_values, dtype=float)
    M, _ = _build_output_matrix(p, output_fn)
    n_param, n_components = M.shape

    base = scan_novelty_consensus(
        p, output_fn, z_threshold=z_threshold,
        baseline_quartiles=baseline_quartiles,
    )

    if n_param < 4:
        return ConsensusCalibration(
            raw_statistic=0.0,
            p_value=1.0,
            null_distribution=np.zeros(0),
            n_perm=n_perm,
            alpha=alpha,
            fired=False,
            z_threshold=z_threshold,
            verdict="no_signal",
            base_result=base,
        )

    raw = consensus_statistic_from_matrix(
        M, z_threshold=z_threshold, baseline_quartiles=baseline_quartiles)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        surr = _circular_shift_columns(M, rng)
        null[i] = consensus_statistic_from_matrix(
            surr, z_threshold=z_threshold, baseline_quartiles=baseline_quartiles)

    p_value = float((np.count_nonzero(null >= raw) + 1) / (n_perm + 1))
    fired = p_value <= alpha
    return ConsensusCalibration(
        raw_statistic=raw,
        p_value=p_value,
        null_distribution=null,
        n_perm=n_perm,
        alpha=alpha,
        fired=fired,
        z_threshold=z_threshold,
        verdict="novel_structure" if fired else "smooth",
        base_result=base,
    )


# ============================================================
# Variant 3: multi-run coincidence calibration (analytic binomial null)
# ============================================================

@dataclass
class MultirunCalibration:
    coincidence_rate: np.ndarray
    parameter_axis: np.ndarray
    per_run_fire_prob: float          # q: per-bin fire probability for ONE run
    per_run_consensus_far: float      # the consensus FAR each run is gated on
    n_runs: int
    n_bins: int
    parameter_tolerance: int
    coincidence_fraction: float
    min_runs_fired: int               # binom threshold (ceil(frac * n_runs))
    p_value_per_bin: np.ndarray       # uncorrected binom.sf per bin
    p_value_min: float                # smallest per-bin p (Bonferroni input)
    p_value_family: float             # Bonferroni-corrected family p-value
    expected_far: float               # analytic family-wise false-fire prob
    alpha: float
    fired: bool
    verdict: str = "smooth"
    base_result: object = field(default=None, repr=False)


def estimate_per_run_fire_prob(
    per_run_consensus_far: float,
    n_bins: int,
    parameter_tolerance: int = 1,
) -> float:
    """Per-bin fire probability ``q`` for a single run.

    The multirun detector does NOT mark exactly one bin per chance consensus
    hit: each hit at parameter index ``i`` is smeared over the window
    ``[i - tol, i + tol]`` (width ``2*tol + 1``) by ``scan_novelty_multirun``.
    So the expected number of fired *slots* per run is

        E[fired slots] = (expected consensus hits per run) * window_width
                       = (per_run_consensus_far * n_bins) * (2*tol + 1)

    and the per-bin fire probability is that divided by ``n_bins``:

        q = per_run_consensus_far * (2*tol + 1)

    capped at 1. This is the quantity that makes the binomial null correct;
    the naive ``1/n_bins`` ignores both the smearing window and the inner ~10%
    consensus gate baked into ``per_run_consensus_far``.
    """
    window = 2 * int(parameter_tolerance) + 1
    q = per_run_consensus_far * window
    return float(min(max(q, 0.0), 1.0))


def calibrate_multirun(
    parameter_values: np.ndarray,
    *,
    per_run_consensus_far: float,
    n_runs: int,
    coincidence_fraction: float = 0.5,
    parameter_tolerance: int = 1,
    alpha: float = 0.05,
    coincidence_rate: np.ndarray | None = None,
    base_result: object = None,
) -> MultirunCalibration:
    """Analytic binomial calibration of the multi-run coincidence detector.

    Given the per-run consensus FAR (the probability that *one* run's inner
    consensus stage fires in a given bin -- obtainable from
    :func:`calibrate_consensus` or directly measured), model the number of runs
    that fire in a fixed bin as ``Binomial(n_runs, q)`` with
    ``q = estimate_per_run_fire_prob(...)``.

    A bin is flagged when ``>= ceil(coincidence_fraction * n_runs)`` runs fire.
    The per-bin false-fire probability is ``binom.sf(k-1, n_runs, q)``; with
    ``n_bins`` simultaneous bins the family-wise false-alarm rate is bounded by
    Bonferroni: ``min(1, n_bins * per_bin_p)``.

    If an observed ``coincidence_rate`` array is supplied the calibration also
    reports a family p-value for the *observed* maximum coincidence; otherwise
    it reports the analytic FAR at the detector's decision rule (useful for
    pure null calibration where no run actually fired).
    """
    p = np.asarray(parameter_values, dtype=float)
    n_bins = len(p)
    q = estimate_per_run_fire_prob(
        per_run_consensus_far, n_bins, parameter_tolerance)

    # Decision rule: >= this many runs must fire in a (smeared) bin.
    min_runs_fired = int(np.ceil(coincidence_fraction * n_runs))
    min_runs_fired = max(1, min_runs_fired)

    # Per-bin false-fire probability under the binomial null.
    per_bin_p = float(stats.binom.sf(min_runs_fired - 1, n_runs, q))
    # Analytic family-wise FAR (Bonferroni upper bound).
    expected_far = float(min(1.0, n_bins * per_bin_p))

    if coincidence_rate is not None:
        cr = np.asarray(coincidence_rate, dtype=float)
        # observed fired-run count per bin
        k_obs = np.rint(cr * n_runs).astype(int)
        p_per_bin = stats.binom.sf(np.maximum(k_obs - 1, -1), n_runs, q)
        p_per_bin = np.asarray(p_per_bin, dtype=float)
        p_min = float(np.min(p_per_bin)) if p_per_bin.size else 1.0
        p_family = float(min(1.0, n_bins * p_min))
    else:
        p_per_bin = np.full(n_bins, per_bin_p, dtype=float)
        p_min = per_bin_p
        p_family = expected_far

    fired = p_family <= alpha
    return MultirunCalibration(
        coincidence_rate=(np.asarray(coincidence_rate, dtype=float)
                          if coincidence_rate is not None
                          else np.zeros(n_bins)),
        parameter_axis=p,
        per_run_fire_prob=q,
        per_run_consensus_far=float(per_run_consensus_far),
        n_runs=int(n_runs),
        n_bins=int(n_bins),
        parameter_tolerance=int(parameter_tolerance),
        coincidence_fraction=float(coincidence_fraction),
        min_runs_fired=min_runs_fired,
        p_value_per_bin=p_per_bin,
        p_value_min=p_min,
        p_value_family=p_family,
        expected_far=expected_far,
        alpha=alpha,
        fired=fired,
        verdict="novel_structure" if fired else "smooth",
        base_result=base_result,
    )


def measure_per_run_consensus_far(
    parameter_values: np.ndarray,
    output_fn_factory: Callable[[np.random.Generator], Callable[[float], np.ndarray]],
    *,
    inner_consensus_fraction: float = 0.10,
    per_run_z_threshold: float = 3.0,
    n_trials: int = 200,
    seed: int | None = 0,
) -> float:
    """Empirically measure the per-bin per-run consensus FAR.

    ``output_fn_factory(rng)`` must return a fresh pure-null output function
    (no planted feature) seeded from ``rng``. We run the *inner* consensus
    stage exactly as ``scan_novelty_multirun`` does -- z_threshold and the
    ~10% inner consensus gate -- and report the mean fraction of parameter
    bins that fire across ``n_trials`` independent null runs.

    This is the ``per_run_consensus_far`` to pass to :func:`calibrate_multirun`.
    Note it counts pre-smear consensus hits (parameter_index of sharp_features);
    the +-tolerance smearing is applied analytically inside
    :func:`estimate_per_run_fire_prob`.
    """
    p = np.asarray(parameter_values, dtype=float)
    n_bins = len(p)
    rng = np.random.default_rng(seed)
    total_hits = 0
    for _ in range(n_trials):
        fn = output_fn_factory(rng)
        res = scan_novelty_consensus(
            p, fn, z_threshold=per_run_z_threshold,
            consensus_fraction=inner_consensus_fraction,
        )
        total_hits += len(res.sharp_features)
    return float(total_hits / (n_trials * n_bins))
