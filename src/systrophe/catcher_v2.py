"""Catcher v2: cross-coordinate coherence, per-component consensus, and
multi-run coincidence detectors back-ported from the GW unmodeled-burst
catcher (Systrophe gravitational-wave investigation, 2026-05-12).

Three new detection statistics that extend the base address-space
catcher's domain to:

  1. scan_novelty_coherent(parameters, output_fn) -- detect emergents
     via SIGN-CORRELATED rank shifts across adjacent components of a
     vector output. Distinguishes structured (coherent) shifts from
     random rank flips.

  2. scan_novelty_consensus(parameters, output_fn) -- detect emergents
     when MULTIPLE components simultaneously exceed an N-sigma z-score
     above their per-component baseline. Robust against single-
     component noise spikes.

  3. scan_novelty_multirun(parameters, output_fn_list) -- detect
     emergents that appear at the SAME parameter location across most
     of N independent runs. Distinguishes real structure from
     stochastic flukes.

All three return verdicts of the form
  {smooth, novel_structure, no_signal}
plus per-parameter trajectories and confidence estimates.

These are NOT replacements for the base scan_novelty; they are
complementary detectors that fire on signal types the base catcher
misses. The recommended pipeline is to run BOTH base catcher and one
or more v2 detectors; an emergent is flagged when it fires in any of
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np


# ============================================================
# Common helpers
# ============================================================

def _stable_rank(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Stable rank along the given axis. ties broken deterministically."""
    return np.argsort(np.argsort(values, axis=axis, kind="stable"),
                        axis=axis, kind="stable").astype(np.float32)


def _build_output_matrix(
    parameter_values: np.ndarray, output_fn: Callable
) -> tuple[np.ndarray, list]:
    """Run output_fn at every parameter and stack into (n_param, n_components)."""
    raw = [np.asarray(output_fn(float(p)), dtype=float).ravel()
            for p in parameter_values]
    n_components = max((len(r) for r in raw), default=1)
    M = np.zeros((len(raw), n_components), dtype=float)
    for i, r in enumerate(raw):
        M[i, :len(r)] = r
    return M, raw


# ============================================================
# Variant 1: Cross-coordinate coherence
# ============================================================

@dataclass
class CoherenceResult:
    parameter_axis: np.ndarray
    coherence_trajectory: np.ndarray
    sharp_features: list = field(default_factory=list)
    max_coherence: float = 0.0
    max_coherence_parameter: float = 0.0
    verdict: str = "smooth"


def scan_novelty_coherent(
    parameter_values: np.ndarray,
    output_fn: Callable[[float], np.ndarray],
    sharp_threshold: float = 0.15,
    min_components: int = 4,
) -> CoherenceResult:
    """Detect emergents via sign-correlated rank shifts across adjacent
    components of a multi-dim output.

    For each parameter transition i -> i+1:
      - delta_rank[i, c] = rank[i+1, c] - rank[i, c] for each component c
      - coherence[i] = (# of adjacent (c, c+1) pairs where sign(delta[i,c])
                         matches sign(delta[i,c+1])) / (M-1) - 0.5

    range: [-0.5, +0.5]. ~0 for random; ~+0.5 for fully correlated
    rank shifts (real structured emergent).

    sharp_threshold: minimum |coherence| to flag a sharp feature.
    min_components: minimum number of output components for this
        detector to apply; falls back to verdict='no_signal' if fewer.
    """
    p = np.asarray(parameter_values, dtype=float)
    M, raw = _build_output_matrix(p, output_fn)
    n_param, n_components = M.shape

    if n_components < min_components:
        return CoherenceResult(
            parameter_axis=p,
            coherence_trajectory=np.zeros(max(0, n_param - 1)),
            verdict="no_signal",
        )

    # Per-component rank across parameter axis
    ranks = _stable_rank(M, axis=0)  # (n_param, n_components)
    delta_ranks = np.diff(ranks, axis=0)  # (n_param-1, n_components)

    sign_a = np.sign(delta_ranks[:, :-1])
    sign_b = np.sign(delta_ranks[:, 1:])
    same_sign = ((sign_a * sign_b) > 0).astype(np.float32)
    coherence = same_sign.mean(axis=1) - 0.5  # (n_param-1,)

    sharp_features = []
    for i, c in enumerate(coherence):
        if abs(c) >= sharp_threshold:
            sharp_features.append({
                "between_indices": [int(i), int(i + 1)],
                "parameter_value": float(p[i + 1]),
                "coherence": float(c),
            })

    max_abs_idx = int(np.argmax(np.abs(coherence))) if len(coherence) > 0 else 0
    max_coh = float(coherence[max_abs_idx]) if len(coherence) > 0 else 0.0
    max_par = float(p[max_abs_idx + 1]) if len(coherence) > 0 else 0.0
    verdict = "novel_structure" if sharp_features else "smooth"

    return CoherenceResult(
        parameter_axis=p,
        coherence_trajectory=coherence,
        sharp_features=sharp_features,
        max_coherence=max_coh,
        max_coherence_parameter=max_par,
        verdict=verdict,
    )


# ============================================================
# Variant 2: Per-component consensus (z-score above baseline)
# ============================================================

@dataclass
class ConsensusResult:
    parameter_axis: np.ndarray
    consensus_trajectory: np.ndarray
    sharp_features: list = field(default_factory=list)
    max_consensus: float = 0.0
    max_consensus_parameter: float = 0.0
    verdict: str = "smooth"


def scan_novelty_consensus(
    parameter_values: np.ndarray,
    output_fn: Callable[[float], np.ndarray],
    z_threshold: float = 3.0,
    consensus_fraction: float = 0.30,
    baseline_quartiles: tuple = (0.25, 0.25),
) -> ConsensusResult:
    """Detect emergents when many components simultaneously exceed
    z_threshold above their per-component median baseline.

    baseline_quartiles: (front_q, back_q) -- fraction of parameter range
        from front and back to use as event-free baseline. Default
        (0.25, 0.25): outer halves are baseline, middle 50% is
        the search region.

    Consensus at parameter i = fraction of components with
        |z_i| > z_threshold. Verdict 'novel_structure' if any
        parameter has consensus >= consensus_fraction.
    """
    p = np.asarray(parameter_values, dtype=float)
    M, raw = _build_output_matrix(p, output_fn)
    n_param, n_components = M.shape

    if n_param < 4:
        return ConsensusResult(
            parameter_axis=p,
            consensus_trajectory=np.zeros(n_param),
            verdict="no_signal",
        )

    # Baseline: outer fractions of the parameter axis
    front_q, back_q = baseline_quartiles
    n_front = max(1, int(front_q * n_param))
    n_back = max(1, int(back_q * n_param))
    baseline = np.concatenate([M[:n_front], M[-n_back:]], axis=0)
    med = np.median(baseline, axis=0)
    mad = np.median(np.abs(baseline - med), axis=0) + 1e-12
    sigma = 1.4826 * mad

    z = (M - med) / sigma  # (n_param, n_components)
    consensus = (np.abs(z) > z_threshold).mean(axis=1)  # (n_param,)

    sharp_features = []
    for i, c in enumerate(consensus):
        if c >= consensus_fraction:
            sharp_features.append({
                "parameter_index": int(i),
                "parameter_value": float(p[i]),
                "consensus_fraction": float(c),
                "max_z": float(np.max(np.abs(z[i]))),
            })

    max_idx = int(np.argmax(consensus))
    max_c = float(consensus[max_idx])
    max_par = float(p[max_idx])
    verdict = "novel_structure" if sharp_features else "smooth"

    return ConsensusResult(
        parameter_axis=p,
        consensus_trajectory=consensus,
        sharp_features=sharp_features,
        max_consensus=max_c,
        max_consensus_parameter=max_par,
        verdict=verdict,
    )


# ============================================================
# Variant 3: Multi-run coincidence
# ============================================================

@dataclass
class MultirunResult:
    parameter_axis: np.ndarray
    coincidence_rate: np.ndarray
    sharp_features: list = field(default_factory=list)
    max_coincidence: float = 0.0
    max_coincidence_parameter: float = 0.0
    n_runs: int = 0
    verdict: str = "smooth"


def scan_novelty_multirun(
    parameter_values: np.ndarray,
    output_fn_list: Sequence[Callable[[float], np.ndarray]],
    per_run_z_threshold: float = 3.0,
    coincidence_fraction: float = 0.5,
    parameter_tolerance: int = 1,
) -> MultirunResult:
    """Detect emergents that appear at the SAME parameter location
    across most of N independent runs.

    For each run k:
      - run scan_novelty_consensus to get per-parameter z-scores
      - flag parameters where consensus > 0 (any component fires)
    For each parameter location:
      - count fraction of runs that fired
    Verdict 'novel_structure' if any parameter has coincidence rate
    >= coincidence_fraction.

    parameter_tolerance: how many adjacent-parameter slots count as
        "the same location" for coincidence purposes.
    """
    p = np.asarray(parameter_values, dtype=float)
    n_param = len(p)
    n_runs = len(output_fn_list)
    if n_runs < 2:
        return MultirunResult(
            parameter_axis=p,
            coincidence_rate=np.zeros(n_param),
            n_runs=n_runs,
            verdict="no_signal",
        )

    # Run consensus per-run with a NON-TRIVIAL threshold (10% of
    # components must fire simultaneously) so chance fluctuations in
    # individual components don't flag every parameter. With 16
    # components and z=3 threshold (P=0.27%), expected random fire
    # per parameter ≈ 4%; 10% consensus is well above the noise floor.
    fired = np.zeros((n_runs, n_param), dtype=bool)
    inner_consensus_fraction = 0.10
    for k, fn in enumerate(output_fn_list):
        result = scan_novelty_consensus(
            p, fn, z_threshold=per_run_z_threshold,
            consensus_fraction=inner_consensus_fraction,
        )
        for sf in result.sharp_features:
            i = sf["parameter_index"]
            lo = max(0, i - parameter_tolerance)
            hi = min(n_param, i + parameter_tolerance + 1)
            fired[k, lo:hi] = True

    coincidence_rate = fired.mean(axis=0)  # fraction of runs that fired
    sharp_features = []
    for i, c in enumerate(coincidence_rate):
        if c >= coincidence_fraction:
            sharp_features.append({
                "parameter_index": int(i),
                "parameter_value": float(p[i]),
                "coincidence_rate": float(c),
                "n_runs_fired": int(fired[:, i].sum()),
            })

    max_idx = int(np.argmax(coincidence_rate)) if n_param > 0 else 0
    max_c = float(coincidence_rate[max_idx]) if n_param > 0 else 0.0
    max_par = float(p[max_idx]) if n_param > 0 else 0.0
    verdict = "novel_structure" if sharp_features else "smooth"

    return MultirunResult(
        parameter_axis=p,
        coincidence_rate=coincidence_rate,
        sharp_features=sharp_features,
        max_coincidence=max_c,
        max_coincidence_parameter=max_par,
        n_runs=n_runs,
        verdict=verdict,
    )
