"""Phase-transition, anomaly, and emergent detection on top of systrophe.novelty_catcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from systrophe.novelty_catcher import (
    hamming_distance,
    real_array_to_address,
    scan_novelty,
)
from systrophe.derivative_catcher import catch_smooth_transition


# ---------------------------------------------------------------------------
# Phase-transition detection (parametric)
# ---------------------------------------------------------------------------


@dataclass
class PhaseTransitionResult:
    """Outcome of `find_phase_transition`."""
    kind: str                           # "discontinuous" | "smooth_sigmoid" | "none"
    transition_at: float | None         # parameter value of the detected transition, or None
    confidence: float                   # 0.0 -> 1.0, based on lambda_2 sharp-feature excess
    n_sharp_features_value: int
    n_sharp_features_derivative: int
    value_verdict: str                  # raw verdict from value-level scan
    derivative_verdict: str             # raw verdict from derivative-level scan
    parameter_values: np.ndarray        # for plotting

    def __repr__(self) -> str:
        if self.transition_at is None:
            return f"PhaseTransitionResult(kind=none)"
        return (
            f"PhaseTransitionResult(kind={self.kind!r}, "
            f"transition_at={self.transition_at:.6g}, "
            f"confidence={self.confidence:.3f})"
        )


def find_phase_transition(
    parameter_values: Sequence[float],
    measurement_fn: Callable[[float], float | np.ndarray],
    n_bits: int = 32,
) -> PhaseTransitionResult:
    """Detect a phase transition in `measurement_fn` over `parameter_values`.

    Two-pass detection on top of `systrophe.derivative_catcher.catch_smooth_transition`:
      * **Value-level scan** -- catches discontinuous jumps in the
        measurement output (the classical Hamming-step outlier).
      * **Derivative-level scan** -- catches smooth-sigmoid centres
        where the value-level scan returns "smooth".

    Parameters
    ----------
    parameter_values:
        1-D sequence of parameter samples (monotonic order assumed).
    measurement_fn:
        Callable `param -> measurement`. Measurement may be a scalar or
        a 1-D array; in the latter case the L2 norm is used so the
        derivative-level scan can operate on a scalar series.
    n_bits:
        Address width for the catcher.

    Returns
    -------
    PhaseTransitionResult.

    Examples
    --------
    >>> import numpy as np
    >>> from catcher_monitor import find_phase_transition
    >>> # synthetic sigmoid centred at param=2.5
    >>> p = np.linspace(0.0, 5.0, 50)
    >>> r = find_phase_transition(p, lambda x: 1.0 / (1.0 + np.exp(-10 * (x - 2.5))))
    >>> abs(r.transition_at - 2.5) < 0.5
    True
    """
    p_arr = np.asarray(parameter_values, dtype=float)

    # Coerce measurement to a scalar fn for the derivative scan.
    def scalar_fn(param: float) -> float:
        out = measurement_fn(float(param))
        if isinstance(out, (int, float, np.floating)):
            return float(out)
        return float(np.linalg.norm(np.asarray(out, dtype=float).ravel()))

    res = catch_smooth_transition(p_arr, scalar_fn, n_bits=n_bits)
    value_scan = res["value_scan"]
    deriv_scan = res["derivative_scan"]

    # Confidence: scaled "excess over median" of the strongest sharp feature.
    # Caps at 1.0 when the Hamming step is the maximum possible.
    confidence = 0.0
    if res["kind"] == "discontinuous" and value_scan.sharp_features:
        best = max(value_scan.sharp_features,
                   key=lambda s: s.get("excess_over_median", 0.0))
        confidence = min(1.0, float(best.get("excess_over_median", 0.0)) / float(n_bits))
    elif res["kind"] == "smooth_sigmoid" and deriv_scan.sharp_features:
        best = max(deriv_scan.sharp_features,
                   key=lambda s: s.get("excess_over_median", 0.0))
        confidence = min(1.0, float(best.get("excess_over_median", 0.0)) / float(n_bits))

    return PhaseTransitionResult(
        kind=res["kind"],
        transition_at=res["estimated_transition_centre"],
        confidence=confidence,
        n_sharp_features_value=len(value_scan.sharp_features),
        n_sharp_features_derivative=len(deriv_scan.sharp_features),
        value_verdict=value_scan.verdict,
        derivative_verdict=deriv_scan.verdict,
        parameter_values=p_arr,
    )


# ---------------------------------------------------------------------------
# Sample-set anomaly detection (non-parametric)
# ---------------------------------------------------------------------------


@dataclass
class AnomalyResult:
    """Outcome of `find_anomalies`."""
    scores: np.ndarray                  # one anomaly score per sample (higher = more anomalous)
    anomaly_indices: np.ndarray         # indices of top_k most anomalous
    threshold: float                    # score threshold used
    n_samples: int

    def __repr__(self) -> str:
        return (
            f"AnomalyResult(n_samples={self.n_samples}, "
            f"n_anomalies={len(self.anomaly_indices)}, "
            f"threshold={self.threshold:.3f})"
        )


def find_anomalies(
    samples: Sequence[np.ndarray] | np.ndarray,
    n_bits: int = 32,
    radius: int | None = None,
    top_k: int | None = None,
    threshold_quantile: float = 0.95,
) -> AnomalyResult:
    """Flag samples that are isolated in address-space.

    Each sample is hashed to an `n_bits` bit address via
    `systrophe.novelty_catcher.real_array_to_address`. A Hamming graph
    is built over the addresses with edge threshold `radius`. The
    anomaly score for sample i is
        score_i = mean Hamming distance from i to every other sample.
    Outliers in address-space have high scores.

    Parameters
    ----------
    samples:
        Either a list of 1-D arrays or a 2-D array of shape
        `(n_samples, n_features)`.
    n_bits:
        Address width.
    radius:
        Edge threshold for the Hamming graph (unused for the mean-
        distance score; kept for API parity / future degree-based
        scoring).
    top_k:
        Return the top-k most anomalous indices. If None, use
        `threshold_quantile`.
    threshold_quantile:
        Fraction of the score distribution to treat as the
        anomaly threshold (default 0.95 -> top 5% flagged).
    """
    arr = (
        np.asarray(samples, dtype=float)
        if isinstance(samples, np.ndarray) and samples.ndim == 2
        else [np.asarray(s, dtype=float).ravel() for s in samples]
    )
    if isinstance(arr, np.ndarray):
        sample_list = [arr[i] for i in range(arr.shape[0])]
    else:
        sample_list = arr
    n = len(sample_list)
    if n < 2:
        raise ValueError("find_anomalies needs at least 2 samples")

    # Use *global* v_min / v_max across the whole sample set so an
    # "all values are large" outlier hashes to a different bit pattern
    # than an "all values are small" normal sample.
    # `real_array_to_address`'s default per-sample normalization would
    # make those two indistinguishable.
    all_finite = np.concatenate([
        s[np.isfinite(s)] for s in sample_list if len(s) > 0
    ])
    if len(all_finite) == 0:
        v_min, v_max = 0.0, 1.0
    else:
        v_min = float(np.min(all_finite))
        v_max = float(np.max(all_finite))
        if v_max <= v_min:
            v_max = v_min + 1.0

    addresses = [
        real_array_to_address(s, n_bits=n_bits, v_min=v_min, v_max=v_max)
        for s in sample_list
    ]
    # Mean Hamming distance to every other sample.
    scores = np.zeros(n, dtype=float)
    for i in range(n):
        d_sum = 0.0
        for j in range(n):
            if i == j:
                continue
            d_sum += hamming_distance(addresses[i], addresses[j])
        scores[i] = d_sum / float(n - 1)

    if top_k is not None:
        threshold = float(np.partition(scores, -int(top_k))[-int(top_k)])
        anomaly_idx = np.argsort(scores)[-int(top_k):][::-1]
    else:
        threshold = float(np.quantile(scores, threshold_quantile))
        anomaly_idx = np.where(scores >= threshold)[0]
        # Sort by score descending
        anomaly_idx = anomaly_idx[np.argsort(-scores[anomaly_idx])]

    return AnomalyResult(
        scores=scores,
        anomaly_indices=anomaly_idx,
        threshold=threshold,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Generic "scan for emergents" pass-through
# ---------------------------------------------------------------------------


@dataclass
class Emergent:
    """Single sharp-feature event from a `scan_emergents` call."""
    parameter_value: float
    hamming_step: int
    median_step: float
    excess_over_median: float
    between_indices: tuple[int, int]


@dataclass
class EmergentScanResult:
    """Result of `scan_emergents` -- mirrors the load-bearing fields of
    Systrophe's NoveltyScanResult in a flat dataclass.
    """
    verdict: str                        # "novel_structure" | "smooth" | "uniform"
    emergents: list[Emergent]
    parameter_axis: np.ndarray
    lambda_2_at_radius: dict[int, float]


def scan_emergents(
    parameter_values: Sequence[float],
    measurement_fn: Callable[[float], np.ndarray | float],
    n_bits: int = 32,
    radii: tuple[int, ...] = (4, 8, 12, 16),
    parameter_label: str = "parameter",
) -> EmergentScanResult:
    """Wrap `systrophe.novelty_catcher.scan_novelty` and return a flat dataclass.

    This is the direct interface for users who want the full sharp-feature
    list rather than the single transition centre from
    `find_phase_transition`.
    """
    p_arr = np.asarray(parameter_values, dtype=float)

    def fn_(param: float):
        out = measurement_fn(float(param))
        if isinstance(out, (int, float, np.floating)):
            return np.array([float(out)])
        return np.asarray(out, dtype=float).ravel()

    result = scan_novelty(
        p_arr, fn_, n_bits=n_bits, radii=radii,
        parameter_label=parameter_label,
    )
    emergents = [
        Emergent(
            parameter_value=float(s["parameter_value"]),
            hamming_step=int(s["hamming_step"]),
            median_step=float(s["median_step"]),
            excess_over_median=float(s["excess_over_median"]),
            between_indices=tuple(s["between_indices"]),
        )
        for s in result.sharp_features
    ]
    return EmergentScanResult(
        verdict=result.verdict,
        emergents=emergents,
        parameter_axis=result.parameter_axis,
        lambda_2_at_radius={int(k): float(v) for k, v in result.lambda_2_at_radius.items()},
    )
