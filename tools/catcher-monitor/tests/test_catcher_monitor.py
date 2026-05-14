"""Tests for catcher_monitor."""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from catcher_monitor import (
    AnomalyResult,
    Emergent,
    PhaseTransitionResult,
    TrainingMonitor,
    find_anomalies,
    find_phase_transition,
    scan_emergents,
)


# ---------------------------------------------------------------------------
# find_phase_transition
# ---------------------------------------------------------------------------


def test_step_function_transition_localized():
    """Step function = the catcher's natural target. Should localise within ~1 grid step."""
    centre_truth = 2.5
    params = np.linspace(0.0, 5.0, 60)
    def fn(x: float) -> float:
        return 0.0 if x < centre_truth else 1.0
    res = find_phase_transition(params, fn)
    assert isinstance(res, PhaseTransitionResult)
    assert res.kind in {"discontinuous", "smooth_sigmoid"}
    assert res.transition_at is not None
    # 60-point [0,5] grid -> grid step ~ 0.085. Two grid steps tolerance.
    assert abs(res.transition_at - centre_truth) < 0.2, (
        f"transition_at={res.transition_at}, truth={centre_truth}, "
        f"kind={res.kind}"
    )


def test_quantised_sigmoid_localized():
    """Sharp quantised sigmoid also lands close to truth (catcher likes
    discretisation)."""
    centre_truth = 2.5
    params = np.linspace(0.0, 5.0, 60)
    def fn(x: float) -> float:
        return round(1.0 / (1.0 + math.exp(-50.0 * (x - centre_truth))), 2)
    res = find_phase_transition(params, fn)
    assert res.transition_at is not None
    assert abs(res.transition_at - centre_truth) < 0.3


def test_smooth_analytic_sigmoid_returns_none_by_design():
    """A smooth analytic sigmoid (no discretisation, slope <= ~15) gets
    'none'. This is the documented Systrophe limitation (Dianoia FINDINGS):
    rank-thermometer addresses on a smooth signal don't produce
    Hamming-step outliers."""
    params = np.linspace(0.0, 5.0, 100)
    def fn(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-8.0 * (x - 2.5)))  # gentle slope
    res = find_phase_transition(params, fn)
    assert res.kind == "none", (
        f"Got {res.kind!r}; the smooth-analytic-sigmoid limitation should "
        f"return 'none' for slope <= ~15."
    )


def test_smooth_constant_signal_returns_none():
    """No transition in a constant signal -> kind='none'."""
    params = np.linspace(0.0, 1.0, 40)
    def fn(x: float) -> float:
        return 1.7   # constant
    res = find_phase_transition(params, fn)
    assert res.kind == "none"
    assert res.transition_at is None


def test_phase_transition_with_array_measurement():
    """Measurement_fn returning a vector is collapsed to its L2 norm."""
    params = np.linspace(0.0, 4.0, 50)
    def fn(x: float) -> np.ndarray:
        # 5-D vector whose norm jumps at x = 2.0
        base = np.array([0.0, 1.0, 0.5, 0.2, 0.1])
        scale = 0.1 if x < 2.0 else 5.0
        return base * scale
    res = find_phase_transition(params, fn)
    assert res.transition_at is not None
    assert abs(res.transition_at - 2.0) < 0.3


# ---------------------------------------------------------------------------
# find_anomalies
# ---------------------------------------------------------------------------


def test_anomaly_detection_finds_outlier():
    """Construct 30 normal points + 1 outlier; the outlier should rank top-1."""
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(30, 8))
    outlier = np.full((1, 8), 20.0)
    samples = np.vstack([normal, outlier])
    res = find_anomalies(samples, top_k=3)
    assert isinstance(res, AnomalyResult)
    assert res.n_samples == 31
    # The outlier index is 30 (last row); it should be in the top-3 anomalies.
    assert 30 in res.anomaly_indices.tolist()


def test_anomaly_quantile_threshold():
    """With quantile-based thresholding, ~5% of points should be flagged on
    a uniform background of 100 normal samples."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(100, 4))
    res = find_anomalies(samples, threshold_quantile=0.95)
    assert 1 <= len(res.anomaly_indices) <= 15  # ~5% with noise


def test_anomaly_requires_two_samples():
    with pytest.raises(ValueError):
        find_anomalies(np.zeros((1, 4)))


# ---------------------------------------------------------------------------
# scan_emergents
# ---------------------------------------------------------------------------


def test_scan_emergents_returns_dataclasses():
    """The flat dataclass shape is preserved through the wrapper."""
    params = np.linspace(0, 1, 20)
    def fn(p: float) -> np.ndarray:
        return np.array([math.sin(10 * p), math.cos(10 * p)])
    res = scan_emergents(params, fn)
    assert res.verdict in {"novel_structure", "smooth", "uniform"}
    for e in res.emergents:
        assert isinstance(e, Emergent)
        assert isinstance(e.parameter_value, float)


# ---------------------------------------------------------------------------
# TrainingMonitor
# ---------------------------------------------------------------------------


def test_training_monitor_quiet_on_steady_loss():
    """Smooth decreasing loss should not trigger any anomaly."""
    mon = TrainingMonitor(window=80, refresh_every=10, min_samples=32)
    losses = np.linspace(2.5, 0.5, 200) + 0.01 * np.random.default_rng(0).normal(size=200)
    n_alerts = 0
    for step, v in enumerate(losses):
        ev = mon.update(float(v), step=step)
        if ev.is_anomaly:
            n_alerts += 1
    # Smooth decrease shouldn't usually trigger many alerts. We allow up to 1
    # because the catcher's noise-rejection isn't perfect.
    assert n_alerts <= 1, f"got {n_alerts} alerts on a steady decreasing loss"


def test_training_monitor_alerts_on_sudden_jump():
    """A sudden jump in loss should produce an anomaly event."""
    mon = TrainingMonitor(window=80, refresh_every=5, min_samples=32, cooldown=0)
    rng = np.random.default_rng(0)
    losses = []
    for step in range(150):
        # Steady around 1.0 for first 100 steps, then jump to 5.0
        loss = (1.0 if step < 100 else 5.0) + 0.05 * rng.normal()
        losses.append(loss)
    saw_alert = False
    for step, v in enumerate(losses):
        ev = mon.update(float(v), step=step)
        if ev.is_anomaly and step > 90:
            saw_alert = True
            # Detected transition should be reasonably near step 100
            if ev.transition_at_step is not None:
                assert 80 <= ev.transition_at_step <= 130, (
                    f"transition at {ev.transition_at_step}, expected ~100"
                )
            break
    assert saw_alert, "TrainingMonitor failed to flag the 1->5 loss jump"


def test_training_monitor_cooldown_suppresses_repeated_alerts():
    """During cooldown, further alerts are suppressed."""
    mon = TrainingMonitor(window=60, refresh_every=2, min_samples=20, cooldown=50)
    rng = np.random.default_rng(0)
    for step in range(40):
        mon.update(1.0 + 0.05 * rng.normal(), step=step)
    # Force an anomaly via a sharp jump
    n_alerts = 0
    for step in range(40, 130):
        v = 5.0 + 0.05 * rng.normal()
        ev = mon.update(v, step=step)
        if ev.is_anomaly:
            n_alerts += 1
    # With a cooldown of 50, we should see at most ~2 alerts in 90 steps.
    assert n_alerts <= 3, f"cooldown not working: {n_alerts} alerts"


def test_training_monitor_reset():
    mon = TrainingMonitor(window=20, refresh_every=5)
    for step in range(15):
        mon.update(float(step), step=step)
    mon.reset()
    assert mon.last_alert is None
    assert mon._update_count == 0
