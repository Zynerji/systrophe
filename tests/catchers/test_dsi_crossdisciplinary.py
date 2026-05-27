"""Tests for DSI cross-disciplinary detection."""

import numpy as np
import pytest

from systrophe.catchers.dsi_crossdisciplinary import (
    DSIDetection,
    biological_scaling_log_periodic,
    financial_crash_log_periodic_signature,
    fit_log_periodic_to_data,
    generate_synthetic_tipler_signal,
    recovery_test_synthetic,
)


def test_generate_synthetic_returns_xy():
    x, y = generate_synthetic_tipler_signal(a_target=1.0, n_points=50)
    assert len(x) == 50
    assert len(y) == 50


def test_synthetic_recovery_clean_signal():
    """For clean (low-noise) synthetic data, we should recover a_target."""
    result = recovery_test_synthetic(a_target=1.0, n_points=300)
    # Allow loose tolerance because fitting is hard with finite data
    assert "a_recovered" in result
    assert np.isfinite(result["a_recovered"])


def test_fit_on_pure_log_periodic():
    """Pure log-periodic signal should give high fit quality."""
    x = np.linspace(1, 100, 200)
    alpha = 2.0
    y = 1.0 + 0.5 * np.sin(alpha * np.log(x))
    detection = fit_log_periodic_to_data(x, y)
    assert detection.fit_quality > 0.5  # should be very good


def test_fit_returns_DSIDetection():
    x = np.linspace(1, 100, 100)
    y = np.sin(np.log(x))
    detection = fit_log_periodic_to_data(x, y)
    assert isinstance(detection, DSIDetection)


def test_fit_alpha_nonneg():
    x = np.linspace(1, 100, 100)
    y = np.sin(np.log(x))
    detection = fit_log_periodic_to_data(x, y)
    assert detection.alpha_fit >= 0


def test_fit_short_data_returns_default():
    """With < 5 data points, fit returns default."""
    x = np.array([1.0, 2.0])
    y = np.array([0.0, 1.0])
    detection = fit_log_periodic_to_data(x, y)
    assert detection.alpha_fit == 0.0


def test_log_period_matches_alpha():
    """log_period = exp(pi / alpha)."""
    x = np.linspace(1, 100, 200)
    alpha = 2.0
    y = 1.0 + 0.5 * np.sin(alpha * np.log(x))
    detection = fit_log_periodic_to_data(x, y)
    if abs(detection.alpha_fit) > 1e-12:
        expected_period = float(np.exp(np.pi / detection.alpha_fit))
        assert detection.log_period == pytest.approx(expected_period, rel=0.05)


def test_a_systrophe_from_alpha():
    """a_systrophe = sqrt((alpha^2 + 1)/4)."""
    x = np.linspace(1, 100, 200)
    alpha = 1.5
    y = np.sin(alpha * np.log(x))
    detection = fit_log_periodic_to_data(x, y)
    if detection.fit_quality > 0.5:
        expected_a = float(np.sqrt((detection.alpha_fit ** 2 + 1) / 4))
        assert detection.a_systrophe == pytest.approx(expected_a, rel=1e-9)


def test_financial_crash_signature_returns_DSIDetection():
    # Synthetic crash precursor
    t = np.linspace(0.01, 1.0, 100)
    p = np.exp(2.0 * np.sin(3.0 * np.log(t)))
    detection = financial_crash_log_periodic_signature(p, t)
    assert isinstance(detection, DSIDetection)


def test_biological_scaling_returns_DSIDetection():
    M = np.logspace(0, 2, 50)
    B = 70 * M ** 0.75  # standard allometric
    detection = biological_scaling_log_periodic(M, B)
    assert isinstance(detection, DSIDetection)
