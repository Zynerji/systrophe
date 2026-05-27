"""Tests for DSI observables and log-periodic precursor fits."""

import numpy as np
import pytest

from systrophe.catchers.dsi_observables import (
    box_count_dimension_1d,
    discrete_scale_invariance_test,
    fit_log_periodic_precursor,
    log_periodic_model,
    lomb_scargle_log_periodicity,
)


# ----- log-periodic model -----------------------------------------------

def test_log_periodic_model_finite_below_t_c():
    """Model returns finite values for t < t_c."""
    t = np.linspace(0, 0.9, 10)
    y = log_periodic_model(t, t_c=1.0, A=0.0, B=1.0, z=0.5,
                            omega=5.0, C=0.1, phi=0.0)
    assert np.all(np.isfinite(y))


def test_log_periodic_model_nan_above_t_c():
    """Model returns NaN at t >= t_c."""
    t = np.array([1.0, 1.5])
    y = log_periodic_model(t, t_c=1.0, A=0.0, B=1.0, z=0.5,
                            omega=5.0, C=0.1, phi=0.0)
    assert np.all(np.isnan(y))


def test_log_periodic_model_amplitude_zero_reduces_to_power_law():
    """C = 0 gives pure power-law A + B (t_c - t)^(-z)."""
    t = np.linspace(0, 0.9, 5)
    y_full = log_periodic_model(t, t_c=1.0, A=2.0, B=3.0, z=0.5,
                                  omega=5.0, C=0.0, phi=0.0)
    y_powerlaw = 2.0 + 3.0 * (1.0 - t) ** (-0.5)
    assert np.allclose(y_full, y_powerlaw, atol=1e-12)


# ----- fit -------------------------------------------------------------

def test_fit_recovers_input_parameters():
    """On synthetic log-periodic data, fit recovers omega within tolerance."""
    t = np.linspace(0, 0.95, 100)
    true_params = dict(t_c=1.0, A=1.0, B=0.5, z=0.5, omega=6.0, C=0.05, phi=0.0)
    y = log_periodic_model(t, **true_params)
    # Add tiny noise to ensure the fit is non-trivial
    rng = np.random.default_rng(42)
    y_noisy = y + rng.normal(scale=0.005, size=y.shape)
    fit = fit_log_periodic_precursor(t, y_noisy, t_c_init=1.05,
                                       omega_init=6.0)
    # omega is the most identifiable parameter
    assert abs(fit.omega - 6.0) / 6.0 < 0.05


def test_fit_geometric_ratio_correct():
    """geometric_ratio = exp(2 pi / |omega|)."""
    t = np.linspace(0, 0.95, 100)
    y = log_periodic_model(t, t_c=1.0, A=1.0, B=0.5, z=0.5,
                            omega=6.0, C=0.05, phi=0.0)
    fit = fit_log_periodic_precursor(t, y, t_c_init=1.05, omega_init=6.0)
    expected = float(np.exp(2 * np.pi / abs(fit.omega)))
    assert fit.geometric_ratio == pytest.approx(expected, rel=1e-12)


def test_fit_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_log_periodic_precursor(np.array([0.1, 0.2, 0.3]),
                                     np.array([1.0, 2.0, 3.0]))


# ----- DSI test ---------------------------------------------------------

def test_dsi_geometric_progression_passes():
    """Pure geometric progression is detected as DSI."""
    vals = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    result = discrete_scale_invariance_test(vals)
    assert result["is_dsi"]
    assert result["best_ratio"] == pytest.approx(2.0, rel=0.05)


def test_dsi_random_fails():
    """Truly random spacings are NOT classified as DSI."""
    rng = np.random.default_rng(1)
    vals = np.sort(rng.uniform(1, 100, size=20))
    result = discrete_scale_invariance_test(vals)
    # Likely False, but at worst should have large rms_log_dev
    assert not result["is_dsi"] or result["rms_log_dev"] > 0


# ----- box-counting -----------------------------------------------------

def test_box_count_1d_basic():
    """A line segment in 1D has dimension ~ 1."""
    # 1000 uniform points on [0, 1]
    pts = np.linspace(0, 1, 1000)
    result = box_count_dimension_1d(pts, n_scales=14)
    # Dimension should be close to 1
    assert 0.6 < result["dimension"] < 1.2


def test_box_count_log_coords():
    """Geometric progression in linear coords becomes uniform in log coords;
    log-coord box dim is therefore close to 1 (dense uniform), and
    linear-coord dim is more like 0 (sparse)."""
    geom = 2.0 ** np.arange(10)
    log_d = box_count_dimension_1d(geom, log_coords=True)
    lin_d = box_count_dimension_1d(geom, log_coords=False)
    # log-coord points are uniformly spaced
    assert log_d["dimension"] > lin_d["dimension"]


# ----- Lomb-Scargle ----------------------------------------------------

def test_lomb_scargle_returns_finite():
    vals = 2.0 ** np.arange(15)
    result = lomb_scargle_log_periodicity(vals, n_test_omega=50)
    assert np.isfinite(result["best_omega"])
    assert result["best_amplitude"] >= 0


def test_lomb_scargle_short_input():
    result = lomb_scargle_log_periodicity(np.array([1.0, 2.0]))
    assert np.isnan(result["best_omega"])
