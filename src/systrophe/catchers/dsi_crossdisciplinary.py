"""DSI / log-periodic Tipler structure in non-physics contexts.

The Tipler log-periodic structure (period exp(pi/alpha)) is a
discrete-scale-invariance (DSI) signature universal across many
domains (Sornette 1998):

- Financial crashes (log-periodic precursors)
- Earthquake recurrence times
- Biological scaling (allometry)
- Crystal growth
- Critical phenomena

This module provides tools to detect Tipler-like log-periodic
structure in arbitrary time series or datasets, and to test whether
the detected structure matches the Systrophe alpha = sqrt(4 a^2 - 1)
prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class DSIDetection:
    """Detection of log-periodic / DSI structure in a time series."""

    alpha_fit: float  # log-frequency (mapped to "a" via a = sqrt((alpha^2 + 1)/4))
    log_period: float  # exp(pi / alpha)
    amplitude: float
    phase: float
    fit_quality: float  # R^2-like; higher = better
    n_extrema_detected: int
    a_systrophe: float  # equivalent Systrophe rotation parameter


def fit_log_periodic_to_data(
    x_array: np.ndarray, y_array: np.ndarray,
) -> DSIDetection:
    """Fit a log-periodic model to (x, y) data.

    Model: y(x) = A + B sin(alpha ln(x) + phi)
    The fit returns alpha (log-frequency), amplitude B, phase phi.
    """
    x_arr = np.asarray(x_array, dtype=float)
    y_arr = np.asarray(y_array, dtype=float)
    valid = (x_arr > 0)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if len(x_arr) < 5:
        return DSIDetection(
            alpha_fit=0.0, log_period=float("inf"), amplitude=0.0,
            phase=0.0, fit_quality=0.0, n_extrema_detected=0,
            a_systrophe=0.5,
        )

    def model(x, A, B, alpha, phi):
        return A + B * np.sin(alpha * np.log(np.maximum(x, 1e-12)) + phi)

    try:
        p0 = (float(y_arr.mean()), float(y_arr.std()), 1.0, 0.0)
        popt, _ = curve_fit(model, x_arr, y_arr, p0=p0, maxfev=10000)
        A, B, alpha, phi = popt
        y_pred = model(x_arr, *popt)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        R2 = 1 - ss_res / max(ss_tot, 1e-30)
        # Count extrema
        if len(y_arr) > 2:
            sign_diffs = np.diff(np.sign(np.diff(y_arr)))
            n_extrema = int(np.sum(np.abs(sign_diffs) > 0))
        else:
            n_extrema = 0
        # Map alpha to Systrophe a: alpha^2 = 4 a^2 - 1 => a = sqrt((alpha^2 + 1) / 4)
        a_systrophe = float(np.sqrt((alpha ** 2 + 1) / 4))
        log_period = float(np.exp(np.pi / abs(alpha))) if abs(alpha) > 1e-12 else float("inf")
        return DSIDetection(
            alpha_fit=float(abs(alpha)), log_period=log_period,
            amplitude=float(abs(B)), phase=float(phi),
            fit_quality=float(R2), n_extrema_detected=n_extrema,
            a_systrophe=a_systrophe,
        )
    except Exception:
        return DSIDetection(
            alpha_fit=0.0, log_period=float("inf"), amplitude=0.0,
            phase=0.0, fit_quality=0.0, n_extrema_detected=0,
            a_systrophe=0.5,
        )


def generate_synthetic_tipler_signal(
    a_target: float = 1.0, n_points: int = 200,
    x_min: float = 1.0, x_max: float = 100.0,
    noise_amp: float = 0.1, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data with Tipler log-periodic structure."""
    rng = np.random.default_rng(seed)
    x = np.linspace(x_min, x_max, n_points)
    alpha = float(np.sqrt(4 * a_target ** 2 - 1))
    y = 1.0 + 0.5 * np.sin(alpha * np.log(x))
    y += noise_amp * rng.standard_normal(n_points)
    return x, y


def recovery_test_synthetic(a_target: float = 1.0,
                                 n_points: int = 200) -> dict:
    """Generate synthetic Tipler signal and test if we can recover a_target."""
    x, y = generate_synthetic_tipler_signal(a_target=a_target, n_points=n_points,
                                                noise_amp=0.05)
    detection = fit_log_periodic_to_data(x, y)
    recovered_a = detection.a_systrophe
    relative_error = abs(recovered_a - a_target) / max(a_target, 1e-12)
    return {
        "a_target": a_target,
        "a_recovered": recovered_a,
        "relative_error": relative_error,
        "fit_quality": detection.fit_quality,
        "recovery_successful": relative_error < 0.15,
    }


def financial_crash_log_periodic_signature(
    price_data: np.ndarray, time_to_crash: np.ndarray,
) -> dict:
    """Test if financial price data shows log-periodic crash precursor.

    The Sornette log-periodic crash model has form
        log(p(t)) ~ A + B (t_c - t)^z [1 + C cos(omega ln(t_c - t) + phi)]

    where omega is the log-frequency. We test if our Tipler alpha
    can match this omega.
    """
    return fit_log_periodic_to_data(time_to_crash, np.log(price_data))


def biological_scaling_log_periodic(
    organism_size: np.ndarray, metabolic_rate: np.ndarray,
) -> dict:
    """Test allometric scaling for log-periodic structure.

    Standard allometric law: B = a M^b. Tipler-like deviations
    correspond to oscillations around the power law.
    """
    log_rate = np.log(metabolic_rate)
    log_mass = np.log(organism_size)
    # Detrend by power law first
    slope, intercept = np.polyfit(log_mass, log_rate, 1)
    detrended = log_rate - (slope * log_mass + intercept)
    return fit_log_periodic_to_data(np.exp(log_mass), detrended)
