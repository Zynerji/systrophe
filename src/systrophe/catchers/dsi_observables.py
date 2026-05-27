"""Discrete-scale-invariance (DSI) observables and log-periodic precursor fits.

Cross-disciplinary toolkit for fitting log-periodic precursors to
time-series and computing box-counting / DSI diagnostics on real-world
datasets.

Built on the mathematical foundation of `tipler_fractal.py` (which
established that Systrophe's Tipler-sinusoid is a clean DSI signature
with geometric ratio rho = exp(pi / alpha)). The same machinery
applies to:

- Financial crash precursors (Johansen-Sornette log-periodic model)
- Critical phenomena and renormalisation-group flows
- Earthquake / rock-fracture data
- Turbulence energy cascades
- Self-organised criticality time series

Sornette's standard log-periodic precursor model is

    f(t) = A + B (t_c - t)^(-z) [1 + C cos(omega ln(t_c - t) + phi)]

where t_c is the critical time, z is the power-law exponent, omega
is the log-periodic angular frequency, and C is the oscillatory
amplitude. The geometric ratio of consecutive log-periodic peaks is

    lambda = exp(2 pi / omega).

For Systrophe's Tipler sinusoid, alpha = sqrt(4 a^2 - 1) plays the
role of omega; the analog is exact.

References
----------
- D. Sornette, "Discrete-scale invariance and complex dimensions",
  Phys. Rep. 297 (1998) 239.
- A. Johansen, D. Sornette, "Log-periodic power law bubbles in
  Latin-American and Asian stock markets", Quant. Fin. 1 (2001) 533.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar


def log_periodic_model(
    t: np.ndarray,
    t_c: float,
    A: float,
    B: float,
    z: float,
    omega: float,
    C: float,
    phi: float,
) -> np.ndarray:
    """Sornette log-periodic precursor model.

    f(t) = A + B (t_c - t)^(-z) [1 + C cos(omega ln(t_c - t) + phi)]

    Defined for t < t_c only. Returns NaN outside that range.
    """
    t = np.asarray(t, dtype=float)
    dt = t_c - t
    out = np.full_like(t, float("nan"))
    valid = dt > 0
    if np.any(valid):
        x = dt[valid]
        out[valid] = A + B * x ** (-z) * (1 + C * np.cos(omega * np.log(x) + phi))
    return out


@dataclass(frozen=True)
class LogPeriodicFit:
    """Result of fitting `log_periodic_model` to data."""

    t_c: float
    A: float
    B: float
    z: float
    omega: float
    C: float
    phi: float
    geometric_ratio: float  # exp(2 pi / omega)
    rss: float  # residual sum of squares


def fit_log_periodic_precursor(
    t: np.ndarray,
    y: np.ndarray,
    t_c_init: float | None = None,
    z_init: float = 0.5,
    omega_init: float = 5.0,
) -> LogPeriodicFit:
    """Fit the Sornette log-periodic precursor model to data.

    Heuristic initial guess: t_c slightly past the last observation,
    z = 0.5, omega = 5 (the typical financial-crash value).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 7:
        raise ValueError("need at least 7 data points to fit 7 parameters")
    if t_c_init is None:
        t_c_init = float(t.max()) + 0.1 * (t.max() - t.min())

    def model(t_eval, t_c, A, B, z, omega, C, phi):
        return log_periodic_model(t_eval, t_c, A, B, z, omega, C, phi)

    # Reasonable parameter bounds
    p0 = [t_c_init, float(y.mean()), float(np.std(y)) * 0.1,
          z_init, omega_init, 0.1, 0.0]
    try:
        popt, _ = curve_fit(model, t, y, p0=p0, maxfev=20000)
    except Exception:
        # Refuse rather than return garbage
        raise
    t_c, A, B, z, omega, C, phi = popt
    residuals = y - model(t, *popt)
    rss = float(np.sum(residuals ** 2))
    geo_ratio = float(np.exp(2 * np.pi / abs(omega)))
    return LogPeriodicFit(
        t_c=float(t_c), A=float(A), B=float(B), z=float(z),
        omega=float(omega), C=float(C), phi=float(phi),
        geometric_ratio=geo_ratio, rss=rss,
    )


def discrete_scale_invariance_test(
    values: np.ndarray,
    candidate_ratios: np.ndarray | None = None,
    n_log_bins: int = 50,
) -> dict:
    """Test whether `values` (sorted positive reals) is DSI.

    Tests whether the log-spacings cluster around a single ratio.
    Returns:
      - best_ratio   : the candidate ratio with the cleanest log-grid fit
      - rms_log_dev  : RMS deviation of log(values) from the best-fit log-grid
      - is_dsi       : True iff rms_log_dev < 0.05 * ln(best_ratio)

    If `candidate_ratios` is None, scans ratios in [1.05, 10].
    """
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    if len(values) < 3:
        return {"best_ratio": float("nan"), "rms_log_dev": float("nan"),
                "is_dsi": False}
    log_v = np.log(np.sort(values))

    def deviation(log_lam: float) -> float:
        # Fit log_v[k] ~ log_v[0] + k * log_lam by least-squares;
        # report RMS residual.
        n = len(log_v)
        ks = np.arange(n)
        c1, c0 = np.polyfit(ks, log_v, 1)
        # Now check whether c1 is close to log_lam
        return float(abs(c1 - log_lam))

    if candidate_ratios is None:
        candidate_ratios = np.exp(np.linspace(np.log(1.05), np.log(10), 200))
    devs = np.array([deviation(np.log(lam)) for lam in candidate_ratios])
    idx_best = int(np.argmin(devs))
    best_ratio = float(candidate_ratios[idx_best])
    # RMS deviation of actual log spacings from best-fit
    n = len(log_v)
    ks = np.arange(n)
    c1, c0 = np.polyfit(ks, log_v, 1)
    predicted = c0 + c1 * ks
    rms = float(np.sqrt(np.mean((log_v - predicted) ** 2)))
    is_dsi = rms < 0.05 * np.log(best_ratio)
    return {
        "best_ratio": best_ratio,
        "rms_log_dev": rms,
        "is_dsi": is_dsi,
        "log_slope": float(c1),
    }


def box_count_dimension_1d(
    points: np.ndarray,
    n_scales: int = 20,
    log_coords: bool = False,
) -> dict:
    """Box-counting dimension of a 1D point set.

    If `log_coords` is True, the points are mapped to log values first
    (useful for testing whether a *geometric* progression has a
    fractal structure on the *log* coordinate, i.e. is genuinely DSI).
    """
    if log_coords:
        points = np.log(np.asarray(points, dtype=float))
    else:
        points = np.asarray(points, dtype=float)
    if len(points) < 4:
        return {"dimension": 0.0, "scales": np.array([]), "counts": np.array([])}
    lo, hi = float(np.min(points)), float(np.max(points))
    total = hi - lo
    if total <= 0:
        return {"dimension": 0.0, "scales": np.array([]), "counts": np.array([])}
    scales = []
    counts = []
    for k in range(1, n_scales + 1):
        eps = total / (2 ** k)
        idx = np.floor((points - lo) / eps).astype(int)
        n_occ = len(np.unique(idx))
        scales.append(eps)
        counts.append(n_occ)
    scales = np.array(scales)
    counts = np.array(counts)
    n_pts = len(points)
    valid = counts < n_pts
    if valid.sum() < 3:
        return {"dimension": 0.0, "scales": scales, "counts": counts}
    log_eps = np.log(scales[valid])
    log_N = np.log(counts[valid])
    slope, _ = np.polyfit(log_eps, log_N, 1)
    return {"dimension": float(-slope), "scales": scales, "counts": counts}


def lomb_scargle_log_periodicity(
    values: np.ndarray,
    log_omega_range: tuple[float, float] = (np.log(0.5), np.log(50)),
    n_test_omega: int = 200,
) -> dict:
    """Lomb-Scargle-like search for the best log-periodic angular frequency.

    For a sorted positive sequence `values`, compute log(values) and
    test which omega in the given range produces the cleanest sinusoidal
    pattern, modelled as

        log(value_k) ~ alpha * k + beta + gamma cos(omega * k + delta).

    Returns the best omega and the corresponding amplitude C.
    """
    values = np.asarray(values, dtype=float)
    values = np.sort(values[values > 0])
    if len(values) < 5:
        return {"best_omega": float("nan"), "best_amplitude": 0.0}
    log_v = np.log(values)
    ks = np.arange(len(log_v))
    # De-trend
    p1, p0 = np.polyfit(ks, log_v, 1)
    res = log_v - (p0 + p1 * ks)
    log_omega_min, log_omega_max = log_omega_range
    omegas = np.exp(np.linspace(log_omega_min, log_omega_max, n_test_omega))
    best_amp = 0.0
    best_om = float("nan")
    for om in omegas:
        # Fit gamma cos(om * k + delta)
        cs = np.cos(om * ks)
        ss = np.sin(om * ks)
        c_coef = float(np.dot(res, cs) / np.dot(cs, cs))
        s_coef = float(np.dot(res, ss) / np.dot(ss, ss))
        amp = float(np.sqrt(c_coef ** 2 + s_coef ** 2))
        if amp > best_amp:
            best_amp = amp
            best_om = float(om)
    return {"best_omega": best_om, "best_amplitude": best_amp,
            "geometric_ratio": float(np.exp(2 * np.pi / best_om)) if best_om > 0 else float("nan")}
