"""Chaotic time machine: feed a Lorenz-attractor rotation a(t) into the CTC harness.

Having induced a Lorenz-class rotation a(t) in the dissipative dust sector
(rotating_dust_lorenz.py), we let the spacetime inherit it *adiabatically*: at
each instant the exterior is the static Bonnor Case III solution for the current
a(t). The supercritical CTC bands are the negative bands of g_phiphi(r); as a(t)
wanders on the attractor, the log-frequency alpha = sqrt(4 a^2 - 1) breathes and
the CTC bands shift along r. Inside a fixed observation window [r_min, r_max] the
total CTC log-measure becomes a chaotic time series that inherits the attractor's
statistics -- a "flickering" time machine.

Adiabatic validity: this quasi-static treatment holds when the Lorenz timescale
(~ 1 / lambda_max, with lambda_max ~ 0.9 in Lorenz time units) is long compared to
the light-crossing / orbital time of the band. We report the ratio so the
approximation is auditable rather than assumed.
"""

from __future__ import annotations

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def ctc_log_measure_for_a(
    a: float, R: float, r_min: float, r_max: float, n_grid: int = 4001
) -> dict:
    """CTC band log-measure of a single supercritical cylinder a = omega R.

    Returns total log-measure of {L < 0} in [r_min, r_max], the band count,
    and the Tipler log-frequency alpha. Sub-critical a (<= 1/2) returns zeros.
    """
    if a <= 0.5:
        return {"log_measure": 0.0, "n_bands": 0, "alpha": 0.0}
    vs = VanStockumInterior(omega=a / R, R=R)
    rs = np.linspace(r_min, r_max, n_grid)
    L = np.asarray(vs.analytic_exterior_L(rs), dtype=float)
    u = np.log(rs)
    neg = L < 0.0
    # total log-measure of negative regions (trapezoidal on the boolean mask)
    log_measure = float(np.trapezoid(neg.astype(float), u))
    # band count = number of contiguous negative runs
    edges = np.diff(neg.astype(int))
    n_bands = int(np.sum(edges == 1) + (1 if neg[0] else 0))
    return {"log_measure": log_measure, "n_bands": n_bands, "alpha": float(vs.alpha)}


def chaotic_ctc_timeseries(
    rotation: dict, R: float = 1.0, r_min: float = 1.05, r_max: float = 20.0,
    stride: int = 1,
) -> dict:
    """Map a rotation a(t) time series to a CTC log-measure time series.

    Parameters
    ----------
    rotation : dict with 't' and 'a' (output of rotation_parameter_timeseries)
    R, r_min, r_max : observation geometry
    stride : subsample factor over the rotation time axis (for speed)

    Returns dict with 't', 'a', 'log_measure', 'n_bands', 'alpha'.
    """
    t = rotation["t"][::stride]
    a = rotation["a"][::stride]
    lm = np.empty_like(a)
    nb = np.empty_like(a)
    al = np.empty_like(a)
    for i, ai in enumerate(a):
        out = ctc_log_measure_for_a(float(ai), R, r_min, r_max)
        lm[i] = out["log_measure"]
        nb[i] = out["n_bands"]
        al[i] = out["alpha"]
    return {"t": t, "a": a, "log_measure": lm, "n_bands": nb, "alpha": al}


def adiabaticity_ratio(
    lyapunov_max: float, a_typical: float, R: float, r_band: float
) -> float:
    """Ratio (band crossing time) / (Lorenz e-folding time).

    < 1 means the spacetime can be treated as quasi-static against the chaotic
    rotation (adiabatic regime). band-crossing ~ proper circumference / c at the
    band's deepest point ~ 2 pi r_band (c = 1); Lorenz e-folding ~ 1/lyapunov_max.
    """
    t_band = 2.0 * np.pi * r_band
    t_efold = 1.0 / max(lyapunov_max, 1e-12)
    return float(t_band / t_efold)
