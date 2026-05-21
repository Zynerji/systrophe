"""Growth catcher: address-space novelty for sustained exponential trends.

The base ``scan_novelty`` catcher flags Hamming-step JUMPS; the
``derivative_catcher`` reduces a smooth sigmoid TRANSITION to a jump in
the derivative. Neither sees a sustained smooth TREND — e.g. an
exponentially growing amplitude track A(u) ~ e^{sigma u}. A pure
exponential has no Hamming-step outlier (every step is the same size),
and the per-third ``catch_novelty_per_quantity`` self-normalises each
third to its own range, erasing the magnitude shift between thirds. This
is the documented catcher-domain boundary that showed up on the 3-SAT
sigmoid (see FINDINGS_MILLENNIUM_PROGRESS.md) and again on the off-line
zeta-zero growth signature (FINDINGS_PRIMES_OFFLINE_ZERO.md).

The growth catcher closes that gap with two complementary signals:

  1. ADDRESS-SPACE (magnitude-aware). Encode every point with a single
     GLOBAL rank-thermometer (shared scale across all points, not
     per-segment), so a sustained trend marches the address monotonically
     from low occupancy to high occupancy. The endpoint Hamming
     displacement and the monotone-step fraction are then the
     address-space signature of a trend — the fix for the
     self-normalisation blind spot.

  2. STATISTICAL. Regress log|value| on the parameter to get the growth
     exponent sigma_hat, and calibrate it against a permutation null
     (shuffle the parameter-value pairing) to get a z-score and p-value.
     This separates a real trend from a noisy flat line.

Verdict: ``growing`` | ``decaying`` | ``stationary``.

For an exponential mode A(u) = A0 e^{sigma u}, sigma_hat recovers sigma.
This is exactly what is needed to read an off-line zeta zero's
Re(rho) - 1/2 = sigma off a measured prime-fluctuation amplitude track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GrowthCatchResult:
    """Result of a growth-trend catch."""

    parameter_label: str
    n_points: int
    growth_exponent: float        # slope of log|value| vs parameter
    r_squared: float              # quality of the log-linear fit
    significance_z: float         # sigma_hat vs permutation-null std
    null_slope_std: float         # std of slope under the permutation null
    p_value: float                # two-sided permutation p-value
    endpoint_drift: float         # address-space, in [0, 1]
    monotone_fraction: float      # fraction of rank-increasing steps, [0, 1]
    verdict: str                  # growing | decaying | stationary


def _global_rank_thermometer(values: np.ndarray, n_bits: int) -> np.ndarray:
    """Address each value by a thermometer code of its GLOBAL rank.

    Shared scale across all points: smallest value -> 0 bits set,
    largest -> n_bits set. This is the magnitude-aware encoding the
    per-segment catcher lacks.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    ranks = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    denom = max(n - 1, 1)
    addrs = np.zeros((n, n_bits), dtype=int)
    for i in range(n):
        b = int(round(ranks[i] / denom * n_bits))
        addrs[i, :b] = 1
    return addrs


def catch_growth(
    parameter_values: np.ndarray,
    values: np.ndarray,
    n_bits: int = 32,
    n_perm: int = 400,
    z_threshold: float = 3.0,
    p_threshold: float = 0.01,
    parameter_label: str = "parameter",
    seed: int = 0,
) -> GrowthCatchResult:
    """Detect and quantify a sustained exponential trend in ``values``.

    Parameters
    ----------
    parameter_values, values
        Equal-length 1D arrays. ``values`` is treated as a positive
        amplitude track; non-finite and non-positive entries are dropped
        before the log-linear fit (a small floor guards against zeros).
    n_bits
        Width of the address-space rank-thermometer encoding.
    n_perm
        Permutation-null resamples for the significance estimate.
    z_threshold, p_threshold
        A trend is declared only if ``|z| >= z_threshold`` AND
        ``p_value <= p_threshold`` (both must agree).
    """
    p = np.asarray(parameter_values, dtype=float)
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(p) & np.isfinite(v)
    p, v = p[mask], v[mask]
    n = len(v)
    if n < 5:
        return GrowthCatchResult(
            parameter_label, n, float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"),
            "stationary")

    # Log-linear fit. Floor guards against non-positive amplitudes.
    floor = 1e-12 * (np.max(np.abs(v)) + 1e-300)
    y = np.log(np.maximum(np.abs(v), floor))
    slope, intercept = np.polyfit(p, y, 1)
    yhat = slope * p + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Permutation null: break the parameter-value pairing.
    rng = np.random.default_rng(seed)
    perm_slopes = np.empty(n_perm)
    for i in range(n_perm):
        slope_i, _ = np.polyfit(p, rng.permutation(y), 1)
        perm_slopes[i] = slope_i
    perm_std = float(np.std(perm_slopes))
    z = float(slope / perm_std) if perm_std > 0 else 0.0
    p_value = float((np.sum(np.abs(perm_slopes) >= abs(slope)) + 1)
                    / (n_perm + 1))

    # Address-space signature (magnitude-aware, global scale).
    addrs = _global_rank_thermometer(v, n_bits)
    endpoint_drift = float(np.sum(addrs[0] != addrs[-1]) / n_bits)
    ranks = np.argsort(np.argsort(v, kind="stable"), kind="stable")
    dr = np.diff(ranks)
    n_up = int(np.sum(dr > 0))
    n_dn = int(np.sum(dr < 0))
    monotone_fraction = (n_up / (n_up + n_dn)) if (n_up + n_dn) else 0.5

    significant = (abs(z) >= z_threshold) and (p_value <= p_threshold)
    if significant and slope > 0:
        verdict = "growing"
    elif significant and slope < 0:
        verdict = "decaying"
    else:
        verdict = "stationary"

    return GrowthCatchResult(
        parameter_label=parameter_label,
        n_points=n,
        growth_exponent=float(slope),
        r_squared=float(r_squared),
        significance_z=z,
        null_slope_std=perm_std,
        p_value=p_value,
        endpoint_drift=endpoint_drift,
        monotone_fraction=float(monotone_fraction),
        verdict=verdict,
    )


def summarize_growth_for_report(result: GrowthCatchResult) -> str:
    """One-line summary for inclusion in paper / commit message."""
    return (
        f"Growth catcher: verdict='{result.verdict}', "
        f"sigma_hat={result.growth_exponent:+.4f} "
        f"(z={result.significance_z:.1f}, p={result.p_value:.3g}, "
        f"R^2={result.r_squared:.2f}), "
        f"endpoint_drift={result.endpoint_drift:.2f}, "
        f"monotone={result.monotone_fraction:.2f}"
    )
