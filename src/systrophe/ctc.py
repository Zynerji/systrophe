"""CTC region detection from g_{phi phi}(r) sign.

For a stationary axisymmetric metric in coordinates with phi -> phi + 2 pi
identification, a closed phi-orbit at fixed (t, r, z) is timelike iff
g_{phi phi}(r) < 0, hence a closed timelike curve.

This module is a pure utility: hand it any callable r -> L(r) and it
returns the intervals in [r_min, r_max] where L < 0.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import brentq


def _refine_zero(
    L_fn: Callable[[float], float], r_lo: float, r_hi: float
) -> float:
    """Refine a sign-change to a precise zero by bisection (brentq)."""
    return float(brentq(lambda r: L_fn(r), r_lo, r_hi, xtol=1e-12, rtol=1e-10))


def find_ctc_intervals(
    L_fn: Callable[[float | np.ndarray], np.ndarray],
    r_min: float,
    r_max: float,
    n_grid: int = 2001,
    atol: float = 1e-12,
) -> list[tuple[float, float]]:
    """Return [(r_a, r_b), ...] intervals in [r_min, r_max] where L_fn < -atol.

    Uses a dense linear grid to detect sign changes, then bisects each
    sign change to find the precise zero. Adjacent zeros bound CTC
    intervals. Values with |L| <= atol are treated as zero to suppress
    spurious CTC reports from floating-point noise.
    """
    if r_min <= 0 or r_max <= r_min:
        raise ValueError("require 0 < r_min < r_max")
    if n_grid < 16:
        raise ValueError("need at least 16 grid points")
    r_grid = np.linspace(r_min, r_max, n_grid)
    L_vals = np.asarray(L_fn(r_grid), dtype=float)

    # Sign with deadband: anything within atol of zero is treated as positive
    signs = np.where(L_vals < -atol, -1.0, 1.0)
    change_idx = np.where(np.diff(signs) != 0)[0]
    zeros = []
    for i in change_idx:
        r_lo, r_hi = r_grid[i], r_grid[i + 1]
        L_lo = float(L_fn(r_lo))
        L_hi = float(L_fn(r_hi))
        if L_lo * L_hi < 0:
            zeros.append(_refine_zero(lambda r, _f=L_fn: float(_f(r)), r_lo, r_hi))
        elif L_lo == 0.0:
            zeros.append(float(r_lo))
        elif L_hi == 0.0:
            zeros.append(float(r_hi))

    intervals: list[tuple[float, float]] = []
    if L_vals[0] < -atol:
        edges = [r_min] + zeros
    else:
        edges = zeros[:]
    if len(edges) % 2 == 1:
        edges = edges + [r_max]
    for j in range(0, len(edges), 2):
        intervals.append((edges[j], edges[j + 1]))
    return intervals


def has_ctc(
    L_fn: Callable[[float | np.ndarray], np.ndarray],
    r_min: float,
    r_max: float,
    n_grid: int = 2001,
    atol: float = 1e-12,
) -> bool:
    """True iff L_fn(r) < -atol anywhere in [r_min, r_max]."""
    r_grid = np.linspace(r_min, r_max, n_grid)
    return bool(np.any(np.asarray(L_fn(r_grid), dtype=float) < -atol))
