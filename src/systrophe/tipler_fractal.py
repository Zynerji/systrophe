"""Discrete-scale invariance (DSI) and fractal extension of the Tipler sinusoid.

The Tipler sinusoid F(r) = A cos(alpha * ln(r/R) + delta) has the
classical signature of *discrete-scale invariance*: it is invariant
under the rescaling r -> lambda r when

    lambda = exp(2 pi / alpha).

Equivalently, its zero set {r_n} forms a geometric progression with
ratio

    rho = exp(pi / alpha).

A single log-periodic cosine is *not* a fractal in the
Minkowski-Bouligand sense: its zero set is countable with box-counting
dimension zero. However, the *natural multi-scale extension* obtained
by superposing log-periodic cosines with a hierarchy of alpha-scales
(equivalent to a hierarchy of co-axial cylinders with different
rotation parameters a) produces structures with non-trivial box-counting
dimension --- which is the actual mathematical content behind Grok's
"Tipler sinusoid extends to a fractal" assertion.

References
----------
- D. Sornette, "Discrete scale invariance and complex dimensions",
  Phys. Rep. 297 (1998) 239.
- F. Tipler, Phys. Rev. D 9 (1974) 2203.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .sinusoid import TiplerSinusoid


def zero_geometric_ratio(alpha: float) -> float:
    """Geometric ratio between consecutive zeros of a Tipler sinusoid.

    Zeros at u_n = (pi/2 - delta + n pi) / alpha in u = ln(r/R), so the
    ratio of consecutive r-zeros is exp(pi / alpha).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return float(np.exp(pi / alpha))


def dsi_rescaling_factor(alpha: float) -> float:
    """Full DSI period: lambda = exp(2 pi / alpha).

    F(lambda r) = F(r) for any phase delta.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return float(np.exp(2 * pi / alpha))


def zero_set(
    R: float, alpha: float, delta: float, r_min: float, r_max: float
) -> np.ndarray:
    """Zeros of F(r) = cos(alpha ln(r/R) + delta) in [r_min, r_max]."""
    if R <= 0 or r_min <= 0 or r_max <= r_min:
        raise ValueError("require R > 0 and 0 < r_min < r_max")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    u_min = np.log(r_min / R)
    u_max = np.log(r_max / R)
    k_min = int(np.ceil((alpha * u_min + delta - pi / 2) / pi))
    k_max = int(np.floor((alpha * u_max + delta - pi / 2) / pi))
    ks = np.arange(k_min, k_max + 1)
    u = (pi / 2 + ks * pi - delta) / alpha
    return R * np.exp(u)


def verify_geometric_progression(zeros: np.ndarray) -> dict:
    """Check whether a zero set is a geometric progression.

    Returns dict with:
      - ratio_mean : mean ratio of consecutive entries
      - ratio_std  : standard deviation of those ratios
      - max_rel_dev: max relative deviation from the mean ratio
      - is_geometric: True iff max_rel_dev < 1e-9
    """
    if len(zeros) < 3:
        raise ValueError("need at least 3 zeros to test geometric progression")
    ratios = zeros[1:] / zeros[:-1]
    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios))
    max_rel_dev = float(np.max(np.abs(ratios - mean_r) / mean_r))
    return {
        "ratio_mean": mean_r,
        "ratio_std": std_r,
        "max_rel_dev": max_rel_dev,
        "is_geometric": max_rel_dev < 1e-9,
    }


def ctc_band_log_widths(
    sinusoid: TiplerSinusoid, r_min: float, r_max: float, n_grid: int = 8001
) -> np.ndarray:
    """Log-widths of consecutive CTC bands {L < 0}.

    For a pure Tipler sinusoid, these widths are constant in u = ln(r/R)
    (each half-period contributes equally) --- a direct signature of DSI.
    """
    from .ctc import find_ctc_intervals

    bands = find_ctc_intervals(sinusoid.L, r_min=r_min, r_max=r_max, n_grid=n_grid)
    if not bands:
        return np.array([], dtype=float)
    return np.array([np.log(b / a) for a, b in bands], dtype=float)


def box_count_log_dimension(
    points: np.ndarray, u_min: float, u_max: float, n_scales: int = 20
) -> dict:
    """Box-counting dimension of `points` in u = ln(r) coordinates.

    For a discrete countable set (e.g. pure log-periodic zeros), this
    returns 0 in the strict limit. For a dense self-similar Cantor-like
    set (e.g. multi-cascade DSI), it returns a non-trivial dimension.

    Parameters
    ----------
    points : np.ndarray
        Points in real r (the function converts to u = ln r internally).
    u_min, u_max : float
        u-range (already in log coords) to box-count over.
    n_scales : int
        Number of box scales eps_k = (u_max - u_min) * 2^-k to test.
    """
    if len(points) == 0:
        return {"dimension": 0.0, "scales": np.array([]), "counts": np.array([])}
    u_pts = np.log(points)
    u_pts = u_pts[(u_pts >= u_min) & (u_pts <= u_max)]
    if len(u_pts) == 0:
        return {"dimension": 0.0, "scales": np.array([]), "counts": np.array([])}

    scales = []
    counts = []
    total_u = u_max - u_min
    for k in range(1, n_scales + 1):
        eps = total_u / (2 ** k)
        # Boxes indexed by floor((u - u_min) / eps)
        idx = np.floor((u_pts - u_min) / eps).astype(int)
        n_occupied = len(np.unique(idx))
        scales.append(eps)
        counts.append(n_occupied)
    scales = np.array(scales)
    counts = np.array(counts)
    # Discard the tail where every point gets its own box (resolution
    # saturation) by fitting only the regime where counts grows.
    n_pts = len(u_pts)
    valid = counts < n_pts
    if valid.sum() < 3:
        # Cannot fit a slope; report 0 (the discrete-set answer)
        return {"dimension": 0.0, "scales": scales, "counts": counts}
    log_eps = np.log(scales[valid])
    log_N = np.log(counts[valid])
    slope, _ = np.polyfit(log_eps, log_N, 1)
    return {
        "dimension": float(-slope),
        "scales": scales,
        "counts": counts,
    }


@dataclass(frozen=True)
class CascadeDSI:
    """Multi-scale extension of a Tipler sinusoid: a hierarchy of log-periodic
    cosines with cascading alpha-scales.

    The total envelope is

        F_cascade(r) = sum_{k=0..L-1} A_k cos(alpha_k * ln(r/R) + delta_k)

    where alpha_k = alpha_0 * scale_factor**k and A_k = A_0 * amp_decay**k.

    This is the natural multi-cylinder cascade that produces a fractal-like
    zero/CTC structure --- and gives mathematical content to the otherwise
    handwaved "Tipler sinusoid is a fractal" claim.
    """

    R: float
    alpha_0: float
    A_0: float
    delta_0: float
    levels: int
    scale_factor: float = 2.0
    amp_decay: float = 0.5

    def __post_init__(self) -> None:
        if self.R <= 0 or self.alpha_0 <= 0 or self.levels < 1:
            raise ValueError("require R > 0, alpha_0 > 0, levels >= 1")
        if self.scale_factor <= 1:
            raise ValueError("scale_factor must exceed 1 to produce cascade")
        if not (0 < self.amp_decay < 1):
            raise ValueError("amp_decay must be in (0, 1)")

    def alphas(self) -> np.ndarray:
        return self.alpha_0 * self.scale_factor ** np.arange(self.levels)

    def amps(self) -> np.ndarray:
        return self.A_0 * self.amp_decay ** np.arange(self.levels)

    def F(self, r: float | np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        u = np.log(r / self.R)
        total = np.zeros_like(r)
        for alpha_k, A_k in zip(self.alphas(), self.amps()):
            total = total + A_k * np.cos(alpha_k * u + self.delta_0)
        return total

    def zeros(self, r_min: float, r_max: float, n_grid: int = 100_001) -> np.ndarray:
        """Approximate zeros of F_cascade on [r_min, r_max] by sign changes."""
        rs = np.linspace(r_min, r_max, n_grid)
        Fs = self.F(rs)
        sign = np.sign(Fs)
        flips = np.where(np.diff(sign) != 0)[0]
        # Linear interpolation refinement
        zeros = []
        for i in flips:
            r1, r2 = rs[i], rs[i + 1]
            F1, F2 = Fs[i], Fs[i + 1]
            zeros.append(r1 - F1 * (r2 - r1) / (F2 - F1))
        return np.array(zeros, dtype=float)


def base_tipler_sinusoid_is_dsi(sinusoid: TiplerSinusoid, n_periods: int = 6) -> dict:
    """Diagnostic: confirm a base Tipler sinusoid has trivial DSI.

    Returns:
      - geometric_ratio: ratio of consecutive zeros
      - expected_ratio: exp(pi / alpha)
      - rel_error: relative error between the two
      - n_zeros_tested
      - is_dsi: True iff zeros form a clean geometric progression to <1e-9

    A pure Tipler sinusoid passes this test exactly.
    """
    R = sinusoid.R
    alpha = sinusoid.alpha
    # Sample enough zeros: log-span 2*pi*n_periods/alpha
    u_span = 2 * pi * n_periods / alpha
    r_min = R * np.exp(0.1)
    r_max = R * np.exp(u_span)
    zs = zero_set(R, alpha, sinusoid.delta, r_min, r_max)
    if len(zs) < 3:
        raise RuntimeError(
            f"need at least 3 zeros (got {len(zs)}); increase n_periods"
        )
    check = verify_geometric_progression(zs)
    expected = zero_geometric_ratio(alpha)
    return {
        "geometric_ratio": check["ratio_mean"],
        "expected_ratio": expected,
        "rel_error": abs(check["ratio_mean"] - expected) / expected,
        "n_zeros_tested": len(zs),
        "is_dsi": check["is_geometric"] and abs(check["ratio_mean"] - expected) / expected < 1e-9,
    }


def cascade_box_dimension(
    cascade: CascadeDSI, r_min: float, r_max: float, n_scales: int = 18
) -> dict:
    """Box-counting dimension of a CascadeDSI zero set in u = ln(r) coords.

    For a single-level cascade (levels=1), this is essentially 0 (discrete
    geometric progression). For levels >= 2 with sufficient scale_factor,
    the zero set becomes self-similar at multiple scales and the box
    dimension exceeds 0.

    Returns the box_count_log_dimension result plus zero-count metadata.
    """
    zs = cascade.zeros(r_min, r_max)
    u_min = np.log(r_min)
    u_max = np.log(r_max)
    bc = box_count_log_dimension(zs, u_min, u_max, n_scales=n_scales)
    bc["n_zeros"] = len(zs)
    return bc


def dsi_extension_from_sinusoid(
    base: TiplerSinusoid,
    levels: int,
    scale_factor: float = 2.0,
    amp_decay: float = 0.5,
) -> CascadeDSI:
    """Build a multi-scale CascadeDSI from a base TiplerSinusoid.

    Cascade level 0 reproduces the base; subsequent levels add
    alpha_k = base.alpha * scale_factor**k and A_k = base.A * amp_decay**k.
    """
    return CascadeDSI(
        R=base.R,
        alpha_0=base.alpha,
        A_0=base.A,
        delta_0=base.delta,
        levels=levels,
        scale_factor=scale_factor,
        amp_decay=amp_decay,
    )
