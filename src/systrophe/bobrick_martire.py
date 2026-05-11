"""Bobrick-Martire 2021: generalised warp-bubble class.

Bobrick and Martire (2021) classify all warp-drive solutions to the
Einstein equations and identify a one-parameter family in which the
required exotic-matter density continuously interpolates between
Alcubierre-extreme (large negative density) and an asymptotically
flat near-classical regime at subluminal velocities.

Their key insight: the energy density inside a warp bubble is
        T_{tt} = m_ADM * F(x, rho)
where m_ADM is the ADM mass of the bubble, and F is a fixed shape
function (no longer requiring v_s > c for non-trivial structure).
Subluminal bubbles with positive m_ADM are physically reasonable;
the FTL Alcubierre limit emerges only as m_ADM -> -infinity.

This module
-----------
- bm_shape: F(x, rho), a shape function with finite support
- bm_metric_components: g_{mu nu} at a point, parameterised by m_ADM
- bm_energy_density: T_{tt} = m_ADM * F
- bm_NEC_radial: T_{kk} along radial null geodesic; sign-controlled
  by m_ADM
- bm_ADM_mass_scan: novelty scan over m_ADM. The catcher should flag
  the m_ADM = 0 transition (sign flip of NEC) as a sharp transition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .novelty_catcher import scan_novelty


def bm_shape(
    x: float | np.ndarray, rho: float | np.ndarray,
    R: float = 1.0, sigma: float = 4.0,
) -> float | np.ndarray:
    """Bobrick-Martire shape function: smooth bump of unit max.

    F = sech(sigma x) * sech(sigma (rho - R)) so the bubble sits at
    x = 0, rho = R, with width 1/sigma.
    """
    return 1.0 / (np.cosh(sigma * np.asarray(x, dtype=float))
                   * np.cosh(sigma * (np.asarray(rho, dtype=float) - R)))


def bm_metric_components(
    x: float, rho: float, m_ADM: float = 1.0,
    R: float = 1.0, sigma: float = 4.0,
) -> dict[str, float]:
    """Newtonian-gauge metric for a small-amplitude warp bubble.

    g_{tt} = -(1 + 2 m_ADM F),  g_{ii} = (1 - 2 m_ADM F).
    Linearised gravity regime valid for |m_ADM F| << 1.
    """
    F = float(bm_shape(x, rho, R, sigma))
    return {
        "g_tt": -(1.0 + 2.0 * m_ADM * F),
        "g_xx": 1.0 - 2.0 * m_ADM * F,
        "g_yy": 1.0 - 2.0 * m_ADM * F,
        "g_zz": 1.0 - 2.0 * m_ADM * F,
        "F": F,
        "m_ADM": float(m_ADM),
    }


def bm_energy_density(
    x: float, rho: float, m_ADM: float = 1.0,
    R: float = 1.0, sigma: float = 4.0,
) -> float:
    """T_{tt} = m_ADM * F (linearised). Negative for m_ADM < 0."""
    return float(m_ADM * bm_shape(x, rho, R, sigma))


def bm_NEC_radial(
    x: float, rho: float, m_ADM: float = 1.0,
    R: float = 1.0, sigma: float = 4.0,
) -> float:
    """T_{kk} along outward radial null geodesic.

    For linearised B-M, T_{kk} ~ m_ADM (F + grad F . k k). The
    SIGN of T_{kk} is controlled by m_ADM. Strict NEC violation
    happens iff m_ADM < 0 (exotic matter regime).
    """
    F = float(bm_shape(x, rho, R, sigma))
    # Gradient contribution (small for smooth F)
    eps = 1e-3
    F_p = float(bm_shape(x + eps, rho, R, sigma))
    F_m = float(bm_shape(x - eps, rho, R, sigma))
    dF_dx = (F_p - F_m) / (2 * eps)
    return float(m_ADM * (F + 0.5 * dF_dx ** 2))


@dataclass(frozen=True)
class BMSweep:
    m_ADM_grid: np.ndarray
    NEC_min: np.ndarray
    novelty_verdict: str


def novelty_scan(
    m_ADM_range: tuple[float, float] = (-2.0, 2.0),
    n_m: int = 31, R: float = 1.0, sigma: float = 4.0,
    x_range: tuple[float, float] = (-2.0, 2.0),
    rho_range: tuple[float, float] = (0.0, 3.0),
    n_x: int = 21, n_rho: int = 21,
) -> dict:
    """Sweep m_ADM and report min/max NEC.

    The predicted m_ADM = 0 transition (where NEC flips sign) is a
    canonical structural transition; the catcher must flag it as a
    sharp Hamming step. If not, sweep too sparse or shape too smooth.
    """
    m_grid = np.linspace(*m_ADM_range, n_m)
    xs = np.linspace(*x_range, n_x)
    rhos = np.linspace(*rho_range, n_rho)
    NEC_min = np.zeros(n_m)
    NEC_max = np.zeros(n_m)
    for i, m in enumerate(m_grid):
        nec_vals = []
        for x in xs:
            for rho in rhos:
                nec_vals.append(bm_NEC_radial(float(x), float(rho),
                                               m_ADM=float(m),
                                               R=R, sigma=sigma))
        arr = np.array(nec_vals)
        NEC_min[i] = float(arr.min())
        NEC_max[i] = float(arr.max())
    def fn(m):
        return np.array([float(NEC_min[
            int(np.argmin(np.abs(m_grid - m)))
        ])])
    result = scan_novelty(m_grid, fn, n_bits=32)
    return {
        "m_ADM_grid": m_grid.tolist(),
        "NEC_min": NEC_min.tolist(),
        "NEC_max": NEC_max.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
