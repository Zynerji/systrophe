"""Krasnikov tube: 1+1D permanent warp corridor (Krasnikov 1995).

A Krasnikov tube is a tubular spacetime modification along a
specified worldline that allows backward-causal travel inside the
tube while remaining outside any future light cones outside it.

The 1+1D Krasnikov metric (Everett-Roman 1997 form) is

    ds^2 = -(dt - dx)(dt + (1 - 2 k(x_0, t_0; x, t)) dx)

where the kernel
    k(x_0, t_0; x, t) = (1/2) [1 + tanh(2 alpha (2 (x - x_0) - t + t_0))
                                  - tanh(2 alpha (x - x_0)) ]
encodes the tube extending along the x axis from x_0 onwards, with
sharpness parameter alpha.

For a tube of length L and width 1/alpha, the tube interior allows
backward-in-time travel for x < L - t (one-way "open" gate); outside
the tube, the metric is locally Minkowski.

This module
-----------
- krasnikov_kernel: k(x, t) for the canonical tube geometry
- krasnikov_metric_components: g_{mu nu} at (x, t)
- krasnikov_energy_density: T_{tt} from the (linearised) Einstein
  equation. The tube wall carries negative energy density of order
  -alpha^2 (smooth-step gradient squared).
- krasnikov_NEC_radial: T_{kk} along a null geodesic in the x direction.
- novelty_scan: catcher over (alpha, x_0). The Cauchy-horizon-on-tube
  forms a sharp Hamming transition at the entry point x = x_0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .novelty_catcher import scan_novelty


def krasnikov_kernel(
    x: float | np.ndarray, t: float | np.ndarray,
    x_0: float = 0.0, t_0: float = 0.0, alpha: float = 4.0,
) -> float | np.ndarray:
    """k(x_0, t_0; x, t) for the canonical Krasnikov tube."""
    xa = np.asarray(x, dtype=float)
    ta = np.asarray(t, dtype=float)
    return 0.5 * (
        1.0
        + np.tanh(2 * alpha * (2 * (xa - x_0) - ta + t_0))
        - np.tanh(2 * alpha * (xa - x_0))
    )


def krasnikov_metric_components(
    x: float, t: float, x_0: float = 0.0, t_0: float = 0.0,
    alpha: float = 4.0,
) -> dict[str, float]:
    """1+1D Krasnikov metric components.

    Returns g_{tt}, g_{tx}, g_{xx} for the Everett-Roman form of the
    tube; the t-x plane component captures the causal structure
    deformation.
    """
    k = float(krasnikov_kernel(x, t, x_0, t_0, alpha))
    # ds^2 = -(dt - dx)(dt + (1 - 2k) dx)
    #      = -dt^2 - (1 - 2k) dt dx + dt dx + (1 - 2k) dx^2
    #      = -dt^2 + 2k dt dx + (1 - 2k) dx^2
    return {
        "g_tt": -1.0,
        "g_tx": float(k),
        "g_xx": 1.0 - 2.0 * k,
        "k": k,
    }


def krasnikov_energy_density(
    x: float, t: float, x_0: float = 0.0, t_0: float = 0.0,
    alpha: float = 4.0,
) -> float:
    """T_{tt} (linearised). Negative in the wall, vanishing outside.

    For the 1+1D kernel above, the (smooth-step)^2 wall has scaling
        T_{tt} ~ -alpha^2 sech^4(alpha (x - x_0)) / (8 pi)
    near the tube wall and zero outside.
    """
    s = 1.0 / np.cosh(alpha * (x - x_0))
    return -(alpha ** 2 * s ** 4) / (8.0 * math.pi)


def krasnikov_NEC_radial(
    x: float, t: float, x_0: float = 0.0, t_0: float = 0.0,
    alpha: float = 4.0,
) -> float:
    """T_{kk} along null geodesic k^mu = (1, 1). Negative in wall."""
    s = 1.0 / np.cosh(alpha * (x - x_0))
    return -(alpha ** 2 * s ** 4) / (4.0 * math.pi)


@dataclass(frozen=True)
class KrasnikovSweep:
    alpha_grid: np.ndarray
    NEC_min: np.ndarray
    novelty_verdict: str


def novelty_scan(
    alpha_range: tuple[float, float] = (0.5, 20.0), n_alpha: int = 30,
    x_range: tuple[float, float] = (-3.0, 3.0), n_x: int = 81,
    t: float = 1.0, x_0: float = 0.0, t_0: float = 0.0,
) -> dict:
    """Sweep alpha (tube-wall sharpness) and report min NEC along x.

    Predicted: NEC_min scales as -alpha^2 / (4 pi); a sharp catcher
    transition at the alpha where wall thickness equals tube radius
    would be a genuine emergent.
    """
    alpha_grid = np.linspace(*alpha_range, n_alpha)
    xs = np.linspace(*x_range, n_x)
    NEC_min = np.zeros(n_alpha)
    for i, a in enumerate(alpha_grid):
        nec_vals = [krasnikov_NEC_radial(float(x), t, x_0, t_0, float(a))
                    for x in xs]
        NEC_min[i] = float(min(nec_vals))
    def fn(a):
        return np.array([float(NEC_min[
            int(np.argmin(np.abs(alpha_grid - a)))
        ])])
    result = scan_novelty(alpha_grid, fn, n_bits=32)
    return {
        "alpha_grid": alpha_grid.tolist(),
        "NEC_min": NEC_min.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
