"""Krasnikov-Pair: anti-phase extinction of engineered tubes.

Two Krasnikov tubes at the same spatial location but offset in their
phase parameter delta superpose like a SystrophePair. The same
phasor-addition identity that extinguishes Tipler-pair CTC bands at
delta = pi should extinguish the Krasnikov-pair wall NEC at delta = pi.

Mathematics
-----------
Each tube wall has NEC density T_{kk}^{(i)}(x; alpha_i, x_0_i, t_0_i, delta_i).
For two co-located tubes with matched alpha, x_0, t_0 but offset delta,
the linear-superposition rule gives

    T_{kk}^{pair}(x; delta) = T_{kk}^{(1)}(x) + T_{kk}^{(2)}(x; delta).

If each tube contributes a Krasnikov NEC that we model as carrying a
phase factor exp(i delta) in the linearised regime, the wall
amplitude at delta is

    A_pair(delta) = A_0 * (1 + exp(i delta)) = 2 A_0 cos(delta/2) exp(i delta/2)

so |A_pair|^2 = 4 |A_0|^2 cos^2(delta/2), vanishing exactly at
delta = pi. This is the canonical SystrophePair extinction identity
applied to the engineered tube wall.

The result is that two Krasnikov tubes at anti-phase RESPECT NEC
locally -- the wall energy is positive (additive) at delta = 0 and
zero at delta = pi.

This module
-----------
- ``krasnikov_pair_NEC_radial`` -- linearly-superposed wall NEC
- ``krasnikov_pair_total_negative_energy`` -- integrated over x
- ``krasnikov_pair_extinction_curve`` -- |E_neg|(delta) for the sweep
- ``novelty_scan`` -- catcher over delta with explicit bracket around
  delta = pi
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .krasnikov_tube import krasnikov_NEC_radial
from .novelty_catcher import scan_novelty


def krasnikov_pair_NEC_radial(
    x: float, t: float = 1.0,
    x_0: float = 0.0, t_0: float = 0.0,
    alpha: float = 4.0, delta: float = 0.0,
) -> float:
    """Phasor-superposed wall NEC at (x, t).

    The two tubes are at the same spatial location with relative phase
    delta. The pair amplitude scales as 2 cos(delta/2) (from the
    phasor identity), so |NEC|_pair = 4 cos^2(delta/2) * |NEC|_single.
    """
    base = krasnikov_NEC_radial(x, t, x_0, t_0, alpha)  # negative
    pair_factor = 4.0 * math.cos(delta / 2.0) ** 2
    return float(pair_factor * base)


def krasnikov_pair_total_negative_energy(
    delta: float = 0.0, alpha: float = 4.0,
    x_range: tuple[float, float] = (-3.0, 3.0), n_x: int = 201,
    t: float = 1.0, x_0: float = 0.0, t_0: float = 0.0,
) -> float:
    """Wall NEC integrated over x for the pair at offset delta."""
    xs = np.linspace(*x_range, n_x)
    vals = [krasnikov_pair_NEC_radial(float(x), t, x_0, t_0, alpha, delta)
            for x in xs]
    return float(np.trapezoid(np.array(vals), xs))


def krasnikov_pair_extinction_curve(
    delta_grid: np.ndarray | None = None,
    alpha: float = 4.0,
    x_range: tuple[float, float] = (-3.0, 3.0), n_x: int = 201,
) -> tuple[np.ndarray, np.ndarray]:
    """|E_neg|(delta) for the pair sweep. Returns (delta_grid, E_neg)."""
    if delta_grid is None:
        delta_grid = np.linspace(0.0, 2.0 * math.pi, 51)
    E = np.array([
        krasnikov_pair_total_negative_energy(
            delta=float(d), alpha=alpha, x_range=x_range, n_x=n_x,
        )
        for d in delta_grid
    ])
    return delta_grid, E


@dataclass(frozen=True)
class KrasnikovPairSweep:
    delta_grid: np.ndarray
    E_neg_total: np.ndarray
    novelty_verdict: str


def novelty_scan(
    delta_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    n_delta: int = 30, alpha: float = 4.0,
) -> dict:
    """Sweep delta and run the catcher on the absolute total negative
    energy across the pair.

    Predicted: the phasor identity gives |E_neg|(delta) ~ cos^2(delta/2);
    a smooth bell-curve vanishing at delta = pi. The catcher should
    NOT flag this as novel (the curve is everywhere smooth). If it
    DOES flag, that is a genuine emergent: non-trivial extinction
    behaviour beyond the phasor identity.
    """
    delta_grid, E = krasnikov_pair_extinction_curve(
        delta_grid=np.linspace(*delta_range, n_delta),
        alpha=alpha,
    )
    def fn(dv):
        idx = int(np.argmin(np.abs(delta_grid - dv)))
        return np.array([float(E[idx])])
    result = scan_novelty(delta_grid, fn, n_bits=32)
    # Identify the extinction angle (numerical minimum of |E|)
    extinction_idx = int(np.argmin(np.abs(E)))
    return {
        "delta_grid": delta_grid.tolist(),
        "E_neg_total": E.tolist(),
        "extinction_delta": float(delta_grid[extinction_idx]),
        "extinction_E_neg": float(E[extinction_idx]),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
