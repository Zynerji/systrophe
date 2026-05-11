"""Krasnikov ring: N tubes at evenly-spaced phase offsets.

Generalises the Krasnikov-Pair to a Z_N-symmetric array of N tubes at
phase offsets delta_k = 2 pi k / N for k = 0, 1, ..., N - 1.

Phasor superposition gives the total amplitude
    A_N = sum_{k=0}^{N-1} A_0 exp(i delta_k) = A_0 (1 - exp(i 2 pi)) / (1 - exp(i 2 pi / N))

For integer N >= 2 the geometric sum collapses to exactly zero:
    sum_{k=0}^{N-1} exp(i 2 pi k / N) = 0

so the Z_N-symmetric Krasnikov ring exhibits **exact extinction at
every N** (not just at the anti-phase delta = pi pair case). This is
a strong statement: 3, 4, 5, ... evenly-spaced tubes ALL cancel.

If we tilt one of the N tubes by epsilon away from its Z_N slot, the
extinction breaks and a residual amplitude proportional to epsilon
remains -- the catcher should flag this as a sharp resilience
transition.

This module
-----------
- ``krasnikov_ring_NEC_radial`` -- N-tube linear superposition
- ``krasnikov_ring_total_negative_energy`` -- integrated over x
- ``krasnikov_ring_extinction_breakdown`` -- |E_neg|(epsilon) sweep
- ``novelty_scan`` -- catcher over (N, epsilon) flagging the
  Z_N-protected extinction
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .krasnikov_tube import krasnikov_NEC_radial
from .novelty_catcher import scan_novelty


def krasnikov_ring_NEC_radial(
    x: float, t: float = 1.0,
    x_0: float = 0.0, t_0: float = 0.0,
    alpha: float = 4.0, N: int = 3, epsilon: float = 0.0,
) -> float:
    """N-tube Krasnikov ring wall NEC at (x, t).

    Tubes at phase offsets delta_k = 2 pi k / N for k = 0..N-1; one of
    them (k = 0) is tilted by epsilon (the symmetry-breaking
    perturbation). The pair amplitude collapses to a Dirichlet kernel
    sum.
    """
    base = krasnikov_NEC_radial(x, t, x_0, t_0, alpha)  # negative
    # Phasor sum: tube k has phase delta_k = 2 pi k / N, with tube 0
    # tilted by epsilon
    phasor = sum(
        math.cos((0.0 if k == 0 else 2 * math.pi * k / N) + (epsilon if k == 0 else 0.0))
        for k in range(N)
    )
    # The "effective" amplitude in the linear regime is |phasor|^2
    pair_factor = phasor ** 2
    return float(pair_factor * base)


def krasnikov_ring_total_negative_energy(
    N: int = 3, epsilon: float = 0.0, alpha: float = 4.0,
    x_range: tuple[float, float] = (-3.0, 3.0), n_x: int = 201,
    t: float = 1.0, x_0: float = 0.0, t_0: float = 0.0,
) -> float:
    """|E_neg| integrated over x for the N-tube ring."""
    xs = np.linspace(*x_range, n_x)
    vals = [krasnikov_ring_NEC_radial(float(x), t, x_0, t_0, alpha, N, epsilon)
            for x in xs]
    return float(np.trapezoid(np.array(vals), xs))


def krasnikov_ring_extinction_breakdown(
    N_values: list[int] | None = None,
    epsilon_grid: np.ndarray | None = None,
    alpha: float = 4.0,
) -> dict:
    """Two-dimensional sweep: |E_neg|(N, epsilon).

    Returns dict with grids and the |E_neg| surface.
    """
    if N_values is None:
        N_values = [2, 3, 4, 5, 6, 7, 8, 12]
    if epsilon_grid is None:
        epsilon_grid = np.linspace(0.0, math.pi / 2, 30)
    E_grid = np.zeros((len(N_values), len(epsilon_grid)))
    for i, N in enumerate(N_values):
        for j, eps in enumerate(epsilon_grid):
            E_grid[i, j] = krasnikov_ring_total_negative_energy(
                N=int(N), epsilon=float(eps), alpha=alpha,
            )
    return {
        "N_values": N_values,
        "epsilon_grid": epsilon_grid.tolist(),
        "E_neg_grid": E_grid.tolist(),
    }


def krasnikov_ring_with_noise_NEC_radial(
    x: float, t: float = 1.0,
    x_0: float = 0.0, t_0: float = 0.0,
    alpha: float = 4.0, N: int = 3,
    noise_amplitude: float = 0.0, seed: int = 0,
) -> float:
    """Z_N ring with Gaussian phase noise on every tube.

    Each tube k has its phase delta_k = 2 pi k / N perturbed by
    Gaussian noise of standard deviation noise_amplitude (in
    radians). The phasor sum is now random.
    """
    rng = np.random.default_rng(seed)
    base = krasnikov_NEC_radial(x, t, x_0, t_0, alpha)
    delta_k = 2 * math.pi * np.arange(N) / N
    if noise_amplitude > 0.0:
        delta_k = delta_k + rng.normal(0.0, noise_amplitude, size=N)
    phasor = float(np.sum(np.cos(delta_k)))
    return float(phasor ** 2 * base)


def krasnikov_ring_noise_robustness(
    N: int = 3, noise_grid: np.ndarray | None = None,
    n_trials: int = 100, alpha: float = 4.0,
    x_range: tuple[float, float] = (-3.0, 3.0), n_x: int = 51,
) -> dict:
    """Sweep noise_amplitude and run the catcher on the trial-averaged
    residual |E_neg|.

    Predicted: a sharp robustness threshold where the noise-driven
    residual amplitude exceeds the wall NEC scale, breaking the Z_N
    protection. The catcher should flag this threshold as a sharp
    transition between "protected" and "unprotected" regimes.
    """
    if noise_grid is None:
        noise_grid = np.linspace(0.0, math.pi, 30)
    xs = np.linspace(*x_range, n_x)
    residuals = np.zeros(len(noise_grid))
    for j, noise in enumerate(noise_grid):
        trial_residuals = []
        for trial in range(n_trials):
            vals = [
                krasnikov_ring_with_noise_NEC_radial(
                    float(x), 1.0, 0.0, 0.0, alpha, N,
                    float(noise), seed=trial,
                )
                for x in xs
            ]
            trial_residuals.append(
                abs(float(np.trapezoid(np.array(vals), xs)))
            )
        residuals[j] = float(np.mean(trial_residuals))
    def fn(nv):
        idx = int(np.argmin(np.abs(noise_grid - nv)))
        return np.array([float(residuals[idx])])
    result = scan_novelty(noise_grid, fn, n_bits=32)
    return {
        "N": N,
        "noise_grid": noise_grid.tolist(),
        "residual_E_neg": residuals.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }


def novelty_scan(
    N_values: list[int] | None = None,
    epsilon_range: tuple[float, float] = (0.0, math.pi / 2),
    n_eps: int = 30, alpha: float = 4.0,
) -> dict:
    """Catcher sweep over epsilon at each N value.

    The Z_N-protected extinction should give |E_neg| = 0 exactly at
    epsilon = 0 for every N. As epsilon grows the perturbation breaks
    the symmetry; the catcher should flag the OUT-OF-PROTECTION
    transition where |E_neg| begins growing nontrivially.
    """
    if N_values is None:
        N_values = [2, 3, 4, 5, 6, 8]
    eps_grid = np.linspace(*epsilon_range, n_eps)
    per_N = {}
    novel_Ns = []
    for N in N_values:
        E_neg = np.array([
            krasnikov_ring_total_negative_energy(
                N=int(N), epsilon=float(eps), alpha=alpha,
            )
            for eps in eps_grid
        ])
        # Cast E_neg into a richness-fix per-eps feature
        def fn(ev, E=E_neg):
            idx = int(np.argmin(np.abs(eps_grid - ev)))
            return np.array([float(E[idx])])
        result = scan_novelty(eps_grid, fn, n_bits=32)
        per_N[N] = {
            "E_neg": E_neg.tolist(),
            "extinction_at_eps_zero": float(E_neg[0]),
            "verdict": result.verdict,
            "n_sharp": len(result.sharp_features),
            "sharp_features": [
                {k: (int(v) if isinstance(v, np.integer) else v)
                 for k, v in s.items()}
                for s in result.sharp_features
            ],
        }
        if result.verdict == "novel_structure":
            novel_Ns.append(N)
    return {
        "N_values": N_values,
        "epsilon_grid": eps_grid.tolist(),
        "per_N": per_N,
        "novel_Ns": novel_Ns,
        "aggregate_verdict": (
            "novel_structure" if novel_Ns else
            ("uniform" if all(
                per_N[N]["verdict"] == "uniform" for N in N_values
            ) else "smooth")
        ),
    }
