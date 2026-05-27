"""Aharonov-Bohm phase for closed loops encircling chronology horizons.

A charged particle traversing a closed loop accumulates a phase
   phi_AB = (e/hbar) oint A_mu dx^mu
For the LP rotating cylinder, the analog "vector potential" is the
shift K(r) (frame-dragging). A loop at constant r encircles K(r) flux
giving:
   phi_AB(r) = 2 pi K(r)

This phase changes sign as the loop crosses a chronology horizon (F=0),
because K changes sign there too. The Aharonov-Bohm phase therefore
witnesses the chronology-horizon structure topologically.

Module:
- aharonov_bohm_phase: phi_AB(r) at radius r
- phase_jump_across_CH: discontinuity at F=0
- enclosed_flux: 2 pi K(r) (the "magnetic flux")
- topological_invariant_winding: integer winding around all CHs
- pair_extinction_AB: at delta=pi, K -> 0, AB -> 0
- novelty_scan: catcher on phase profile
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior


def aharonov_bohm_phase(vs: VanStockumInterior, r: float) -> float:
    """phi_AB = 2 pi K(r) in natural units."""
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    return float(2 * math.pi * K)


def enclosed_flux(vs: VanStockumInterior, r: float) -> float:
    """K(r) -- analog magnetic flux enclosed by the phi-loop."""
    return float(vs.analytic_exterior_K(np.array([r]))[0])


def phase_jump_across_CH(
    vs: VanStockumInterior, r_CH: float, eps: float = 0.01,
) -> dict:
    """Phase discontinuity across a chronology horizon."""
    phase_inside = aharonov_bohm_phase(vs, r_CH - eps)
    phase_outside = aharonov_bohm_phase(vs, r_CH + eps)
    jump = phase_outside - phase_inside
    return {
        "r_CH": r_CH,
        "phase_inside": phase_inside,
        "phase_outside": phase_outside,
        "phase_jump": float(jump),
    }


def topological_invariant_winding(
    vs: VanStockumInterior, r_min: float = None,
    r_max: float = None, n_samples: int = 500,
) -> int:
    """Total phase winding around all CHs in [r_min, r_max]."""
    if r_min is None:
        r_min = vs.R * 1.05
    if r_max is None:
        r_max = vs.R * 20.0
    r_grid = np.linspace(r_min, r_max, n_samples)
    phases = np.array([aharonov_bohm_phase(vs, float(r)) for r in r_grid])
    unwrapped = np.unwrap(phases)
    total_winding = (unwrapped[-1] - unwrapped[0]) / (2 * math.pi)
    return int(round(total_winding))


def pair_extinction_AB(
    vs: VanStockumInterior, r: float, delta: float,
) -> dict:
    """Pair AB phase: phi = 2 pi K(r) * (1+cos delta)/2."""
    base = aharonov_bohm_phase(vs, r)
    extinction = 0.5 * (1.0 + math.cos(delta))
    return {
        "r": r,
        "delta": delta,
        "extinction_factor": extinction,
        "phi_single": base,
        "phi_pair": float(base * extinction),
        "trivial_at_pi": bool(abs(delta - math.pi) < 1e-9),
    }


def detector_interference_amplitude(
    vs: VanStockumInterior, r: float, n_loops: int = 1,
) -> float:
    """cos(N * phi_AB(r)) for a particle traversing N loops."""
    phase = aharonov_bohm_phase(vs, r)
    return float(math.cos(n_loops * phase))


def novelty_scan(n_radii: int = 30) -> dict:
    """Catcher on AB-phase profile across radii."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05 * vs.R, 10 * vs.R, n_radii)
    def fn(r: float) -> np.ndarray:
        phase = aharonov_bohm_phase(vs, r)
        return np.array([math.cos(phase), math.sin(phase),
                         enclosed_flux(vs, r)])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
