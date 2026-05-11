"""Anyonic CTC encoding: topological order threading chronology bands.

Suppose CTC bands of the LP supercritical exterior support topological
order (gapped phases with anyonic excitations). Each band carries
quasiparticles obeying braid statistics with a phase factor e^(i theta)
where theta is the anyonic phase. Successive bands at radii r_n form a
ring-like topological hierarchy.

Notable special cases:
- theta = 0: bosonic CTC quasiparticles
- theta = pi: fermionic
- theta = 2 pi / n: Z_n anyons (n=3 connects to Phase 35 Z_3 cover)
- theta = pi / k: non-abelian / Fibonacci anyons (k = golden ratio)

Module:
- braid_phase_at_band: e^(i theta_n) per band
- fusion_rules: how anyons in adjacent bands combine
- topological_entanglement_entropy: log(D) where D = sum d_a^2
- fibonacci_anyon_dimension: golden ratio appears here
- pair_extinction_topological_order: extinction kills the order
- novelty_scan: catcher on the fusion-rule structure
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior

PHI_GOLDEN = (1 + math.sqrt(5)) / 2  # Fibonacci/golden ratio


@dataclass(frozen=True)
class AnyonicBand:
    band_index: int
    braid_phase: float
    quantum_dimension: float


def braid_phase_at_band(band_index: int, alpha: float) -> float:
    """theta_n = 2 pi n / alpha (heuristic: LP-derived anyonic phase)."""
    return float(2 * math.pi * band_index / alpha)


def quantum_dimension_for_band(band_index: int, alpha: float,
                                  fibonacci: bool = False) -> float:
    """Quantum dimension d_a.

    For Z_n abelian anyons, d_a = 1.
    For Fibonacci anyons, d_tau = phi (golden ratio).
    Default: abelian Z_n unless fibonacci=True.
    """
    if fibonacci:
        return PHI_GOLDEN
    return 1.0


def topological_entanglement_entropy(
    band_indices: list[int], alpha: float,
    fibonacci: bool = False,
) -> float:
    """S_topo = log(D) where D = sqrt(sum d_a^2) over anyon types."""
    d_vals = [quantum_dimension_for_band(n, alpha, fibonacci)
              for n in band_indices]
    D_squared = sum(d ** 2 for d in d_vals)
    if D_squared <= 0:
        return 0.0
    return float(math.log(math.sqrt(D_squared)))


def fibonacci_anyon_dimension() -> float:
    """d_tau = (1 + sqrt(5)) / 2 = golden ratio."""
    return PHI_GOLDEN


def fusion_rules(theta_1: float, theta_2: float) -> dict:
    """For abelian anyons: theta_12 = theta_1 + theta_2 (mod 2 pi)."""
    theta_combined = (theta_1 + theta_2) % (2 * math.pi)
    return {
        "theta_1": theta_1,
        "theta_2": theta_2,
        "theta_combined": float(theta_combined),
        "is_bosonic": bool(abs(theta_combined) < 1e-9),
        "is_fermionic": bool(abs(theta_combined - math.pi) < 1e-9),
    }


def pair_extinction_topological_order(
    band_index: int, alpha: float, delta: float,
) -> dict:
    """At delta=pi, all braid phases collapse to 0 (no anyon order)."""
    base = braid_phase_at_band(band_index, alpha)
    extinction = 0.5 * (1.0 + math.cos(delta))
    return {
        "band_index": band_index,
        "delta": delta,
        "extinction_factor": extinction,
        "theta_single": float(base),
        "theta_pair": float(base * extinction),
        "order_destroyed_at_pi": bool(abs(delta - math.pi) < 1e-9),
    }


def all_band_anyon_data(vs: VanStockumInterior, n_bands: int = 5,
                         fibonacci: bool = False) -> list[AnyonicBand]:
    """One AnyonicBand entry per CTC band."""
    if not vs.is_supercritical():
        return []
    alpha = vs.alpha
    out = []
    for n in range(1, n_bands + 1):
        out.append(AnyonicBand(
            band_index=n,
            braid_phase=braid_phase_at_band(n, alpha),
            quantum_dimension=quantum_dimension_for_band(n, alpha, fibonacci),
        ))
    return out


def novelty_scan(n_band_max: int = 10) -> dict:
    """Catcher: braid-phase spectrum across band index."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return {"verdict": "subcritical", "sharp_features": []}
    alpha = vs.alpha
    band_indices = np.arange(1, n_band_max + 1).astype(float)
    def fn(band_float: float) -> np.ndarray:
        band = max(int(round(band_float)), 1)
        theta = braid_phase_at_band(band, alpha)
        return np.array([math.cos(theta), math.sin(theta), theta])
    result = scan_novelty(band_indices, fn, n_bits=32, parameter_label="band_index")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
