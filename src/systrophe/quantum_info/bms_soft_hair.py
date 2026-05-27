"""BMS-like soft hair on the LP cylinder asymptotic boundary.

The Bondi-Metzner-Sachs (BMS) group of asymptotic symmetries at null
infinity of asymptotically-flat spacetimes contains supertranslations
and superrotations beyond the Poincaré subgroup. The associated "soft
hair" (Hawking-Perry-Strominger 2016) carries information that may
resolve the black-hole information paradox.

For the LP cylinder, the exterior is NOT asymptotically flat (the
metric oscillates log-periodically). The natural "asymptotic" boundary
is a 2D surface at finite r where F, K, L acquire stationary phase
relative to a fiducial reference. Soft hair on this boundary is
parameterized by log-periodic supertranslation modes of the form
   T(phi, z) = sum_n T_n cos(alpha ln(r/R) + phi_n)

This module:
- supertranslation_mode_amplitudes: T_n decomposition
- superrotation_charges: angular-momentum analogs at boundary
- soft_hair_information_content: bits stored in soft modes
- pair_extinction_soft_hair: extinction at delta=pi removes hair
- novelty_scan: applies the catcher to the soft-hair spectrum
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class SoftHairMode:
    n: int  # mode number
    amplitude: float
    phase: float
    log_periodic_frequency: float


def supertranslation_mode_amplitudes(
    vs: VanStockumInterior, n_modes: int = 5,
    r_boundary: float = 50.0,
) -> list[SoftHairMode]:
    """Decompose F(r) near r_boundary into supertranslation modes.

    For LP supercritical, F(r) ~ beta cos(alpha ln(r/R) + delta).
    Higher harmonics appear from non-linearities; we model them as
    cos(n alpha ln(r/R) + phi_n) with rapidly-decaying amplitudes.
    """
    if not vs.is_supercritical():
        return []
    out = []
    alpha = vs.alpha
    for n in range(1, n_modes + 1):
        amp = 1.0 / (n ** 2)  # heuristic decay
        phase = math.pi - math.atan(alpha) + n * math.pi / 4
        out.append(SoftHairMode(
            n=n, amplitude=amp, phase=phase,
            log_periodic_frequency=n * alpha,
        ))
    return out


def superrotation_charge(
    vs: VanStockumInterior, n: int = 1,
    cylinder_length: float = 1.0,
) -> float:
    """Q_n = integral T_{phi t} cos(n alpha ln r) over boundary."""
    if not vs.is_supercritical():
        return 0.0
    # Heuristic from K(r) = omega r^2 -like:
    a = vs.a
    return float(a * cylinder_length * math.pi / (n ** 2))


def soft_hair_information_content(
    vs: VanStockumInterior, n_modes: int = 10,
) -> dict:
    """Bits of information stored in n_modes supertranslation modes.

    For each mode, amplitude is in (0, 1) -> bit = log2(amplitude / floor).
    """
    if not vs.is_supercritical():
        return {"total_bits": 0.0, "n_modes": 0}
    modes = supertranslation_mode_amplitudes(vs, n_modes)
    floor = 1e-3
    bits_per_mode = []
    for m in modes:
        if m.amplitude > floor:
            bits_per_mode.append(math.log2(m.amplitude / floor))
    return {
        "total_bits": float(sum(bits_per_mode)),
        "bits_per_mode": [float(b) for b in bits_per_mode],
        "n_modes_used": len(bits_per_mode),
    }


def pair_extinction_soft_hair(
    vs: VanStockumInterior, delta: float,
    n_modes: int = 5,
) -> dict:
    """Extinction factor scales all soft-hair amplitudes by (1+cos delta)/2."""
    extinction = 0.5 * (1.0 + math.cos(delta))
    modes = supertranslation_mode_amplitudes(vs, n_modes)
    scaled_amps = [m.amplitude * extinction for m in modes]
    total_info = sum(math.log2(max(a, 1e-12) / 1e-3)
                     for a in scaled_amps if a > 1e-3)
    return {
        "delta": float(delta),
        "extinction_factor": float(extinction),
        "scaled_amplitudes": [float(a) for a in scaled_amps],
        "total_info_bits": float(total_info),
        "hair_removed_at_pi": bool(abs(delta - math.pi) < 1e-9),
    }


def novelty_scan(n_modes: int = 30) -> dict:
    """Address-space catcher on soft-hair mode amplitudes across n.

    Catches any sudden mode-amplitude jump that would indicate a
    discrete-scale-invariance breakdown.
    """
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return {"verdict": "subcritical", "sharp_features": []}
    def fn(n_float: float) -> np.ndarray:
        n = max(int(round(n_float)), 1)
        modes = supertranslation_mode_amplitudes(vs, n)
        return np.array([m.amplitude for m in modes])
    ns = np.linspace(1, n_modes, n_modes).astype(float)
    result = scan_novelty(ns, fn, n_bits=32, parameter_label="mode_count")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
