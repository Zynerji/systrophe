"""Black-hole pair production in the LP rotating background.

The Schwinger pair-production rate for charged particles in an
electric field is Gamma ~ exp(-pi m^2 / (eE)). For black-hole pair
production in a strong gravitational field, the analog rate uses the
Euclidean action of the instanton geometry mediating the transition.

In the LP supercritical exterior, regions of high curvature (near
chronology horizons) can act as "field-like" sources for BH-pair
production. The pair-production rate is enhanced near the CHs and
suppressed in the bulk.

Module:
- schwinger_analog_rate: Gamma ~ exp(-pi M^2 / |grad F|)
- production_locus: r where rate peaks
- pair_extinction_modification: how delta=pi suppresses rate
- novelty_scan: catcher on rate(r) curve
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .novelty_catcher import scan_novelty
from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class BHPairRate:
    r: float
    field_strength: float  # |grad F|(r)
    bh_mass_threshold: float  # M above which production is exponentially small
    production_rate: float


def field_strength_at_r(vs: VanStockumInterior, r: float) -> float:
    """|F'(r)|, the analog 'electric' field strength."""
    eps = 1e-4 * max(r, 1.0)
    F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    Fp = (F_plus - F_minus) / (2 * eps)
    return float(abs(Fp))


def schwinger_analog_rate(
    vs: VanStockumInterior, r: float, bh_mass: float = 1.0,
) -> BHPairRate:
    """Pair-production rate per unit volume at radius r."""
    E = field_strength_at_r(vs, r)
    if E < 1e-30:
        return BHPairRate(r=r, field_strength=E,
                          bh_mass_threshold=float("inf"),
                          production_rate=0.0)
    exponent = -math.pi * bh_mass ** 2 / E
    rate = math.exp(max(exponent, -700))
    M_threshold = float(math.sqrt(E / math.pi))
    return BHPairRate(
        r=r, field_strength=E,
        bh_mass_threshold=M_threshold,
        production_rate=float(rate),
    )


def production_locus(
    vs: VanStockumInterior, bh_mass: float = 0.5,
    r_min: float = None, r_max: float = None,
    n_samples: int = 500,
) -> dict:
    """Find r where Schwinger rate is maximal."""
    if r_min is None:
        r_min = vs.R * 1.01
    if r_max is None:
        r_max = vs.R * 20.0
    r_grid = np.linspace(r_min, r_max, n_samples)
    rates = []
    for r in r_grid:
        rate = schwinger_analog_rate(vs, float(r), bh_mass=bh_mass)
        rates.append(rate.production_rate)
    rates_arr = np.asarray(rates)
    idx = int(np.argmax(rates_arr))
    return {
        "r_max_rate": float(r_grid[idx]),
        "max_rate": float(rates_arr[idx]),
        "r_grid": r_grid,
        "rates": rates_arr,
    }


def pair_extinction_modification(
    vs: VanStockumInterior, delta: float,
    r: float, bh_mass: float = 0.5,
) -> dict:
    """For Systrophe pair, F-amplitude is scaled by (1+cos delta)/2.

    F' scales similarly -> Schwinger exponent grows by 1/extinction^2.
    At delta=pi, rate -> 0.
    """
    base = schwinger_analog_rate(vs, r, bh_mass=bh_mass)
    extinction = 0.5 * (1.0 + math.cos(delta))
    if extinction < 1e-30:
        rate_pair = 0.0
    else:
        E_pair = base.field_strength * extinction
        exponent = -math.pi * bh_mass ** 2 / E_pair
        rate_pair = math.exp(max(exponent, -700))
    return {
        "delta": delta,
        "extinction_factor": extinction,
        "rate_single": base.production_rate,
        "rate_pair": float(rate_pair),
        "suppression_factor": float(rate_pair / max(base.production_rate, 1e-30)),
    }


def novelty_scan(n_radii: int = 30) -> dict:
    """Catcher on the production-rate profile across r."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    if not vs.is_supercritical():
        return {"verdict": "subcritical", "sharp_features": []}
    r_grid = np.linspace(1.05 * vs.R, 10 * vs.R, n_radii)
    def fn(r: float) -> np.ndarray:
        rate = schwinger_analog_rate(vs, r)
        return np.array([math.log10(max(rate.production_rate, 1e-300)),
                         math.log10(max(rate.field_strength, 1e-30)),
                         rate.bh_mass_threshold])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
