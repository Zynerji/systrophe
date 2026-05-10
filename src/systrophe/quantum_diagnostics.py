"""Classical pre-quantum diagnostics related to chronology protection.

This module provides simple invariants that *would* appear in a full
semi-classical analysis of the Tipler / Systrophe spacetime, without
attempting the full QFT-in-curved-spacetime calculation.

The chronology-protection conjecture (Hawking 1992) predicts that
quantum back-reaction destabilises spacetimes containing CTCs by
diverging the renormalised stress-energy tensor at the Cauchy
horizon. The full calculation involves point-splitting renormalisation
of two-point functions for a free scalar field on the Tipler
background and is out of scope for this package.

What we provide
---------------
- `tolman_blueshift_factor(F)`: the Tolman blueshift of a locally static
  observer at $g_{tt} = -F$ relative to one with $F = 1$. Diverges as
  $1/\\sqrt{F}$ as $F \\to 0$ (ergosurface).
- `chronology_protection_indicator(F)`: a simple $1/F^2$ proxy that
  approximates the leading divergence of the renormalised stress-
  energy near the ergosurface (and hence near the inner CTC band edges
  in Bonnor Case III). This is a *qualitative* indicator only.
- `cauchy_horizon_estimate(vs)`: returns the radii of the Tipler
  exterior where the Tolman blueshift formally diverges (i.e. the
  $F$-zeros). These are candidate Cauchy horizons.

These quantities are entirely classical. A genuine chronology-protection
test requires the renormalised semi-classical stress-energy tensor;
this module exposes the *classical* invariants that gate that
calculation. References for the full programme: Hawking 1992;
Visser 1996, *Lorentzian Wormholes*, Ch. 14.
"""

from __future__ import annotations

import numpy as np


def tolman_blueshift_factor(F: float | np.ndarray) -> np.ndarray:
    """Tolman blueshift factor 1 / sqrt(F) for a locally static observer.

    Diverges as F -> 0 (ergosurface / Cauchy horizon candidate).
    Returns inf where F <= 0; the user may want to mask such points.
    """
    F_arr = np.asarray(F, dtype=float)
    out = np.where(F_arr > 0, 1.0 / np.sqrt(np.where(F_arr > 0, F_arr, 1.0)), np.inf)
    return out


def chronology_protection_indicator(F: float | np.ndarray) -> np.ndarray:
    """Heuristic 1 / F^2 indicator of the leading classical divergence.

    A rough proxy for the leading divergence of the renormalised
    stress-energy tensor near the ergosurface, motivated by the
    Tolman / Hawking blueshift scaling. Not a substitute for a real
    QFTCS calculation; use only as a qualitative chronology-protection
    flag.
    """
    F_arr = np.asarray(F, dtype=float)
    return np.where(np.abs(F_arr) > 1e-30, 1.0 / (F_arr * F_arr), np.inf)


def cauchy_horizon_estimate(vs) -> np.ndarray:
    """Radii of the Tipler exterior where the Tolman blueshift diverges (F = 0).

    For a supercritical van Stockum cylinder these are the closed-form
    F-zeros r_n = R * exp((n pi - gamma) / alpha) for n = 1, 2, ....
    The Cauchy horizon of the maximally-extended chronology-violating
    region is the *first* such surface (r_1).

    Returns the array of F-zero radii up to a default upper bound of 10 R.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    if not vs.is_supercritical():
        raise ValueError("Cauchy horizon estimate only defined in Case III (a > 1/2)")

    alpha = vs.alpha
    gamma = float(np.pi - np.arctan(alpha))
    zeros = []
    n = 1
    upper_u = float(np.log(10.0))  # default r_max = 10 R
    while True:
        u_n = (n * np.pi - gamma) / alpha
        if u_n > upper_u:
            break
        if u_n > 0:
            zeros.append(float(vs.R * np.exp(u_n)))
        n += 1
    return np.array(zeros, dtype=float)
