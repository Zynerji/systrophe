"""Tipler-Krasnikov composite: a Krasnikov tube threaded through the LP
exterior of a Tipler cylinder.

Concept
-------
A Tipler cylinder pre-tilts the light cones in an annular region around
its rotation axis (the supercritical LP exterior). A Krasnikov tube
adds an engineered second tilt along a specified worldline.  When the
tube's worldline lies inside the LP CTC band, the Tipler tilt
supplies a large fraction of the tilt that the Krasnikov tube would
otherwise need to engineer from scratch.

Predicted shortcut
------------------
Net exotic-matter requirement of the tube is reduced by the Tipler
envelope amplitude::

    |T_kk^hybrid| = max(|T_kk^krasnikov| - |Tipler_tilt_at(r)|, 0)

so the hybrid drops the integrated NEC violation in proportion to
the local LP envelope. At the deepest part of the first CTC band
(maximum |L(r)|), the reduction can be order-of-magnitude.

This module
-----------
- ``tipler_tilt_at`` -- a non-negative scalar measuring local
  Tipler-induced light-cone tilt at radius r.
- ``hybrid_NEC_radial`` -- the composite Krasnikov-style wall NEC
  with the Tipler subtraction baked in.
- ``hybrid_total_negative_energy`` -- volumetric integral of the
  composite NEC.
- ``novelty_scan`` -- catcher over (r, alpha) sweep, looking for the
  sharp Hamming transition where the tube worldline crosses into the
  Tipler CTC band (geometric onset of the shortcut).

Falsifiers / limits
-------------------
- The hybrid is **linearised**: we add the two metric perturbations
  without modelling their cross-product (formally O(G^2)).
- The Krasnikov tube is taken in its 1+1D form; the embedding into
  the LP exterior at radius r treats the radial direction as the
  effective Krasnikov "x" axis.
- The Tipler exterior is the Bonnor Case III analytic form
  (supercritical), so a > 1/2 is required for non-trivial tilt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.geometry.krasnikov_tube import (
    krasnikov_NEC_radial,
    krasnikov_energy_density,
    krasnikov_kernel,
)
from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior


def tipler_tilt_at(
    vs: VanStockumInterior, r: float,
) -> float:
    """Non-negative scalar measuring local Tipler-induced light-cone
    tilt at radius r.

    Defined as the magnitude of the local g_tphi component relative
    to the proper-time Coordinate g_tt. For supercritical a > 1/2,
    this oscillates with the log-periodic envelope and crosses zero
    at the CTC band boundaries.
    """
    if r <= vs.R:
        return 0.0  # interior; no exterior tilt
    K = float(vs.analytic_exterior_K(r))
    F = float(vs.analytic_exterior_F(r))
    if not math.isfinite(F) or abs(F) < 1e-15:
        return 0.0
    return abs(K) / max(abs(F), 1e-15)


def hybrid_NEC_radial(
    vs: VanStockumInterior,
    x: float, t: float, r: float,
    x_0: float = 0.0, t_0: float = 0.0, alpha: float = 4.0,
    coupling: float = 1.0,
) -> float:
    """Composite Krasnikov+Tipler NEC at radius r and tube coord x.

    The Krasnikov wall contribution is reduced (in magnitude) by the
    Tipler-tilt amplitude. `coupling` is a dimensionless scale that
    expresses how much of the geometric tilt is co-opted by the
    engineered tube (1.0 = full co-opt, 0 = no co-opt = pure Krasnikov).
    """
    nec_kras = krasnikov_NEC_radial(x, t, x_0, t_0, alpha)  # negative
    tilt = tipler_tilt_at(vs, r)  # non-negative
    # Reduction in magnitude:
    reduced_mag = max(abs(nec_kras) - coupling * tilt * abs(nec_kras), 0.0)
    return -reduced_mag  # keep sign convention


def hybrid_total_negative_energy(
    vs: VanStockumInterior,
    r: float, alpha: float = 4.0, coupling: float = 1.0,
    x_range: tuple[float, float] = (-3.0, 3.0), n_x: int = 121,
    t: float = 1.0,
) -> float:
    """Integrate the composite NEC over x (transverse wall coord)."""
    xs = np.linspace(*x_range, n_x)
    vals = [hybrid_NEC_radial(vs, float(x), t, r,
                                alpha=alpha, coupling=coupling)
            for x in xs]
    return float(np.trapezoid(np.array(vals), xs))


@dataclass(frozen=True)
class HybridSweep:
    r_grid: np.ndarray
    alpha_grid: np.ndarray
    NEC_min: np.ndarray
    E_neg_total: np.ndarray
    novelty_verdict: str


def novelty_scan(
    vs: VanStockumInterior | None = None,
    r_range: tuple[float, float] = (1.05, 12.0), n_r: int = 40,
    alpha_range: tuple[float, float] = (1.0, 12.0), n_alpha: int = 12,
    coupling: float = 1.0,
) -> dict:
    """Sweep (r, alpha) and run the catcher on the integrated
    hybrid NEC as a function of r at the median alpha.

    Predicted: a sharp Hamming transition at each radius r where the
    Tipler envelope crosses through a CTC band boundary, because
    the geometric tilt switches sign locally and the Krasnikov-tube
    subtraction discontinuously changes.
    """
    if vs is None:
        vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(*r_range, n_r)
    alpha_grid = np.linspace(*alpha_range, n_alpha)
    E_neg = np.zeros((n_r, n_alpha))
    for j, a in enumerate(alpha_grid):
        for i, r in enumerate(r_grid):
            E_neg[i, j] = hybrid_total_negative_energy(
                vs, float(r), alpha=float(a), coupling=coupling,
            )
    # Return the full alpha-column of E_neg(r) per radius so the
    # catcher's per-radius address has rich bit-occupancy (one bit
    # per alpha bin) and Hamming distances can resolve the CTC-band
    # ON / OFF transitions, not just shifts in a single scalar.
    def fn(rv):
        idx = int(np.argmin(np.abs(r_grid - rv)))
        return E_neg[idx]
    result = scan_novelty(r_grid, fn, n_bits=32)
    return {
        "r_grid": r_grid.tolist(),
        "alpha_grid": alpha_grid.tolist(),
        "E_neg_grid": E_neg.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
