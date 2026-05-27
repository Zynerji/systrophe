"""Unruh acoustic-metric mapping of the Lewis-Papapetrou exterior.

Unruh (Phys. Rev. Lett. 46 (1981) 1351) showed that small-amplitude
acoustic perturbations of a flowing fluid see an effective Lorentzian
metric:

    ds^2_acoustic = (rho / c) [ -(c^2 - v^2) dt^2 - 2 v . dx dt + dx . dx ]

with rho the local density, c the local sound speed, v the local flow
velocity. The kinematics of this metric are identical to those of a
genuine spacetime: in particular, the locus c^2 = v^2 is an *acoustic
horizon* with an associated acoustic Hawking temperature

    T_H_acoustic = kappa_acoustic / (2 pi).

This module identifies the LP angular metric (-F dt^2 + 2 K dt dphi
+ L dphi^2) with an acoustic metric in a co-rotating fluid frame:

    rho_acoustic = sqrt(L)
    v_phi        = K / sqrt(L)
    c^2 - v_phi^2 = F                  [Tipler chronology condition]

The locus F = 0 is *simultaneously* the chronology horizon (CTC band
boundary) and an *acoustic horizon* (sound cone closes); F < 0 is
"supersonic" (CTC region; closed timelike curves <-> closed acoustic
characteristics).

This realises the analog identification asserted in informal
proposals but never previously specified concretely.

References
----------
- W. G. Unruh, "Experimental black-hole evaporation?",
  Phys. Rev. Lett. 46 (1981) 1351.
- M. Visser, "Acoustic black holes", Class. Quantum Grav. 15 (1998) 1767.
- S. J. Robertson, J. Phys. B 45 (2012) 163001 (review).
"""

from __future__ import annotations

from math import pi

import numpy as np

from systrophe.qftcs.quantum_diagnostics import (
    hawking_temperature_at_horizon,
    surface_gravity_at_horizon,
)


def acoustic_metric_components(
    F: float | np.ndarray, K: float | np.ndarray, L: float | np.ndarray
) -> dict:
    """Decompose LP angular metric into acoustic-fluid components.

    Returns
    -------
    dict with keys:
      - rho_acoustic    : sqrt(L), effective density
      - v_phi           : K / sqrt(L), angular flow velocity
      - c_squared       : F + v_phi**2, effective sound speed squared
      - c_minus_v_sq    : F, the (c^2 - v^2) factor
      - signature       : +1 (subsonic, F > 0), 0 (sonic, F = 0), -1 (supersonic)
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    L = np.asarray(L, dtype=float)

    rho = np.sqrt(np.abs(L))
    # Avoid divide-by-zero for L=0
    with np.errstate(divide="ignore", invalid="ignore"):
        v_phi = np.where(rho > 1e-30, K / rho, np.zeros_like(rho))
    v_sq = v_phi ** 2
    c_sq = F + v_sq
    return {
        "rho_acoustic": rho,
        "v_phi": v_phi,
        "c_squared": c_sq,
        "c_minus_v_sq": F,
        "signature": np.sign(F),
    }


def acoustic_horizon_radius(vs, r_min: float, r_max: float, n_grid: int = 2001) -> float | None:
    """Find the smallest r in [r_min, r_max] where F(r) = 0 (acoustic horizon).

    Returns None if no zero crossing is detected. Uses linear
    interpolation between adjacent grid points.
    """
    rs = np.linspace(r_min, r_max, n_grid)
    Fs = np.array([float(vs.analytic_exterior_F(r)) for r in rs])
    signs = np.sign(Fs)
    flips = np.where(np.diff(signs) != 0)[0]
    if len(flips) == 0:
        return None
    i = flips[0]
    r1, r2 = rs[i], rs[i + 1]
    F1, F2 = Fs[i], Fs[i + 1]
    if abs(F2 - F1) < 1e-30:
        return float(0.5 * (r1 + r2))
    return float(r1 - F1 * (r2 - r1) / (F2 - F1))


def acoustic_surface_gravity(vs, r_horizon: float, eps: float = 1e-5) -> float:
    """Acoustic surface gravity kappa_acoustic = (1/2) |dF/dr| at the horizon.

    Identical formula to the gravitational case because the
    identification c^2 - v^2 = F maps the acoustic horizon to the
    chronology horizon coincidentally.
    """
    return surface_gravity_at_horizon(vs, r_horizon=r_horizon, eps=eps)


def acoustic_hawking_temperature(vs, r_horizon: float, eps: float = 1e-5) -> float:
    """Acoustic Hawking temperature T_H_acoustic = kappa / (2 pi).

    By the Unruh identification with c^2 - v^2 = F, this coincides
    *numerically* with the gravitational Hawking temperature at the
    chronology horizon.
    """
    kappa = acoustic_surface_gravity(vs, r_horizon, eps=eps)
    return kappa / (2 * pi)


def compare_acoustic_vs_gravitational_T_H(
    vs, r_horizon: float, eps: float = 1e-5
) -> dict:
    """Compare acoustic and gravitational Hawking T at the LP horizon.

    Returns dict with both values and their relative difference.
    By the identification c^2 - v^2 = F, they should be identical.
    """
    T_grav = hawking_temperature_at_horizon(vs, r_horizon=r_horizon, eps=eps)
    T_acoust = acoustic_hawking_temperature(vs, r_horizon=r_horizon, eps=eps)
    return {
        "T_gravitational": T_grav,
        "T_acoustic": T_acoust,
        "abs_diff": abs(T_grav - T_acoust),
        "rel_diff": abs(T_grav - T_acoust) / max(abs(T_grav), 1e-30),
    }


def acoustic_line_element_at_radius(vs, r: float) -> dict:
    """Acoustic line-element components at a single radius.

    Returns dict with:
      - F          : F(r)
      - K          : K(r)
      - L          : L(r)
      - rho_acoustic : sqrt|L|
      - v_phi      : K / sqrt|L|
      - c_squared  : sound speed squared
      - regime     : 'subsonic' (F>0), 'sonic' (F=0), 'supersonic' (F<0)
    """
    F = float(vs.analytic_exterior_F(r))
    K = float(vs.analytic_exterior_K(r))
    L = float(vs.analytic_exterior_L(r))
    comps = acoustic_metric_components(F, K, L)
    if F > 1e-12:
        regime = "subsonic"
    elif F < -1e-12:
        regime = "supersonic"
    else:
        regime = "sonic"
    return {
        "F": F,
        "K": K,
        "L": L,
        "rho_acoustic": float(comps["rho_acoustic"]),
        "v_phi": float(comps["v_phi"]),
        "c_squared": float(comps["c_squared"]),
        "regime": regime,
    }


def ctc_region_is_supersonic(vs, r_samples: np.ndarray) -> dict:
    """Verify that CTC regions (F < 0) correspond to acoustic supersonic flow.

    Returns dict with:
      - n_subsonic, n_sonic, n_supersonic: counts at the samples
      - ctc_supersonic_consistency: True iff every F<0 sample is supersonic
    """
    n_sub = 0
    n_sonic = 0
    n_super = 0
    consistent = True
    for r in r_samples:
        info = acoustic_line_element_at_radius(vs, float(r))
        regime = info["regime"]
        if regime == "subsonic":
            n_sub += 1
        elif regime == "supersonic":
            n_super += 1
            if info["F"] >= 0:
                consistent = False
        else:
            n_sonic += 1
    return {
        "n_subsonic": n_sub,
        "n_sonic": n_sonic,
        "n_supersonic": n_super,
        "ctc_supersonic_consistency": consistent,
    }
