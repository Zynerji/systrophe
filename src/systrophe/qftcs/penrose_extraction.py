"""Penrose-like energy extraction from rotating-cylinder CTC bands.

In Kerr spacetime, the Penrose process extracts rotational energy
by sending a particle into the ergoregion, splitting it, and
sending the negative-energy fragment into the black hole while the
positive-energy fragment escapes with E_+ > E_initial.

For the Tipler / van Stockum cylinder spacetime, the supercritical
exterior contains *ergosurfaces* (loci F = 0 where ∂_t becomes
spacelike). Beyond each F = 0 surface, negative-energy timelike
orbits exist, enabling an analogous extraction process.

This module:

- Identifies all ergosurfaces (F = 0 zeros)
- Tests existence of negative-energy timelike orbits in each
  supercritical band
- Computes maximum Penrose efficiency
- Compares to the Kerr value (~21% for extreme Kerr)

References
----------
- R. Penrose, Rev. Nuovo Cim. 1 (1969) 252.
- J. M. Bardeen, W. H. Press, S. A. Teukolsky, Astrophys. J. 178
  (1972) 347.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from systrophe.geometry.geodesic import timelike_omega_bounds


@dataclass(frozen=True)
class PenroseDescriptor:
    """Penrose-extraction descriptor at a single ergosurface."""

    r_ergosurface: float  # F = 0 locus
    F: float
    K: float
    L: float
    energy_min: float  # most-negative E achievable on a timelike orbit
    energy_max: float  # most-positive E achievable
    efficiency: float  # |E_min| / E_max, the Penrose extraction ratio
    is_extractable: bool  # True iff energy_min < 0


def ergosurface_radii(vs, r_min: float = 1.05, r_max: float = 50.0,
                        n_grid: int = 2001) -> list[float]:
    """Find all F = 0 ergosurfaces in [r_min, r_max]."""
    rs = np.linspace(r_min, r_max, n_grid)
    Fs = np.array([float(vs.analytic_exterior_F(r)) for r in rs])
    signs = np.sign(Fs)
    flips = np.where(np.diff(signs) != 0)[0]
    horizons = []
    for i in flips:
        r1, r2 = rs[i], rs[i + 1]
        F1, F2 = Fs[i], Fs[i + 1]
        if abs(F2 - F1) < 1e-30:
            horizons.append(float(0.5 * (r1 + r2)))
        else:
            horizons.append(float(r1 - F1 * (r2 - r1) / (F2 - F1)))
    return horizons


def orbit_energy(F: float, K: float, L: float, Omega: float) -> float:
    """Killing energy of a circular orbit at angular velocity Omega.

    E = -p_t for a timelike geodesic with u^phi = Omega u^t.
    Normalising u_mu u^mu = -1 gives
        E = (F - K Omega) / sqrt(F - 2 K Omega - L Omega^2).
    """
    norm_sq = F - 2 * K * Omega - L * Omega ** 2
    if norm_sq <= 0:
        return float("nan")
    return float((F - K * Omega) / np.sqrt(norm_sq))


def negative_energy_in_band(
    vs, r: float, n_omega: int = 50,
) -> tuple[float, float]:
    """At radius r, find min and max Killing energy across timelike orbits."""
    F = float(vs.analytic_exterior_F(r))
    K = float(vs.analytic_exterior_K(r))
    L = float(vs.analytic_exterior_L(r))
    try:
        om_lo, om_hi = timelike_omega_bounds(F, K, L, r)
    except Exception:
        return (float("nan"), float("nan"))
    # Avoid the timelike-boundary itself (where energy diverges)
    eps = 0.02 * (om_hi - om_lo)
    omegas = np.linspace(om_lo + eps, om_hi - eps, n_omega)
    energies = [orbit_energy(F, K, L, float(om)) for om in omegas]
    energies = [e for e in energies if np.isfinite(e)]
    if not energies:
        return (float("nan"), float("nan"))
    return (float(min(energies)), float(max(energies)))


def penrose_efficiency_at(vs, r: float) -> dict:
    """Penrose extraction at radius r.

    For an extractable orbit (E_min < 0), the maximum energy gained
    by the escaping fragment is |E_min| relative to the initial energy
    E_max. Efficiency = |E_min| / E_max.
    """
    E_min, E_max = negative_energy_in_band(vs, r)
    if not np.isfinite(E_min) or not np.isfinite(E_max):
        return {"r": r, "E_min": E_min, "E_max": E_max,
                "efficiency": 0.0, "extractable": False}
    extractable = E_min < 0
    eff = abs(E_min) / E_max if E_max > 0 else 0.0
    return {
        "r": r, "E_min": E_min, "E_max": E_max,
        "efficiency": eff, "extractable": bool(extractable),
    }


def penrose_extraction_scan(
    vs, r_min: float = 1.05, r_max: float = 20.0, n_grid: int = 101,
) -> list[dict]:
    """Scan r and report Penrose-extraction at each point.

    Returns a list of per-r dicts with efficiency and extractability.
    """
    rs = np.linspace(r_min, r_max, n_grid)
    return [penrose_efficiency_at(vs, float(r)) for r in rs]


def maximum_penrose_efficiency(vs, r_max: float = 20.0,
                                  n_grid: int = 401) -> dict:
    """Find the maximum Penrose efficiency across the exterior."""
    scan = penrose_extraction_scan(vs, r_max=r_max, n_grid=n_grid)
    extractable = [s for s in scan if s["extractable"]]
    if not extractable:
        return {"max_efficiency": 0.0, "r_optimal": None,
                "kerr_reference": 0.21,  # extreme Kerr Penrose limit
                "fraction_of_kerr": 0.0,
                "n_extractable_radii": 0}
    best = max(extractable, key=lambda s: s["efficiency"])
    return {
        "max_efficiency": best["efficiency"],
        "r_optimal": best["r"],
        "kerr_reference": 0.21,
        "fraction_of_kerr": best["efficiency"] / 0.21,
        "n_extractable_radii": len(extractable),
    }


def ergosurface_descriptors(vs, r_max: float = 50.0) -> list[PenroseDescriptor]:
    """One descriptor per ergosurface, summarising Penrose potential."""
    horizons = ergosurface_radii(vs, r_max=r_max)
    out = []
    for r_h in horizons:
        F = float(vs.analytic_exterior_F(r_h))
        K = float(vs.analytic_exterior_K(r_h))
        L = float(vs.analytic_exterior_L(r_h))
        # Just inside the horizon (probe one side)
        r_test = r_h * 1.01
        E_min, E_max = negative_energy_in_band(vs, r_test)
        eff = abs(E_min) / E_max if (E_max > 0 and np.isfinite(E_min)) else 0.0
        out.append(PenroseDescriptor(
            r_ergosurface=r_h, F=F, K=K, L=L,
            energy_min=E_min, energy_max=E_max,
            efficiency=eff,
            is_extractable=bool(np.isfinite(E_min) and E_min < 0),
        ))
    return out


def compare_to_kerr() -> dict:
    """Reference values for Kerr Penrose efficiency.

    For extreme Kerr (a/M = 1), the maximum energy extractable via
    Penrose process is about 21% of the total mass-energy.
    """
    return {
        "kerr_extreme_efficiency": 0.21,
        "kerr_extreme_mass_efficiency": 1 - 1 / np.sqrt(2),
        "comment": ("For extreme Kerr (a/M = 1), the maximum energy "
                    "extractable via Penrose process is "
                    "1 - 1/sqrt(2) ~= 29.3% of irreducible mass."),
    }
