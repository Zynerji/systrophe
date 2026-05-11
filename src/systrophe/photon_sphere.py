"""Photon spheres on the Lewis-Papapetrou rotating-cylinder exterior.

A *photon sphere* is a locus where photons orbit in circles. For a
stationary axisymmetric metric in Weyl coordinates, the photon
sphere is determined by the stationarity of the impact parameter:

    db(r) / dr = 0,   where   b(r) = (K(r) +- r) / F(r).

Result (numerical, verified across subcritical / critical /
supercritical regimes): **The bare Lewis-Papapetrou exterior has
no photon spheres.** db(r)/dr > 0 monotonically for all r outside
the source cylinder, in every rotation regime tested. The
impact parameter diverges at chronology horizons (F = 0) but
does not have local extrema between them.

Why: in canonical Weyl coordinates with K^2 + LF = r^2, the
impact parameter is b(r) = (K +- r) / F. Even when F and K
oscillate log-periodically, the geometric pole at F = 0 dominates
over the oscillation, giving a monotonic b(r) in each connected
domain of F > 0.

**Creating a photon sphere** therefore requires perturbing the
spacetime --- e.g., adding a Schwarzschild-like compact object at
the cylinder axis. This module provides the perturbation framework:

- `bare_lp_has_photon_sphere(vs)`: quick no-go check (returns False
  for any standard van Stockum cylinder)
- `effective_b_hybrid(vs, r, M)`: photon impact parameter on a
  hybrid Schwarzschild-cylinder spacetime
- `engineer_photon_sphere_via_mass(vs, r_target)`: solve for the
  Schwarzschild mass M needed to place a photon sphere at r_target
- `hybrid_photon_sphere_radii(vs, M, r_min, r_max)`: all photon
  spheres of the hybrid geometry
- `photon_sphere_descriptor(vs, r_ph, M, branch)`: full descriptor

For the unperturbed cylinder case, see also `photon_orbits.py`,
which exposes the photon `null_circular_omega` and
`null_impact_parameters` functions.

References
----------
- N. Cornish, J. J. Levin, ``Lyapunov exponents and quasinormal
  modes,'' Class. Quantum Grav. 20 (2003) 1649.
- C. Bambi, ``Shadow of black holes,'' Phys. Rev. D 87 (2013) 107501.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class PhotonSphere:
    """Photon-sphere descriptor."""

    r: float
    branch: str  # "prograde" or "retrograde"
    omega: float  # orbital angular velocity
    impact_parameter: float
    is_stable: bool  # False = unstable saddle (Schwarzschild-like)
    schwarzschild_mass: float = 0.0  # 0 = bare cylinder; >0 = hybrid


# ----- Bare LP analysis (no-go result) -----------------------------------

def impact_parameter_bare(vs, r, branch="prograde"):
    """Photon impact parameter b(r) = (K +/- r) / F on bare LP exterior."""
    F = float(vs.analytic_exterior_F(r))
    K = float(vs.analytic_exterior_K(r))
    if abs(F) < 1e-30:
        return float("nan")
    if branch == "prograde":
        return (K + r) / F
    elif branch == "retrograde":
        return (K - r) / F
    else:
        raise ValueError("branch must be 'prograde' or 'retrograde'")


def _db_dr(vs, r, branch="prograde", eps=1e-5):
    """db/dr via central differences (bare LP)."""
    b_plus = impact_parameter_bare(vs, r + eps, branch)
    b_minus = impact_parameter_bare(vs, r - eps, branch)
    return (b_plus - b_minus) / (2 * eps)


def bare_lp_has_photon_sphere(
    vs, r_min: float = 1.05, r_max: float = 50.0, n_grid: int = 1001,
    branch: str = "prograde",
) -> bool:
    """Test whether the bare LP exterior admits a photon sphere.

    Empirically: returns False for any van Stockum cylinder in any
    regime. Provided as a check function rather than asserting the
    no-go result.
    """
    rs = np.linspace(r_min, r_max, n_grid)
    derivs = []
    for r in rs:
        try:
            d = _db_dr(vs, float(r), branch)
            if np.isfinite(d):
                derivs.append(d)
        except Exception:
            pass
    if len(derivs) < 5:
        return False
    derivs = np.array(derivs)
    signs = np.sign(derivs)
    return bool(np.any(np.diff(signs) != 0))


# ----- Hybrid: Schwarzschild mass + LP cylinder --------------------------

def hybrid_F(vs, r, M):
    """F(r) of the hybrid Schwarzschild + LP cylinder spacetime.

    Approximate construction: take the LP F and modulate by the
    Schwarzschild factor (1 - 2M/r) at large r.
    """
    if r <= 2 * M:
        return float("nan")  # inside Schwarzschild horizon
    F_lp = float(vs.analytic_exterior_F(r))
    schwarzschild_factor = 1 - 2 * M / r
    return F_lp * schwarzschild_factor


def hybrid_K(vs, r, M):
    """K(r) of the hybrid. Add no direct Schwarzschild contribution;
    K is the cylinder twist."""
    return float(vs.analytic_exterior_K(r))


def effective_b_hybrid(vs, r, M, branch="prograde"):
    """Impact parameter b(r) of the hybrid Schwarzschild-cylinder spacetime."""
    F = hybrid_F(vs, r, M)
    K = hybrid_K(vs, r, M)
    if not np.isfinite(F) or abs(F) < 1e-30:
        return float("nan")
    if branch == "prograde":
        return (K + r) / F
    elif branch == "retrograde":
        return (K - r) / F
    else:
        raise ValueError("branch must be 'prograde' or 'retrograde'")


def _db_dr_hybrid(vs, r, M, branch="prograde", eps=1e-5):
    b_plus = effective_b_hybrid(vs, r + eps, M, branch)
    b_minus = effective_b_hybrid(vs, r - eps, M, branch)
    return (b_plus - b_minus) / (2 * eps)


def _d2b_dr2_hybrid(vs, r, M, branch="prograde", eps=1e-4):
    b_plus = effective_b_hybrid(vs, r + eps, M, branch)
    b_0 = effective_b_hybrid(vs, r, M, branch)
    b_minus = effective_b_hybrid(vs, r - eps, M, branch)
    return (b_plus - 2 * b_0 + b_minus) / (eps ** 2)


def hybrid_photon_sphere_radii(
    vs, M: float, r_min: float = 1.05, r_max: float = 50.0,
    n_grid: int = 2001, branch: str = "prograde",
) -> list[float]:
    """Find photon-sphere radii on the hybrid Schwarzschild-LP exterior.

    Standard Schwarzschild photon sphere is at r = 3M; the cylinder
    distorts this away from the bare 3M.
    """
    rs = np.linspace(max(r_min, 2 * M + 0.01), r_max, n_grid)
    derivs = []
    for r in rs:
        try:
            d = _db_dr_hybrid(vs, float(r), M, branch)
            if not np.isfinite(d):
                d = float("nan")
            derivs.append(d)
        except Exception:
            derivs.append(float("nan"))
    derivs = np.array(derivs)
    candidates = []
    for i in range(len(rs) - 1):
        if not (np.isfinite(derivs[i]) and np.isfinite(derivs[i + 1])):
            continue
        if derivs[i] * derivs[i + 1] < 0:
            try:
                root = brentq(
                    lambda r: _db_dr_hybrid(vs, r, M, branch),
                    rs[i], rs[i + 1], maxiter=100,
                )
                candidates.append(float(root))
            except Exception:
                pass
    return candidates


def hybrid_photon_sphere_stability(vs, r_ph: float, M: float,
                                       branch: str = "prograde") -> bool:
    """Stable (True) iff d²b/dr² > 0 at r_ph in the hybrid spacetime."""
    d2b = _d2b_dr2_hybrid(vs, r_ph, M, branch)
    return bool(d2b > 0)


def hybrid_photon_sphere_omega(vs, r_ph: float, M: float,
                                  branch: str = "prograde") -> float:
    """Photon angular velocity at the hybrid photon sphere."""
    F = hybrid_F(vs, r_ph, M)
    K = hybrid_K(vs, r_ph, M)
    L = float(vs.analytic_exterior_L(r_ph))
    if abs(L) < 1e-30:
        return float("nan")
    om_minus = (-K - r_ph) / L
    om_plus = (-K + r_ph) / L
    return om_plus if branch == "prograde" else om_minus


def photon_sphere_descriptor(
    vs, r_ph: float, M: float = 0.0, branch: str = "prograde",
) -> PhotonSphere:
    """Build PhotonSphere descriptor for hybrid (M>0) or bare (M=0) LP."""
    if M > 0:
        omega = hybrid_photon_sphere_omega(vs, r_ph, M, branch)
        b = effective_b_hybrid(vs, r_ph, M, branch)
        stable = hybrid_photon_sphere_stability(vs, r_ph, M, branch)
    else:
        b = impact_parameter_bare(vs, r_ph, branch)
        # Bare LP: no photon spheres expected, but if r_ph is provided,
        # compute omega from null_circular_omega
        from .photon_orbits import null_circular_omega
        F = float(vs.analytic_exterior_F(r_ph))
        K = float(vs.analytic_exterior_K(r_ph))
        L = float(vs.analytic_exterior_L(r_ph))
        try:
            om_minus, om_plus = null_circular_omega(F, K, L, r_ph)
            omega = om_plus if branch == "prograde" else om_minus
        except ValueError:
            omega = float("nan")
        stable = False  # bare LP has no true photon spheres
    return PhotonSphere(
        r=r_ph, branch=branch, omega=omega, impact_parameter=b,
        is_stable=stable, schwarzschild_mass=M,
    )


def engineer_photon_sphere_via_mass(
    vs, r_target: float, branch: str = "prograde",
    M_range: tuple[float, float] = (0.01, 5.0),
) -> float | None:
    """Solve for Schwarzschild mass M needed to place a photon sphere at r_target.

    Brent-search for M in `M_range` such that db_hybrid/dr(r_target) = 0.
    Returns M, or None if no such M exists in range.
    """
    def residual(M):
        try:
            return _db_dr_hybrid(vs, r_target, M, branch)
        except Exception:
            return float("inf")
    Ms = np.linspace(*M_range, 30)
    residuals = [residual(M) for M in Ms]
    for i in range(len(Ms) - 1):
        if np.isfinite(residuals[i]) and np.isfinite(residuals[i + 1]):
            if residuals[i] * residuals[i + 1] < 0:
                try:
                    M_sol = brentq(residual, Ms[i], Ms[i + 1], maxiter=100)
                    return float(M_sol)
                except Exception:
                    pass
    return None


def photon_sphere_shadow_radius(
    ph: PhotonSphere, observer_r: float = float("inf")
) -> float:
    """Apparent shadow radius at observer_r >> r_ph (small-angle limit)."""
    if observer_r == float("inf"):
        return abs(ph.impact_parameter)
    return abs(ph.impact_parameter) / observer_r


def schwarzschild_photon_sphere_radius(M: float) -> float:
    """Photon sphere radius for pure Schwarzschild: r = 3M."""
    return 3.0 * M
