"""Null geodesics in stationary axisymmetric spacetimes.

For a metric of Lewis-Papapetrou form

    ds^2 = -F(r) dt^2 + 2 K(r) dt dphi + L(r) dphi^2 + h(r)(dr^2 + dz^2),

the line element along a circular phi-orbit at fixed (r, z) is
    ds^2 = (-F + 2 K Omega + L Omega^2) dt^2.
A *null* (lightlike) circular orbit requires this to vanish:
    L Omega^2 + 2 K Omega - F = 0,
giving the two roots
    Omega_+- = (-K +- r) / L,
where K^2 + LF = r^2 (the canonical Weyl constraint) was used.

These are the boundary of the timelike-orbit Omega-sector;
inside the boundary the orbit is timelike, outside it is spacelike.
For a particle to remain on a *null* circular orbit it must follow
exactly one of these two Omega values.

Photon orbits are interesting because:
- The two roots correspond to "prograde" and "retrograde" photon
  trajectories.
- In the Tipler / Systrophe supercritical exterior, both K and L
  oscillate log-periodically, so Omega_+- oscillate too. The
  photon-orbit structure inherits the log-periodic Tipler signature.
- In the Goedel universe and Gott pair, the roots have closed forms
  in their respective coordinate systems.

For *non-circular* null geodesics (radial motion), the radial equation
is the kappa = 0 case of the timelike geodesic in `geodesic.py`:
    h (r')^2 = - V_eff(r),
    V_eff(r) = (F l^2 - 2 K E l - L E^2) / r^2,
turning at V_eff(r) = 0, i.e., where the impact parameter b = l/E satisfies
    F b^2 - 2 K b - L = 0,
which has solutions  b_+- = (K +- r) / F.

This module exposes:
- null_circular_omega(F, K, L, r): the two prograde / retrograde Omega values.
- null_impact_parameters(F, K, r): the two photon impact parameters.
- integrate_null_geodesic: thin wrapper around integrate_geodesic with
  kappa = 0 (null) for radial photon trajectories.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .geodesic import integrate_geodesic


def null_circular_omega(F: float, K: float, L: float, r: float) -> tuple[float, float]:
    """Return (Omega_-, Omega_+) for null circular phi-orbits at radius r.

    Solves L Omega^2 + 2 K Omega - F = 0 using K^2 + L F = r^2:
        Omega_+- = (-K +- r) / L.
    Degenerate at L = 0 (the boundary of a CTC band).
    """
    if abs(L) < 1e-300:
        raise ValueError("null circular omega is degenerate at L = 0")
    om_minus = (-K - r) / L
    om_plus = (-K + r) / L
    return (min(om_minus, om_plus), max(om_minus, om_plus))


def null_impact_parameters(F: float, K: float, r: float) -> tuple[float, float]:
    """Return (b_-, b_+) for photon turning radii at given F, K, r.

    For a photon with conserved E and l, the radial turning point is at
    F b^2 - 2 K b - L = 0 (where L = (r^2 - K^2)/F), giving
        b_+- = (K +- r) / F.
    Degenerate at F = 0 (the ergosurface).
    """
    if abs(F) < 1e-300:
        raise ValueError("null impact parameter is degenerate at F = 0 (ergosurface)")
    b_minus = (K - r) / F
    b_plus = (K + r) / F
    return (min(b_minus, b_plus), max(b_minus, b_plus))


def integrate_null_geodesic(
    F_fn: Callable[[float], float],
    K_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    h_fn: Callable[[float], float],
    E: float,
    ell: float,
    r0: float,
    rdot0: float,
    affine_max: float,
    n_samples: int = 2001,
) -> dict:
    """Integrate a planar (z = const) null geodesic.

    Wrapper around `integrate_geodesic` with kappa = 0. Affine parameter
    plays the role of proper time for null geodesics.
    """
    return integrate_geodesic(
        F_fn=F_fn,
        K_fn=K_fn,
        L_fn=L_fn,
        h_fn=h_fn,
        E=E,
        ell=ell,
        r0=r0,
        rdot0=rdot0,
        tau_max=affine_max,
        n_samples=n_samples,
        kappa=0.0,
    )


def vanstockum_photon_omega(vs, r: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Photon circular Omega_+- at exterior radius r for a single van Stockum cylinder.

    Returns two arrays (Omega_-, Omega_+) each shaped like r. Uses the
    analytic Case III closed forms; valid for the supercritical exterior.
    Raises ValueError on subcritical inputs.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    if not vs.is_supercritical():
        raise ValueError("photon-orbit closed form requires supercritical (a > 1/2)")
    r_arr = np.asarray(r, dtype=float)
    F = vs.analytic_exterior_F(r_arr)
    K = vs.analytic_exterior_K(r_arr)
    L = vs.analytic_exterior_L(r_arr)
    # Element-wise division; near L=0 yields large values (photon orbits at CTC boundary
    # become unbounded in coordinate Omega).
    om_a = (-K - r_arr) / L
    om_b = (-K + r_arr) / L
    om_minus = np.minimum(om_a, om_b)
    om_plus = np.maximum(om_a, om_b)
    return om_minus, om_plus
