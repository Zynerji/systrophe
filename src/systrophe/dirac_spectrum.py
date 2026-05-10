"""Dirac spectrum and bound-state analysis on the Tipler exterior.

Builds on `dirac.py`: the radial Dirac ODE system becomes an eigenvalue
problem when boundary conditions are imposed at both endpoints. We use
a shooting method:

  1. For trial energy E, integrate the radial Dirac system from
     r = r_min with chosen seed (R1, R2) = (1, 0) outward to r_max.
  2. Bound state condition: |R(r_max)| = 0 (Dirichlet) or some other
     specified outer condition.
  3. Scan E, locate the roots of the boundary functional via bisection.

For the Tipler exterior, the radial Dirac equation is a coupled two-
component system whose F, K, L, h have log-periodic structure in the
supercritical regime. Bound states exist when a confining mechanism
(e.g. truncating to r in [R, r_max] with Dirichlet BCs) is imposed.
The continuous unbounded-r spectrum is left to a separate scattering
analysis.

Tests (in test_dirac_spectrum.py) verify:
- Minkowski cylindrical limit: the lowest bound state energies
  approach the free-Dirac levels in a cylindrical box.
- Bound-state count grows with (r_max - r_min) as expected.
- The shooting functional is continuous and has discrete zeros.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .dirac import solve_radial_dirac


def boundary_functional(
    F_fn: Callable[[float], float],
    K_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    h_fn: Callable[[float], float],
    E: float,
    m: int,
    k: float,
    mass: float,
    r_min: float,
    r_max: float,
    R1_seed: complex = 1.0 + 0j,
    R2_seed: complex = 0.0 + 0j,
    n_samples: int = 401,
) -> float:
    """Boundary functional for the shooting method.

    Integrates the radial Dirac with Dirichlet seed at r_min and returns
    |R1(r_max)|^2 + |R2(r_max)|^2. Bound states are at zeros (or local
    minima) of this functional as a function of E.
    """
    sol = solve_radial_dirac(
        F_fn, K_fn, L_fn, h_fn,
        E=E, m=m, k=k, mass=mass,
        r0=r_min, R1_0=R1_seed, R2_0=R2_seed,
        r_max=r_max, n_samples=n_samples,
    )
    R1_end = sol["R1"][-1]
    R2_end = sol["R2"][-1]
    return float(abs(R1_end) ** 2 + abs(R2_end) ** 2)


def find_bound_states(
    F_fn: Callable[[float], float],
    K_fn: Callable[[float], float],
    L_fn: Callable[[float], float],
    h_fn: Callable[[float], float],
    m: int,
    k: float,
    mass: float,
    r_min: float,
    r_max: float,
    E_min: float,
    E_max: float,
    n_E: int = 200,
    refine: bool = True,
) -> np.ndarray:
    """Find bound-state energies E in [E_min, E_max].

    Sweep the boundary functional over E, locate local minima close to
    zero, and (if `refine=True`) bisect to refine each root using
    derivative-sign-change detection of the functional.

    Returns the array of bound-state energies.
    """
    E_grid = np.linspace(E_min, E_max, n_E)
    F_vals = np.array([
        boundary_functional(
            F_fn, K_fn, L_fn, h_fn,
            E=float(E), m=m, k=k, mass=mass,
            r_min=r_min, r_max=r_max,
        ) for E in E_grid
    ])

    # Local minima where the functional approaches zero
    candidates = []
    for i in range(1, len(E_grid) - 1):
        if F_vals[i] < F_vals[i - 1] and F_vals[i] < F_vals[i + 1]:
            candidates.append(E_grid[i])

    if not refine:
        return np.array(candidates)

    refined = []
    for Ec in candidates:
        try:
            # Use scipy.optimize.minimize_scalar to refine inside a window
            from scipy.optimize import minimize_scalar
            dE = (E_max - E_min) / n_E
            result = minimize_scalar(
                lambda E: boundary_functional(
                    F_fn, K_fn, L_fn, h_fn,
                    E=float(E), m=m, k=k, mass=mass,
                    r_min=r_min, r_max=r_max,
                ),
                bounds=(Ec - dE, Ec + dE),
                method="bounded",
                options={"xatol": 1e-9},
            )
            refined.append(float(result.x))
        except Exception:
            refined.append(float(Ec))
    return np.array(refined)


def vanstockum_bound_states(
    vs,
    m: int,
    k: float,
    mass: float,
    r_max_factor: float = 5.0,
    E_min: float = 0.5,
    E_max: float = 5.0,
    n_E: int = 200,
) -> dict:
    """Find bound-state energies for the Tipler exterior of a van Stockum cylinder.

    Uses Dirichlet BCs at r = R and r = r_max_factor * R.

    Returns a dict with the bound-state energies and metadata.
    """
    from .vanstockum import VanStockumInterior

    if not isinstance(vs, VanStockumInterior):
        raise TypeError("vs must be a VanStockumInterior")
    if not vs.is_supercritical():
        raise ValueError(
            "Bound-state search requires a > 1/2 (Tipler exterior); subcritical"
            " case has no oscillatory exterior to confine the spinor."
        )

    F_fn = lambda r: float(vs.analytic_exterior_F(r))
    K_fn = lambda r: float(vs.analytic_exterior_K(r))
    L_fn = lambda r: float(vs.analytic_exterior_L(r))
    h_fn = lambda r: 1.0  # leading-order h approximation

    r_min = vs.R + 0.01 * vs.R
    r_max = r_max_factor * vs.R
    energies = find_bound_states(
        F_fn, K_fn, L_fn, h_fn,
        m=m, k=k, mass=mass,
        r_min=r_min, r_max=r_max,
        E_min=E_min, E_max=E_max, n_E=n_E,
    )
    return {
        "energies": energies,
        "n_bound": int(energies.size),
        "r_min": r_min,
        "r_max": r_max,
        "m": m,
        "k": k,
        "mass": mass,
    }
