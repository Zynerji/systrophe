"""Loop-Quantum-Gravity-like discretization of the LP geometry.

Compute discrete area and volume spectra on a constant-t slice of the
LP exterior. In LQG, area and volume are quantized:

    Area eigenvalues: A_j = 8 pi gamma l_P^2 sqrt(j(j+1)), j = 0, 1/2, 1, ...
    Volume eigenvalues: V_v = (8 pi gamma l_P^2)^{3/2} f(j_e)

Here gamma is the Immirzi parameter (gamma ~ 0.2375 from BH entropy
matching), l_P is the Planck length.

For the LP cylinder, we triangulate a constant-t slice into
spinfoam vertices, compute LQG-quantized areas of the faces, and
compare against classical proper areas A_classical = int sqrt(L) dphi dz.

Functions
---------
- planck_area_unit: 8 pi gamma l_P^2
- lqg_area_spectrum_at_j: A_j formula
- lqg_volume_spectrum_at_j: V_j formula
- classical_proper_area: A_class on the LP cylinder
- spin_for_classical_area: invert A_j to get j matching A_class
- discretization_error_relative: |A_class - A_j| / A_class
- count_vertices_in_region: estimate spinfoam vertex count
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior

# LQG constants (natural units, l_P = 1)
IMMIRZI = 0.2375
LPLANCK = 1.0  # natural units


@dataclass(frozen=True)
class LQGAreaQuantization:
    j: float
    A_j: float
    A_classical: float
    relative_error: float


def planck_area_unit() -> float:
    """8 pi gamma l_P^2."""
    return float(8 * math.pi * IMMIRZI * LPLANCK ** 2)


def lqg_area_spectrum_at_j(j: float) -> float:
    """A_j = 8 pi gamma l_P^2 sqrt(j(j+1))."""
    if j < 0:
        raise ValueError("j must be non-negative")
    return float(planck_area_unit() * math.sqrt(j * (j + 1)))


def lqg_volume_spectrum_at_j(j: float, n_edges: int = 3) -> float:
    """V_v = (8 pi gamma l_P^2)^{3/2} * f(j, n_edges).

    Use the simple f = sqrt(j(j+1)/12) approximation for a trivalent vertex.
    """
    if j < 0:
        raise ValueError("j must be non-negative")
    A_unit = planck_area_unit()
    if n_edges < 3:
        raise ValueError("Vertex needs at least 3 edges")
    return float(A_unit ** 1.5 * math.sqrt(j * (j + 1) / 12.0))


def classical_proper_area(
    vs: VanStockumInterior, r: float,
    phi_range: tuple[float, float] = (0.0, 2 * math.pi),
    z_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Proper area of a (phi, z)-slice at fixed r."""
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    if L <= 0:
        # CTC region: area imaginary; return |L|^{1/2}
        return float(math.sqrt(abs(L)) * (phi_range[1] - phi_range[0])
                     * (z_range[1] - z_range[0]))
    return float(math.sqrt(L) * (phi_range[1] - phi_range[0])
                 * (z_range[1] - z_range[0]))


def spin_for_classical_area(
    vs: VanStockumInterior, r: float,
    phi_range: tuple[float, float] = (0.0, 2 * math.pi),
    z_range: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Solve A_j = A_classical for j (continuous; round to half-integer)."""
    A_class = classical_proper_area(vs, r, phi_range, z_range)
    A_unit = planck_area_unit()
    # A_j = A_unit * sqrt(j (j+1)) -> j^2 + j - (A_class/A_unit)^2 = 0
    disc = 1 + 4 * (A_class / A_unit) ** 2
    j = (-1 + math.sqrt(disc)) / 2.0
    return float(j)


def discretization_error_relative(
    vs: VanStockumInterior, r: float,
    phi_range: tuple[float, float] = (0.0, 2 * math.pi),
    z_range: tuple[float, float] = (0.0, 1.0),
) -> LQGAreaQuantization:
    """Compute the LQG quantization error at radius r."""
    A_class = classical_proper_area(vs, r, phi_range, z_range)
    j_cont = spin_for_classical_area(vs, r, phi_range, z_range)
    # Round to nearest half-integer
    j_quant = round(j_cont * 2) / 2.0
    A_q = lqg_area_spectrum_at_j(j_quant)
    rel_err = abs(A_class - A_q) / max(A_class, 1e-30)
    return LQGAreaQuantization(
        j=j_quant, A_j=A_q, A_classical=A_class,
        relative_error=float(rel_err),
    )


def count_vertices_in_region(
    vs: VanStockumInterior, r_inner: float, r_outer: float,
    cylinder_length: float = 1.0,
    vertex_density: float = 1.0,
) -> int:
    """Heuristic count of spinfoam vertices in a region.

    n_vertices ~ V_proper / (l_P^3).
    """
    if r_outer <= r_inner:
        raise ValueError("r_outer must exceed r_inner")
    r_samples = np.linspace(r_inner, r_outer, 50)
    L_vals = vs.analytic_exterior_L(r_samples)
    integrand = np.sqrt(np.abs(L_vals))
    dr = (r_outer - r_inner) / (len(r_samples) - 1)
    V_proper = float(np.trapezoid(integrand, dx=dr) * 2 * math.pi * cylinder_length)
    n_v = int(V_proper / (LPLANCK ** 3) * vertex_density)
    return n_v


def lp_area_quantum_signature(
    vs: VanStockumInterior, n_radii: int = 30,
    r_min: float = None, r_max: float = None,
) -> dict:
    """Sample the LQG quantization across a range of r and look for
    log-periodic structure in the discretization error (LP signature)."""
    if r_min is None:
        r_min = vs.R * 1.05
    if r_max is None:
        r_max = vs.R * 50.0
    r_grid = np.geomspace(r_min, r_max, n_radii)
    rel_errs = []
    for r in r_grid:
        q = discretization_error_relative(vs, float(r))
        rel_errs.append(q.relative_error)
    return {
        "r_grid": r_grid,
        "rel_errors": np.asarray(rel_errs),
        "mean_rel_error": float(np.mean(rel_errs)),
        "max_rel_error": float(np.max(rel_errs)),
    }


def regge_calculus_deficit_angle(
    vs: VanStockumInterior, r: float, n_edges: int = 6,
) -> float:
    """Regge deficit angle around a vertex at radius r.

    Deficit ~ 2 pi (1 - n_edges * theta / (2 pi)) where theta is
    the angular contribution from each edge. For a regular n-gon,
    theta = (n-2) pi / n.
    """
    if n_edges < 3:
        raise ValueError("Vertex needs at least 3 edges")
    theta = (n_edges - 2) * math.pi / n_edges
    deficit = 2 * math.pi - n_edges * theta
    # Modulate by local curvature: deficit *= |F| (heuristic curvature proxy)
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    return float(deficit * abs(F))
