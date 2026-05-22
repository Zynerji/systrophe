"""Thick-wall Van den Broeck variant -- the variational closure.

The §10 calibration left one open escape: maybe a thick/gradual blow-up wall
(transition width w >> R_shell) lowers the pocket energy enough to matter, while
keeping the interior usable. This module settles it by CALCULUS OF VARIATIONS --
finding the wall profile B(r) that minimizes the exact pocket energy

    E[B] = (1/2) integral (B'^2 / B) r^2 dr        (geometrized; x c^4/G for J)

over ALL profiles at once, not just tanh walls.

The Euler-Lagrange equation
---------------------------
With B = u^2 the functional's EL equation collapses to (r^2 u')' = 0, so
u = d - c/r and

    B_opt(r) = (1 + |c|/r)^2,        |c| = R_shell (sqrt(B_max) - 1),

the ISOTROPIC-SCHWARZSCHILD spatial form. For boundary conditions
B(R_shell)=B_max, B(R_2)=1 the optimal energy is

    E = 2 (sqrt(B_max)-1)^2 * R_shell * R_2 / (R_2 - R_shell),

which DECREASES as the wall thickens (R_2 -> inf), bottoming at the global
minimum

    E_min = 2 R_shell (sqrt(B_max) - 1)^2.

So the thick wall is genuinely optimal -- but it does not escape the floor.

The floor
---------
The VdB point is a large interior from a small exterior:
B_max = rho_use / R_shell (rho_use = proper interior radius). Substituting,

    E_min = 2 rho_use - 4 sqrt(rho_use * R_shell) + 2 R_shell  ->  2 rho_use
            as R_shell -> 0.

The blow-up floor is ~ 2 (c^4/G) rho_use -- set by the PROPER interior radius,
Jupiter-scale for a metre interior, and shrinking the exterior shell makes it
slightly WORSE, not better. This is a positive-mass-theorem-flavoured
inevitability: holding open that much proper volume costs ~ its proper size.

Interior usability
------------------
The flat-B core (r < R_shell, B = B_max constant) is genuinely FLAT space
(rescaled) -- zero tidal field, fully habitable. Van den Broeck's
interior-usability claim is correct. It is the ENERGY that is prohibitive, not
the livability. Verdict: the thick-wall escape is CLOSED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .knopp_ratchet import _C_SI, _G_SI, _GEOM_ENERGY_PER_METRE_J, _JUPITER_MASS_KG
from .novelty_catcher import scan_novelty


def optimal_energy_geometrized(R_shell: float, B_max: float,
                               R2_over_Rshell: float = float("inf")) -> float:
    """EL-optimal pocket energy (geometrized length) for a wall of outer edge
    R_2 = R2_over_Rshell * R_shell. Infinite => the global minimum
    E_min = 2 R_shell (sqrt(B_max)-1)^2.
    """
    s = (math.sqrt(B_max) - 1.0) ** 2
    if math.isinf(R2_over_Rshell):
        return 2.0 * R_shell * s
    f = R2_over_Rshell
    return 2.0 * s * R_shell * f / (f - 1.0)


def optimal_energy_numeric(R_shell: float, B_max: float,
                           rmax_factor: float = 1e6, n: int = 1_000_000) -> float:
    """Numerical check of E[B_opt] using the isotropic-Schwarzschild profile."""
    c = R_shell * (math.sqrt(B_max) - 1.0)
    r = np.linspace(R_shell, rmax_factor * R_shell, n)
    B = (1.0 + c / r) ** 2
    Bp = np.gradient(B, r)
    return float(np.trapezoid(0.5 * (Bp ** 2 / B) * r ** 2, r))


@dataclass(frozen=True)
class ThickWallReport:
    rho_use_m: float
    R_shell_m: float
    B_max: float
    E_min_geometrized_m: float
    E_min_J: float
    E_min_jupiter_masses: float
    energy_per_proper_metre: float        # E_min_geom / rho_use -> ~2
    thicker_wall_is_optimal: bool         # finite R2 gives higher E
    shrinking_shell_helps: bool           # False (floor worsens as R_shell->0)
    interior_is_flat_usable: bool         # True (flat-B core, no tides)
    escape_closed: bool                   # True
    lab_source_J: float
    residual_oom: float
    novelty_verdict: str
    novelty_n_sharp: int


def thick_wall_floor(
    rho_use_m: float = 2.0,
    R_shell_m: float = 1e-15,
    lab_source_J: float = 1e-15,
) -> ThickWallReport:
    """Minimum blow-up energy to hold open a usable interior of proper radius
    rho_use from an exterior coordinate radius R_shell, using the variationally
    optimal (thickest) wall. Reports that the escape is closed.
    """
    B_max = rho_use_m / R_shell_m
    E_geom = optimal_energy_geometrized(R_shell_m, B_max)  # global min (R2->inf)
    E_J = E_geom * _GEOM_ENERGY_PER_METRE_J

    # thick-wall optimality: a finite (thinner) wall costs more
    E_thin = optimal_energy_geometrized(R_shell_m, B_max, R2_over_Rshell=2.0)
    thicker_optimal = E_thin > E_geom

    # does shrinking the exterior help? compare a 10x larger shell
    E_biggershell = optimal_energy_geometrized(
        R_shell_m * 10.0, rho_use_m / (R_shell_m * 10.0))
    shrink_helps = E_geom < E_biggershell  # True only if smaller shell is cheaper

    residual = math.log10(E_J / max(lab_source_J, 1e-300))

    # catcher over R_shell sweep at fixed rho_use (the floor is ~flat -> smooth)
    Rs_grid = np.logspace(math.log10(rho_use_m) - 30, math.log10(rho_use_m) - 1, 40)

    def fn(Rs: float) -> np.ndarray:
        bm = rho_use_m / Rs
        return np.array([optimal_energy_geometrized(Rs, bm)])

    nov = scan_novelty(Rs_grid, fn, n_bits=32, parameter_label="R_shell")

    return ThickWallReport(
        rho_use_m=float(rho_use_m),
        R_shell_m=float(R_shell_m),
        B_max=float(B_max),
        E_min_geometrized_m=float(E_geom),
        E_min_J=float(E_J),
        E_min_jupiter_masses=float(E_J / (_JUPITER_MASS_KG * _C_SI ** 2)),
        energy_per_proper_metre=float(E_geom / rho_use_m),
        thicker_wall_is_optimal=bool(thicker_optimal),
        shrinking_shell_helps=bool(shrink_helps),
        interior_is_flat_usable=True,   # flat-B core is flat space, no tides
        escape_closed=True,
        lab_source_J=float(lab_source_J),
        residual_oom=float(residual),
        novelty_verdict=nov.verdict,
        novelty_n_sharp=len(nov.sharp_features),
    )


def summarise_thick_wall(r: ThickWallReport) -> str:
    return (
        f"ThickWallVdB: rho_use={r.rho_use_m} m, R_shell={r.R_shell_m:.0e} m, "
        f"B_max={r.B_max:.0e} -> E_min={r.E_min_J:.2e} J "
        f"({r.E_min_jupiter_masses:.2g} Jupiter); E/rho_use={r.energy_per_proper_metre:.2f}; "
        f"thicker_wall_optimal={r.thicker_wall_is_optimal}; "
        f"shrinking_shell_helps={r.shrinking_shell_helps}; "
        f"interior_usable={r.interior_is_flat_usable}; ESCAPE_CLOSED={r.escape_closed}"
    )
