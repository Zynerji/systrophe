"""Warp-geometry floor optimizer - shrink the Ford-Roman principal itself.

The bias controller, capacitor, nested cylinders, and band-coverage optimizer
all *relocate* the Jupiter-mass principal; none reduce it. The one lever that
reduces the principal *itself* is geometry, because the bare negative energy
scales as

    E_principal ~ kappa * (v_s/c)^2 * R_shell^2 / sigma * c^4/G          (1)

(the same scaling used by ``knopp_ratchet.feasibility_report``). Van den Broeck
(1999) decouples the energy-driving *shell* radius R_shell from the passenger
*pocket* radius R_pocket via a topological neck: the dragged warp wall can be
microscopic (tiny R_shell -> tiny R_shell^2) while the interior the crew
occupies stays macroscopic. Equation (1) then collapses the principal by many
orders of magnitude.

What bounds it (kept honest)
----------------------------
1. The Ford-Roman quantum inequality forbids arbitrarily thin walls: sigma is
   floored at ``planck_multiple * l_Planck`` (Pfenning-Ford 1997 found the
   Alcubierre wall must be >~ 10^2 Planck lengths). At that floor the wall is
   Planck-scale and the SEMICLASSICAL treatment is no longer trustworthy.
2. This module models only the Alcubierre *shell* energy (Eq. 1). Van den
   Broeck's *blow-up* region (the neck expansion R_shell -> R_pocket) carries
   additional negative energy that Pfenning-Ford showed re-inflates the total
   in a model-dependent, contested way. The shell-only number here is therefore
   an OPTIMISTIC lower bound, not a validated total.

So the deliverable is a rigorous SCALING reduction with an explicit residual
gap - not a path to lab feasibility, and not a claim that the wall is broken.
The principal-per-metre is conservation-bounded; geometry only changes the
prefactor R_shell^2/sigma within the QI-allowed region.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .knopp_ratchet import (
    _C_SI,
    _G_SI,
    _GEOM_ENERGY_PER_METRE_J,
    _JUPITER_MASS_KG,
)
from .novelty_catcher import scan_novelty

PLANCK_LENGTH_M = 1.616255e-35  # m


def principal_joules(
    R_shell_m: float,
    sigma_m: float,
    v_s_over_c: float = 1.0,
    kappa: float = 1.0 / 12.0,
) -> float:
    """Bare negative-energy principal (SI joules) for a shell of radius
    ``R_shell_m`` and wall thickness ``sigma_m`` via Eq. (1).
    """
    e_geom_m = kappa * v_s_over_c ** 2 * R_shell_m ** 2 / sigma_m
    return abs(e_geom_m) * _GEOM_ENERGY_PER_METRE_J


def qi_wall_floor_m(planck_multiple: float = 100.0) -> float:
    """Ford-Roman QI-limited minimum wall thickness (Pfenning-Ford scale)."""
    return float(planck_multiple) * PLANCK_LENGTH_M


@dataclass(frozen=True)
class GeometryFloorReport:
    # Inputs
    v_s_over_c: float
    kappa: float
    planck_multiple: float
    R_pocket_m: float
    # Baseline (macroscopic 1 m bubble, sigma = 1 m)
    baseline_principal_J: float
    baseline_jupiter_masses: float
    # Optimized shell geometry (R_shell = sigma = QI floor)
    R_shell_opt_m: float
    sigma_opt_m: float
    optimized_principal_J: float
    # Reductions
    reduction_factor: float
    reduction_orders_of_magnitude: float
    # Combined with band-coverage paid fraction
    out_of_band_fraction: float
    combined_paid_principal_J: float
    # Residual reality check
    lab_source_J: float
    residual_orders_of_magnitude: float
    # Catcher
    novelty_verdict: str
    novelty_n_sharp: int
    # Honest flags
    wall_is_planck_scale: bool
    blowup_energy_modeled: bool


def optimize_geometry(
    R_pocket_m: float = 2.0,
    v_s_over_c: float = 1.0,
    kappa: float = 1.0 / 12.0,
    planck_multiple: float = 100.0,
    out_of_band_fraction: float = 0.139,
    lab_source_J: float = 1e-15,
) -> GeometryFloorReport:
    """Minimize the shell principal over QI-allowed (R_shell, sigma).

    Eq. (1) is decreasing in 1/sigma and increasing in R_shell^2, so subject to
    sigma <= R_shell (a coherent wall is no thicker than its bubble) and
    R_shell >= sigma_min (QI floor), the minimum is at
    R_shell = sigma = sigma_min. ``out_of_band_fraction`` folds in the
    band-coverage optimizer result (the paid portion of the route).
    """
    sigma_min = qi_wall_floor_m(planck_multiple)
    R_shell_opt = sigma_min
    sigma_opt = sigma_min

    baseline = principal_joules(1.0, 1.0, v_s_over_c, kappa)
    optimized = principal_joules(R_shell_opt, sigma_opt, v_s_over_c, kappa)

    reduction = baseline / max(optimized, 1e-300)
    red_oom = math.log10(reduction) if reduction > 0 else float("inf")

    combined = optimized * out_of_band_fraction
    residual = combined / max(lab_source_J, 1e-300)
    res_oom = math.log10(residual) if residual > 0 else float("-inf")

    # Catcher over the R_shell -> floor sweep (log-spaced).
    R_grid = np.logspace(math.log10(sigma_min), 0.0, 40)  # sigma_min ... 1 m

    def fn(Rv: float) -> np.ndarray:
        # at each R_shell use the thickest QI-allowed wall (sigma = R_shell)
        return np.array([principal_joules(Rv, max(Rv, sigma_min), v_s_over_c, kappa)])

    nov = scan_novelty(R_grid, fn, n_bits=32, parameter_label="R_shell")

    return GeometryFloorReport(
        v_s_over_c=float(v_s_over_c),
        kappa=float(kappa),
        planck_multiple=float(planck_multiple),
        R_pocket_m=float(R_pocket_m),
        baseline_principal_J=float(baseline),
        baseline_jupiter_masses=float(baseline / (_JUPITER_MASS_KG * _C_SI ** 2)),
        R_shell_opt_m=float(R_shell_opt),
        sigma_opt_m=float(sigma_opt),
        optimized_principal_J=float(optimized),
        reduction_factor=float(reduction),
        reduction_orders_of_magnitude=float(red_oom),
        out_of_band_fraction=float(out_of_band_fraction),
        combined_paid_principal_J=float(combined),
        lab_source_J=float(lab_source_J),
        residual_orders_of_magnitude=float(res_oom),
        novelty_verdict=nov.verdict,
        novelty_n_sharp=len(nov.sharp_features),
        wall_is_planck_scale=bool(sigma_min < 1e-30),
        blowup_energy_modeled=False,
    )


def summarise_geometry(r: GeometryFloorReport) -> str:
    """One-line honest summary."""
    return (
        f"WarpGeometry: baseline={r.baseline_principal_J:.2e} J "
        f"({r.baseline_jupiter_masses:.2g} Jupiter) -> optimized="
        f"{r.optimized_principal_J:.2e} J "
        f"(-{r.reduction_orders_of_magnitude:.0f} OOM); "
        f"x band-coverage -> {r.combined_paid_principal_J:.2e} J paid; "
        f"still ~{r.residual_orders_of_magnitude:.0f} OOM above lab; "
        f"PLANCK-WALL={r.wall_is_planck_scale}, blowup_modeled="
        f"{r.blowup_energy_modeled}"
    )
