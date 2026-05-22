"""Van den Broeck neck blow-up energy -- RESEARCH SCAFFOLD (uncalibrated).

============================== READ THIS FIRST ==============================
This module is a SCALING ESTIMATE of the energy in the Van den Broeck "blow-up"
neck -- the region whose spatial expansion factor B(r_s) grows from 1 (exterior)
to B_max = R_pocket/R_shell (interior pocket). ``warp_geometry`` deliberately
modelled only the Alcubierre *shell* energy and flagged this term as omitted;
this scaffold supplies the missing term so the total is honest about the
Pfenning-Ford (1997) re-inflation.

It is NOT a full Einstein-tensor integral of the VdB metric. It uses the
standard gradient-energy form -- the negative energy that holds an expanded
region open scales as the square of the expansion gradient,

    rho_blowup  ~  C * (c^4 / G) * (B'/B)^2          [J/m^3],   C = O(1)

integrated over the neck shell volume ~ 4 pi R_shell^2 * Delta. With
B'/B ~ ln(B_max)/Delta this gives

    E_blowup  ~  C * (c^4/G) * (ln(R_pocket/R_shell))^2 * R_shell^2 / Delta.

The O(1) prefactor C (and whether the dominant term is (B'/B)^2 vs B''/B) is
UNCALIBRATED. The number this returns is a magnitude/scaling estimate to show
the DIRECTION and SCALE of the blow-up correction, flagged ``calibrated=False``.
Per the project rules it is not "validated" and no headline claim should rest
on its precise value -- only on the qualitative result that the blow-up term
DOMINATES the shell term and re-inflates the optimistic floor.
============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .knopp_ratchet import _C_SI, _G_SI, _GEOM_ENERGY_PER_METRE_J, _JUPITER_MASS_KG
from .novelty_catcher import scan_novelty
from .warp_geometry import PLANCK_LENGTH_M, principal_joules, qi_wall_floor_m


def blowup_energy_estimate(
    R_shell_m: float,
    R_pocket_m: float,
    neck_thickness_m: float,
    prefactor_C: float = 1.0,
) -> float:
    """Scaling estimate of the VdB neck blow-up negative energy (J).

    E_blowup ~ C * (c^4/G) * (ln(R_pocket/R_shell))^2 * R_shell^2 / Delta.
    UNCALIBRATED O(1) prefactor C. Returns 0 if there is no expansion.
    """
    if R_pocket_m <= R_shell_m:
        return 0.0
    ln_ratio = math.log(R_pocket_m / R_shell_m)
    e_geom_m = (prefactor_C * ln_ratio ** 2
                * R_shell_m ** 2 / max(neck_thickness_m, 1e-300))
    return float(e_geom_m * _GEOM_ENERGY_PER_METRE_J)


@dataclass(frozen=True)
class VdbFloorReport:
    R_shell_m: float
    R_pocket_m: float
    neck_thickness_m: float
    v_s_over_c: float
    prefactor_C: float
    shell_principal_J: float
    blowup_energy_J: float
    total_floor_J: float
    blowup_dominates: bool
    blowup_to_shell_ratio: float
    baseline_principal_J: float            # R=sigma=1 m, shell-only
    total_reduction_oom: float             # baseline -> total
    shell_only_reduction_oom: float        # baseline -> shell (optimistic)
    blowup_reinflation_oom: float          # how much the blow-up costs back
    lab_source_J: float
    residual_oom: float
    novelty_verdict: str
    novelty_n_sharp: int
    calibrated: bool                       # always False (scaffold)
    wall_is_planck_scale: bool


def vdb_total_floor(
    R_pocket_m: float = 2.0,
    v_s_over_c: float = 1.0,
    kappa: float = 1.0 / 12.0,
    planck_multiple: float = 100.0,
    prefactor_C: float = 1.0,
    lab_source_J: float = 1e-15,
) -> VdbFloorReport:
    """Honest VdB total: shell (calibrated scaling) + blow-up (uncalibrated).

    The shell minimum sits at R_shell = Delta = sigma_min (QI wall floor); the
    blow-up uses the same neck thickness. Reports how much the blow-up term
    re-inflates the optimistic shell-only floor.
    """
    sigma_min = qi_wall_floor_m(planck_multiple)
    R_shell = sigma_min
    delta = sigma_min

    shell = principal_joules(R_shell, sigma_min, v_s_over_c, kappa)
    blowup = blowup_energy_estimate(R_shell, R_pocket_m, delta, prefactor_C)
    total = shell + blowup
    baseline = principal_joules(1.0, 1.0, v_s_over_c, kappa)

    ratio = blowup / max(shell, 1e-300)
    total_red = math.log10(baseline / total) if total > 0 else float("inf")
    shell_red = math.log10(baseline / shell) if shell > 0 else float("inf")
    reinflation = shell_red - total_red  # OOM the blow-up costs back
    residual = math.log10(total / max(lab_source_J, 1e-300))

    # Catcher over the blow-up energy vs R_shell (log sweep).
    R_grid = np.logspace(math.log10(sigma_min), 0.0, 40)

    def fn(Rv: float) -> np.ndarray:
        return np.array([blowup_energy_estimate(Rv, R_pocket_m, max(Rv, sigma_min),
                                                prefactor_C)])

    nov = scan_novelty(R_grid, fn, n_bits=32, parameter_label="R_shell")

    return VdbFloorReport(
        R_shell_m=float(R_shell),
        R_pocket_m=float(R_pocket_m),
        neck_thickness_m=float(delta),
        v_s_over_c=float(v_s_over_c),
        prefactor_C=float(prefactor_C),
        shell_principal_J=float(shell),
        blowup_energy_J=float(blowup),
        total_floor_J=float(total),
        blowup_dominates=bool(blowup > shell),
        blowup_to_shell_ratio=float(ratio),
        baseline_principal_J=float(baseline),
        total_reduction_oom=float(total_red),
        shell_only_reduction_oom=float(shell_red),
        blowup_reinflation_oom=float(reinflation),
        lab_source_J=float(lab_source_J),
        residual_oom=float(residual),
        novelty_verdict=nov.verdict,
        novelty_n_sharp=len(nov.sharp_features),
        calibrated=False,
        wall_is_planck_scale=bool(sigma_min < 1e-30),
    )


def summarise_vdb(r: VdbFloorReport) -> str:
    """One-line summary. Always advertises the uncalibrated flag."""
    return (
        f"VdB[UNCALIBRATED C={r.prefactor_C:g}]: shell={r.shell_principal_J:.2e} J "
        f"+ blowup={r.blowup_energy_J:.2e} J -> total={r.total_floor_J:.2e} J; "
        f"blowup/shell={r.blowup_to_shell_ratio:.1e}; "
        f"reduction {r.total_reduction_oom:.0f} OOM (blow-up costs back "
        f"{r.blowup_reinflation_oom:.0f}); residual {r.residual_oom:.0f} OOM; "
        f"PLANCK-WALL={r.wall_is_planck_scale}"
    )
