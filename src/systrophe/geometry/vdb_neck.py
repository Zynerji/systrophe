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

from systrophe.knopp.knopp_ratchet import _C_SI, _G_SI, _GEOM_ENERGY_PER_METRE_J, _JUPITER_MASS_KG
from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.warp_geometry import PLANCK_LENGTH_M, principal_joules, qi_wall_floor_m


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


# ===========================================================================
# CALIBRATED via the exact Einstein-tensor integral
# ===========================================================================
#
# For the VdB pocket the spatial slice is conformally flat, gamma_ij = B(r)^2
# delta_ij. For a static slice (vanishing extrinsic curvature) the Hamiltonian
# constraint gives the energy density rho = R^(3)/(16 pi). Using the 3D
# conformal Ricci scalar of gamma_ij = B^2 delta_ij and E = integral rho
# sqrt(gamma) d^3x with sqrt(gamma) = B^3, the curvature total-derivative terms
# (-d/dr(B' r^2)) integrate to zero for flat-at-both-ends B, leaving the EXACT
# result (geometrized units; multiply by c^4/G for joules):
#
#     E_pocket = (1/2) integral_0^inf (B'^2 / B) r^2 dr.
#
# This is a real Einstein-tensor integral, not an estimate. Evaluated
# numerically it scales as E ~ K * B_max * R_w^2 / w (verified: the B_max
# exponent is 1.0, NOT the ln^2 of the scaffold above), with K ~ 0.55 for a
# tanh wall. Since B_max = R_pocket / R_shell, this collapses to
#
#     E_pocket ~ K * (c^4/G) * R_pocket * (R_shell / w),
#
# so for a localized thin wall (w ~ R_shell) the shell radius CANCELS and the
# blow-up floor is ~ (c^4/G) * R_pocket -- Jupiter-scale for a metre pocket,
# independent of how small the shell is made. This OVERTURNS the optimistic
# scaffold (which omitted the blow-up and used the wrong ln^2 scaling).


def pocket_energy_geometrized(
    B_max: float,
    R_w: float = 1.0,
    w: float = 0.1,
    rmax_factor: float = 6.0,
    n: int = 200000,
) -> float:
    """Exact static-pocket energy (1/2) int (B'^2/B) r^2 dr, geometrized (m).

    B(r) is a tanh wall: B_max inside r < R_w, 1 outside, transition width w.
    This is the Hamiltonian-constraint energy of the conformally-flat pocket.
    """
    r = np.linspace(1e-4 * R_w, rmax_factor * R_w, n)
    B = 1.0 + (B_max - 1.0) * 0.5 * (1.0 - np.tanh((r - R_w) / w))
    Bp = np.gradient(B, r)
    integrand = 0.5 * (Bp ** 2 / B) * r ** 2
    return float(np.trapezoid(integrand, r))


def calibrate_K(B_max: float = 1e5, R_w: float = 1.0, w: float = 0.1) -> float:
    """Calibrate K in E = K * B_max * R_w^2 / w from the exact integral.

    Computed numerically (auditable), not hardcoded. K -> ~0.55 (tanh wall) in
    the deep-pocket limit, where E/B_max has converged to a constant.
    """
    E = pocket_energy_geometrized(B_max, R_w, w)
    return float(E * w / (B_max * R_w ** 2))


def blowup_energy_calibrated(
    R_shell_m: float,
    R_pocket_m: float,
    wall_thickness_m: float,
    K: float | None = None,
) -> float:
    """Calibrated VdB blow-up energy (J) from the exact integral scaling.

    E = K * (c^4/G) * B_max * R_shell^2 / w,  B_max = R_pocket / R_shell.
    """
    if R_pocket_m <= R_shell_m:
        return 0.0
    if K is None:
        K = calibrate_K()
    B_max = R_pocket_m / R_shell_m
    e_geom_m = K * B_max * R_shell_m ** 2 / max(wall_thickness_m, 1e-300)
    return float(e_geom_m * _GEOM_ENERGY_PER_METRE_J)


@dataclass(frozen=True)
class VdbCalibratedReport:
    R_shell_m: float
    R_pocket_m: float
    wall_thickness_m: float
    v_s_over_c: float
    K: float
    shell_principal_J: float
    blowup_calibrated_J: float
    blowup_scaffold_J: float            # the old uncalibrated ln^2 estimate
    total_floor_J: float
    total_floor_jupiter_masses: float
    baseline_principal_J: float
    net_reduction_oom: float            # baseline -> calibrated total
    scaffold_overestimated_reduction_by_oom: float
    lab_source_J: float
    residual_oom: float
    novelty_verdict: str
    novelty_n_sharp: int
    calibrated: bool                    # True
    geometry_reduces_floor: bool        # False for thin localized wall


def vdb_calibrated_floor(
    R_pocket_m: float = 2.0,
    v_s_over_c: float = 1.0,
    kappa: float = 1.0 / 12.0,
    planck_multiple: float = 100.0,
    lab_source_J: float = 1e-15,
) -> VdbCalibratedReport:
    """Calibrated VdB floor: shell + EXACT-integral blow-up, thin localized wall
    (R_shell = wall thickness = QI floor). Reports that geometry does NOT reduce
    the floor for a habitable (metre-scale) pocket.
    """
    sigma_min = qi_wall_floor_m(planck_multiple)
    R_shell = sigma_min
    w = sigma_min
    K = calibrate_K()

    shell = principal_joules(R_shell, sigma_min, v_s_over_c, kappa)
    blow_cal = blowup_energy_calibrated(R_shell, R_pocket_m, w, K)
    blow_scaffold = blowup_energy_estimate(R_shell, R_pocket_m, w, 1.0)
    total = shell + blow_cal
    baseline = principal_joules(1.0, 1.0, v_s_over_c, kappa)

    net_red = math.log10(baseline / total) if total > 0 else float("inf")
    scaffold_total = shell + blow_scaffold
    scaffold_red = math.log10(baseline / scaffold_total) if scaffold_total > 0 else float("inf")
    overestimate = scaffold_red - net_red
    residual = math.log10(total / max(lab_source_J, 1e-300))

    # Catcher over the calibrated blow-up vs R_pocket (the true driver).
    Rp_grid = np.logspace(-3, 1, 40)

    def fn(Rp: float) -> np.ndarray:
        return np.array([blowup_energy_calibrated(R_shell, max(Rp, R_shell * 1.01), w, K)])

    nov = scan_novelty(Rp_grid, fn, n_bits=32, parameter_label="R_pocket")

    return VdbCalibratedReport(
        R_shell_m=float(R_shell),
        R_pocket_m=float(R_pocket_m),
        wall_thickness_m=float(w),
        v_s_over_c=float(v_s_over_c),
        K=float(K),
        shell_principal_J=float(shell),
        blowup_calibrated_J=float(blow_cal),
        blowup_scaffold_J=float(blow_scaffold),
        total_floor_J=float(total),
        total_floor_jupiter_masses=float(total / (_JUPITER_MASS_KG * _C_SI ** 2)),
        baseline_principal_J=float(baseline),
        net_reduction_oom=float(net_red),
        scaffold_overestimated_reduction_by_oom=float(overestimate),
        lab_source_J=float(lab_source_J),
        residual_oom=float(residual),
        novelty_verdict=nov.verdict,
        novelty_n_sharp=len(nov.sharp_features),
        calibrated=True,
        geometry_reduces_floor=bool(total < baseline),
    )


def summarise_vdb_calibrated(r: VdbCalibratedReport) -> str:
    return (
        f"VdB[CALIBRATED K={r.K:.3f}]: shell={r.shell_principal_J:.2e} + "
        f"blowup={r.blowup_calibrated_J:.2e} -> total={r.total_floor_J:.2e} J "
        f"({r.total_floor_jupiter_masses:.2g} Jupiter); net reduction "
        f"{r.net_reduction_oom:.0f} OOM (scaffold overestimated by "
        f"{r.scaffold_overestimated_reduction_by_oom:.0f} OOM); "
        f"geometry_reduces_floor={r.geometry_reduces_floor}; "
        f"residual {r.residual_oom:.0f} OOM"
    )
