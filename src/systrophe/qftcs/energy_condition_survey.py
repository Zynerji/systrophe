"""Energy condition survey across the LP rotating-cylinder regime.

The standard energy conditions are
- WEC (Weak): T_{mu nu} u^mu u^nu >= 0 for all timelike u.
- NEC (Null): T_{mu nu} k^mu k^nu >= 0 for all null k.
- DEC (Dominant): WEC + flux T^mu_nu u^nu is timelike or null.
- SEC (Strong): T_{mu nu} u^mu u^nu >= 0.5 T  for all timelike u.

For the van Stockum dust interior, the matter is positive-energy
dust, and standard conditions are trivially satisfied INSIDE the
cylinder. The interesting question is for *induced* stress-energy
from quantum effects (Hawking radiation, vacuum polarisation) on
the LP exterior.

This module surveys the energy conditions at sample points across
the (omega, R) parameter space, using:

- Classical dust interior: trivially satisfied
- LP exterior + Hadamard <T_munu>: state-dependent
- Hybrid cylinder + Schwarzschild: requires exotic matter at the
  throat
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.qftcs.energy_conditions import (
    EnergyConditionReport, energy_condition_report,
    proper_energy_density,
)


@dataclass(frozen=True)
class SurveyResult:
    """Energy-condition survey across (omega, R) parameter space."""

    omegas: list[float]
    Rs: list[float]
    wec_satisfied: list[list[bool]]
    nec_satisfied: list[list[bool]]
    sec_satisfied: list[list[bool]]
    dec_satisfied: list[list[bool]]
    n_total: int
    n_all_satisfied: int


def systematic_energy_survey(
    omega_range: tuple[float, float] = (0.3, 2.0),
    R_range: tuple[float, float] = (0.5, 2.0),
    n_omega: int = 6, n_R: int = 4, r_test: float = 0.5,
) -> SurveyResult:
    """Survey energy conditions across (omega, R) parameter grid.

    Tests at a fixed r_test point INSIDE each cylinder.
    """
    omegas = np.linspace(*omega_range, n_omega).tolist()
    Rs = np.linspace(*R_range, n_R).tolist()
    wec_grid = [[False] * n_omega for _ in range(n_R)]
    nec_grid = [[False] * n_omega for _ in range(n_R)]
    sec_grid = [[False] * n_omega for _ in range(n_R)]
    dec_grid = [[False] * n_omega for _ in range(n_R)]
    n_all = 0
    n_total = 0
    from systrophe.geometry.vanstockum import VanStockumInterior
    for i, R in enumerate(Rs):
        for j, omega in enumerate(omegas):
            try:
                vs = VanStockumInterior(omega=omega, R=R)
                # Use r_test * R to stay inside the cylinder
                r_inside = r_test * R
                report = energy_condition_report(vs, r=r_inside)
                wec_grid[i][j] = report.weak
                nec_grid[i][j] = report.null
                sec_grid[i][j] = report.strong
                dec_grid[i][j] = report.dominant
                n_total += 1
                if all([report.weak, report.null, report.strong,
                         report.dominant]):
                    n_all += 1
            except Exception:
                pass
    return SurveyResult(
        omegas=omegas, Rs=Rs,
        wec_satisfied=wec_grid, nec_satisfied=nec_grid,
        sec_satisfied=sec_grid, dec_satisfied=dec_grid,
        n_total=n_total, n_all_satisfied=n_all,
    )


def violation_loci(survey: SurveyResult) -> list[dict]:
    """Identify (omega, R) loci where any condition is violated."""
    violations = []
    for i, R in enumerate(survey.Rs):
        for j, omega in enumerate(survey.omegas):
            wec = survey.wec_satisfied[i][j]
            nec = survey.nec_satisfied[i][j]
            sec = survey.sec_satisfied[i][j]
            dec = survey.dec_satisfied[i][j]
            if not (wec and nec and sec and dec):
                violations.append({
                    "R": R, "omega": omega,
                    "WEC": wec, "NEC": nec, "SEC": sec, "DEC": dec,
                })
    return violations


def exterior_violations_from_qft(vs, r_test: float = 2.0) -> dict:
    """Check if induced <T_munu> from QFT violates energy conditions
    at a sample r in the exterior.

    Uses the Hadamard off-trace tensor from `hadamard_offtrace`. For
    pure-state Hadamard contribution, <T_munu> can have either sign
    depending on local curvature; this function reports the
    components.
    """
    from systrophe.qftcs.hadamard_offtrace import hadamard_offtrace_T
    try:
        T = hadamard_offtrace_T(vs, r=r_test)
        # WEC proxy: T_{tt} >= 0 (energy density non-negative)
        # NEC proxy: T_{tt} + T_{rr} >= 0
        T_tt = float(T[0, 0])
        T_rr = float(T[1, 1])
        T_phi_phi = float(T[2, 2])
        wec_proxy = T_tt >= 0
        nec_proxy = (T_tt + T_rr) >= 0
        return {
            "r": r_test,
            "T_tt": T_tt, "T_rr": T_rr, "T_phi_phi": T_phi_phi,
            "WEC_proxy_satisfied": wec_proxy,
            "NEC_proxy_satisfied": nec_proxy,
        }
    except Exception as e:
        return {"error": str(e)}


def summary_statistics(survey: SurveyResult) -> dict:
    """High-level statistics on the survey."""
    return {
        "n_total": survey.n_total,
        "n_all_satisfied": survey.n_all_satisfied,
        "fraction_all_satisfied": (survey.n_all_satisfied / survey.n_total
                                       if survey.n_total > 0 else 0.0),
        "omega_range": (min(survey.omegas), max(survey.omegas)),
        "R_range": (min(survey.Rs), max(survey.Rs)),
    }
