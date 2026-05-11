"""Anomaly inflow on the Z_3 Mobius cover.

Builds the Callan-Harvey (Nucl. Phys. B 250 (1985) 427) anomaly-inflow
structure for a Dirac fermion on the Z_3 Mobius cover of the Tipler
exterior. The Z_3 cover has three branches indexed by b in {0, 1, 2}
with twisted boundary conditions

    psi(phi + 2 pi) = exp(2 pi i (b/3 + gamma_eff / (2 pi))) psi(phi)

so the effective twist on each branch is

    alpha_b = (b/3 + gamma_eff / (2 pi))  mod 1.

The eta-invariant of the Dirac operator on the angular S^1 with twist
alpha (Atiyah-Patodi-Singer):

    eta_D(alpha) = 1 - 2 alpha    for alpha in (0, 1),
    eta_D(0) = 0                  (symmetric regularisation; zero mode).

The Callan-Harvey relation states that the *bulk* axial anomaly
inflowing into a fixed locus equals the eta-invariant of the link
Dirac operator at that locus:

    integral_(bulk) d^4 x   <axial anomaly density>   =   eta_link / 2.

For a *closed* Z_3 cover (no external gauge field), the sum of the
three branch etas vanishes:

    eta_0 + eta_1 + eta_2 = 0       (gamma_eff = 0).

A non-zero gauge twist gamma_eff breaks this cancellation, producing
a residual anomaly that must be cancelled by an inflowing bulk
Chern-Simons-like current. This is the "anomaly inflow" Grok
referenced.

References
----------
- C. Callan & J. Harvey, "Anomalies and Fermion Zero Modes on Strings
  and Domain Walls," Nucl. Phys. B 250 (1985) 427.
- M. Atiyah, V. Patodi, I. Singer, "Spectral asymmetry and Riemannian
  geometry," Math. Proc. Camb. Phil. Soc. 77 (1975) 43.
"""

from __future__ import annotations

from math import pi

import numpy as np


# Standard axial anomaly coefficient: (1/16 pi^2) F_dual F.
AXIAL_ANOMALY_COEFFICIENT = 1.0 / (16.0 * pi * pi)


def dirac_eta_invariant(alpha: float | np.ndarray) -> np.ndarray:
    """APS eta-invariant of the angular Dirac operator with twist alpha.

    On S^1 with twisted boundary condition psi(2 pi) = exp(2 pi i alpha) psi(0),
    eigenvalues are lambda_n = n + alpha. Spectral asymmetry zeta'(0)
    evaluates (modulo 2 Z) to

        eta(alpha) = 1 - 2 alpha       for alpha in (0, 1),
        eta(0)     = 0                 (symmetric reg).

    The argument is taken modulo 1 first.
    """
    a = np.asarray(alpha, dtype=float)
    frac = np.mod(a, 1.0)
    # Symmetric regularisation: eta(0) = 0
    eta = np.where(np.abs(frac) < 1e-12, 0.0, 1.0 - 2.0 * frac)
    return eta


def z3_branch_twists(gamma_eff: float = 0.0) -> np.ndarray:
    """Effective twists for the three Z_3 branches.

    alpha_b = (b/3 + gamma_eff / (2 pi)) mod 1, for b = 0, 1, 2.
    """
    twists = np.array([b / 3.0 + gamma_eff / (2 * pi) for b in (0, 1, 2)])
    return np.mod(twists, 1.0)


def z3_branch_etas(gamma_eff: float = 0.0) -> np.ndarray:
    """eta-invariants on each of the three Z_3 branches."""
    return dirac_eta_invariant(z3_branch_twists(gamma_eff))


def z3_total_eta(gamma_eff: float = 0.0) -> float:
    """Sum of branch etas.

    Vanishes at gamma_eff = 0 (Z_3 sum cancels). Non-zero values
    indicate a net inflow required to cancel a gauge anomaly.
    """
    return float(np.sum(z3_branch_etas(gamma_eff)))


def axial_anomaly_density(
    F: np.ndarray, F_dual: np.ndarray
) -> np.ndarray:
    """Axial anomaly density: (1/16 pi^2) F_dual^{mu nu} F_{mu nu}.

    Parameters
    ----------
    F : (..., 4, 4) array
        Field strength tensor F_{mu nu}.
    F_dual : (..., 4, 4) array
        Hodge dual F-tilde^{mu nu} = (1/2) epsilon^{mu nu rho sigma} F_{rho sigma}.

    Returns
    -------
    (...) scalar density.
    """
    F = np.asarray(F, dtype=float)
    F_dual = np.asarray(F_dual, dtype=float)
    if F.shape[-2:] != (4, 4) or F_dual.shape[-2:] != (4, 4):
        raise ValueError("F and F_dual must have shape (..., 4, 4)")
    contracted = np.einsum("...ij,...ij->...", F_dual, F)
    return AXIAL_ANOMALY_COEFFICIENT * contracted


def callan_harvey_bulk_inflow(
    branch_etas: np.ndarray,
) -> float:
    """Bulk anomaly inflow required to close the Z_3 link.

    Callan-Harvey: integral_(bulk) d^4 x <anomaly>  =  -(1/2) Sum_b eta_b
    (the bulk inflow exactly cancels the boundary anomaly sum).

    Returns the *required bulk inflow* in units of integrated anomaly
    density. A zero result means the Z_3 link is anomaly-closed.
    """
    branch_etas = np.asarray(branch_etas, dtype=float)
    return -0.5 * float(np.sum(branch_etas))


def callan_harvey_consistency(branch_etas: np.ndarray, atol: float = 1e-12) -> bool:
    """True iff Sum_b eta_b = 0 (mod 2), i.e. the cover is anomaly-closed."""
    branch_etas = np.asarray(branch_etas, dtype=float)
    total = float(np.sum(branch_etas))
    return abs(np.mod(total + 1.0, 2.0) - 1.0) < atol


def index_density_2form(B_z: float, area: float = 1.0) -> float:
    """2D APS index density: (1/2 pi) integral F = q.

    For a constant magnetic field B_z (along the cylinder axis) on a
    region of area `area`, the integrated field strength flux is
    (1/2 pi) B_z * area = q (the Chern number).

    Connects to the angular Z_3 cover via the bulk-boundary relation:
    a unit of bulk flux q induces a shift of 1 in the branch index.
    """
    return float(B_z * area / (2 * pi))


def chern_simons_5form_coefficient() -> float:
    """5D Chern-Simons coefficient k such that

        S_CS = k * integral A wedge F wedge F

    cancels the 4D axial anomaly via Callan-Harvey inflow.

    Standard result: k = 1 / (24 pi^2).
    """
    return 1.0 / (24.0 * pi * pi)


def z3_anomaly_inflow_balance(
    gamma_eff: float, B_z: float, area: float
) -> dict:
    """Full balance: boundary etas + bulk inflow + Chern-Simons coefficient.

    Returns dict with
      - branch_twists
      - branch_etas
      - boundary_anomaly_sum (Sum_b eta_b)
      - required_bulk_inflow
      - bulk_flux_q (integrated F over area)
      - chern_simons_coeff
      - residual: deviation from anomaly closure (0 iff closed)
    """
    twists = z3_branch_twists(gamma_eff)
    etas = dirac_eta_invariant(twists)
    boundary_sum = float(np.sum(etas))
    required_inflow = -0.5 * boundary_sum
    flux_q = index_density_2form(B_z, area)
    cs_coeff = chern_simons_5form_coefficient()
    # Residual: difference between required inflow and the actual flux-induced
    # bulk anomaly contribution. Zero residual means the cover is closed.
    residual = required_inflow - cs_coeff * flux_q
    return {
        "branch_twists": twists,
        "branch_etas": etas,
        "boundary_anomaly_sum": boundary_sum,
        "required_bulk_inflow": required_inflow,
        "bulk_flux_q": flux_q,
        "chern_simons_coeff": cs_coeff,
        "residual": residual,
    }
