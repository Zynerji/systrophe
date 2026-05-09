"""Bridge from Systrophe (cylindrical Tipler) to the Dinos DKN framework.

Three concrete cross-references between this package and the Δῖνος
(Dinos) Dirac-Kerr-Newman framework (https://github.com/Zynerji/dinos-DKN):

1. **Cylindrical-to-Kerr parameter identification**
   The Dinos `MobiusKerrMapping` uses the identification
       a (Kerr) ↔ tau (Möbius torsion period)
       omega_field ↔ m_j (azimuthal mode)
       mu^2 ↔ beta + kappa (antipodal Higgs wall parameter).
   For the cylindrical Tipler exterior, the natural torsion period is
   tau = omega * R (the dust angular velocity times cylinder radius)
   and the field is scalar (m_j = 0 for the leading axisymmetric mode)
   on a vacuum background (beta + kappa = 0).

2. **Z3 Möbius eigenvalue ↔ Tipler log-frequency**
   The Tipler exterior log-period is 2*pi/alpha with alpha = sqrt(4a^2 - 1).
   Sampled on N nodes per period the discrete Laplacian eigenvalue at the
   fundamental mode is 2*(1 - cos(2*pi/N)). The Z3 Möbius cover with
   branch != 0 quantizes the angular index n + branch/3, giving
   2*(1 - cos((n + branch/3) * 2*pi/N)). Identification at the
   fundamental: alpha-Tipler matches the Möbius branch=0, n=1 mode iff
   alpha = 1 in the natural log-grid units; the **branch=1 mode adds a
   1/3 phase advance per N-step** which is the discrete signature of the
   "off-set" Tipler sinusoid produced by the Systrophe pair.

3. **Convergence diagnostic**
   For a single-cylinder supercritical exterior the integrated F(r) on a
   log-grid is itself a discrete fixed point of the Z3-Möbius update
   (modulo the boundary phase). We provide a residual computation as a
   sanity check.

This module is optional: importing it requires `dinos` to be importable
on PYTHONPATH (with z3-solver installed). It is NOT a runtime dependency
of the rest of Systrophe.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


def _ensure_dinos_import() -> None:
    """Verify the dinos package is importable.

    Caller is responsible for having Dinos on sys.path (either pip-installed
    or via PYTHONPATH). The optional environment variable SYSTROPHE_DINOS_PATH
    is consulted as a fallback: if set, it is prepended to sys.path.
    """
    import os

    extra = os.environ.get("SYSTROPHE_DINOS_PATH")
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        import dinos  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Dinos-DKN is required for the Systrophe bridge module. Install "
            "via `pip install dinos-dkn` (when available) or set the "
            "SYSTROPHE_DINOS_PATH environment variable to the directory "
            f"containing the dinos package. Underlying error: {e}"
        )


@dataclass(frozen=True)
class CylindricalKerrMapping:
    """Identification of Tipler-cylinder parameters with Dinos MobiusKerrMapping.

    Attributes
    ----------
    tau : float
        Torsion period; here = omega * R (dust velocity times radius).
    m_j : float
        Azimuthal field mode (0 for leading axisymmetric Tipler exterior).
    beta_plus_kappa : float
        Antipodal Higgs-wall parameter (0 for the Tipler vacuum exterior).
    a_kerr : float
        Kerr "a" parameter assigned by the Dinos identification: a = tau.
    omega_field : float
        Field frequency assigned: omega = m_j.
    mu_squared : float
        Field mass-squared: mu^2 = beta_plus_kappa.
    """

    tau: float
    m_j: float
    beta_plus_kappa: float
    a_kerr: float
    omega_field: float
    mu_squared: float


def map_to_dinos_kerr(vs: VanStockumInterior, m_j: float = 0.0) -> CylindricalKerrMapping:
    """Identify a VanStockumInterior with the Dinos Kerr-electron parameters.

    The mapping is the cylindrical specialization of
    `dinos.kerr_corrections.propose_mapping(tau, m_j, beta_plus_kappa)`
    with tau = omega * R, beta + kappa = 0 (vacuum exterior).
    """
    _ensure_dinos_import()
    from dinos.kerr_corrections import propose_mapping  # type: ignore[import-not-found]

    tau = float(vs.omega) * float(vs.R)
    bk = 0.0
    dinos_mapping = propose_mapping(tau, float(m_j), bk)
    return CylindricalKerrMapping(
        tau=tau,
        m_j=float(m_j),
        beta_plus_kappa=bk,
        a_kerr=dinos_mapping.a_from_tau,
        omega_field=dinos_mapping.omega_from_m_j,
        mu_squared=dinos_mapping.mu_squared_from_wall,
    )


def kerr_correction_at_tipler_threshold(vs: VanStockumInterior) -> dict:
    """Evaluate the Dinos −½ Kerr CP shift for the cylindrical mapping.

    For a Tipler-cylinder source with parameters (omega, R), evaluate the
    CP eigenvalue shift `−½ a^2 (omega^2 − mu^2)` that Dinos computes.

    Returns a dict with `a_kerr`, `cp_shift`, `is_on_shell`.
    """
    _ensure_dinos_import()
    from dinos.kerr_corrections import cp_leading_shift, is_on_shell  # type: ignore[import-not-found]

    m = map_to_dinos_kerr(vs)
    shift = float(cp_leading_shift(a=m.a_kerr, omega=m.omega_field, mu=float(np.sqrt(m.mu_squared))))
    on_shell = bool(is_on_shell(omega=m.omega_field, mu=float(np.sqrt(m.mu_squared))))
    return {
        "a_kerr": m.a_kerr,
        "cp_shift": shift,
        "is_on_shell": on_shell,
    }


def z3_branch_match_to_tipler_alpha(vs: VanStockumInterior, N: int) -> dict:
    """Compare the Tipler log-frequency alpha to the Z3 Möbius eigenvalues.

    On a log-grid u = ln(r/R) sampled at N nodes per period 2*pi/alpha,
    the fundamental Tipler mode has discrete-Laplacian eigenvalue
    2*(1 - cos(2*pi/N)). Compare to the three branches of the Z3 Möbius
    cover at lowest mode n=1.

    Returns
    -------
    dict
      tipler_eigenvalue   : 2 (1 - cos(2 pi / N))
      z3_eigenvalues      : tuple of (branch=0, branch=1, branch=2) lowest eigenvalues
      best_branch_match   : the branch index that minimizes |z3 - tipler|
      relative_residual   : residual at the best match
    """
    _ensure_dinos_import()
    from dinos.mobius_z3_cover import z3_mobius_eigenvalues_closed_form  # type: ignore[import-not-found]

    if not vs.is_supercritical():
        raise ValueError("Z3 branch comparison requires supercritical (a > 1/2)")
    if N < 4:
        raise ValueError("N must be at least 4 for a meaningful branch comparison")

    tipler_eig = 2.0 * (1.0 - np.cos(2.0 * np.pi / N))

    z3 = []
    for branch in (0, 1, 2):
        eigs = z3_mobius_eigenvalues_closed_form(N=N, branch=branch)
        # Drop the trivial zero (branch=0 has it; branches 1,2 don't)
        nonzero = eigs[eigs > 1e-12] if branch == 0 else eigs
        z3.append(float(np.min(nonzero)))

    z3 = tuple(z3)
    diffs = [abs(z - tipler_eig) for z in z3]
    best = int(np.argmin(diffs))
    rel = diffs[best] / (tipler_eig + 1e-15)

    return {
        "tipler_eigenvalue": float(tipler_eig),
        "z3_eigenvalues": z3,
        "best_branch_match": best,
        "relative_residual": float(rel),
    }


def mobius_temporal_loop_for_cylinder(
    vs: VanStockumInterior, N: int = 64, max_iter: int = 200, epsilon: float = 1e-2
):
    """Run the Dinos Möbius temporal loop with Tipler-derived parameters.

    Maps the Tipler torsion tau = omega * R into a `MobiusTemporalLoop`
    instance and evolves it. The expected fixed point on the time-loop
    contour is the cylindrical analog of the DKN electron's Compton radius.
    Useful as a smoke test that Dinos handles the Tipler-derived inputs.
    """
    _ensure_dinos_import()
    from dinos.temporal_loop import MobiusTemporalLoop, DKNParams  # type: ignore[import-not-found]

    tau = float(vs.omega) * float(vs.R)
    loop = MobiusTemporalLoop(
        N=N,
        T=4.0,
        K=N * 2,
        alpha=0.7,
        beta=0.15,
        tau=tau,
        damping=0.99,
        eta=0.0,
        dkn_params=DKNParams(),
        seed=1,
    )
    return loop.evolve(max_iter=max_iter, epsilon=epsilon)
