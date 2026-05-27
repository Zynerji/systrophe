"""Quantum-gravity probes for the Toroidal Knopp Drive.

The "next layer" suggested in `update.txt`: apply LQG area/volume
discretization + holographic complexity proxies to the *finite*
toroidal CTC band of `EffectiveToroidalKerrBinary`. Both probes are
ill-defined on the infinite Tipler cylinder (divergent volume, infinite
spin counts) but become meaningful on the binary-Kerr toroidal
realisation.

Two probe families
------------------

(A) **LQG area discretization** of the band's inner and outer
    boundary 2-surfaces. For each boundary (a topological 2-torus of
    proper area A), solve A_j = 8 pi gamma l_P^2 sqrt(j(j+1)) for the
    LQG spin j and report the rounded half-integer plus the relative
    error. Small j means the boundary is poorly resolved by the spin
    network -- a discrete-gravity analog of chronology protection
    (the band can't be "seen" by the underlying spinfoam).

(B) **Holographic complexity** of the band's bulk volume. Computes
    both the Complexity = Volume (CV) and Complexity = Action (CA)
    proxies on the finite toroidal slab, plus the Lloyd-bound growth
    rate dC/dt <= 2 E / pi hbar (E = 2 M total binary energy).

Honesty
-------
These are proxies, not exact AdS/CFT or full spinfoam calculations.
They are dimensionally consistent and reproduce the right scaling, but
the numerical prefactors should be treated as order-of-magnitude.
Their value is in giving *finite* quantum-gravity diagnostics for the
toroidal CTC band -- exactly the regime where the infinite-cylinder
versions of these probes diverge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary


# ----- constants ---------------------------------------------------------

# Reuse the same LQG constants as systrophe.quantum_info.lqg_discretization
# so the area unit matches across modules.
IMMIRZI = 0.2375
LPLANCK = 1.0  # natural units; rescale externally if needed


def planck_area_unit() -> float:
    """8 pi gamma l_P^2."""
    return float(8.0 * math.pi * IMMIRZI * LPLANCK ** 2)


def lqg_area_spectrum_at_j(j: float) -> float:
    """A_j = 8 pi gamma l_P^2 sqrt(j(j+1))."""
    if j < 0:
        raise ValueError(f"j must be non-negative, got {j}")
    return float(planck_area_unit() * math.sqrt(j * (j + 1.0)))


# ----- toroidal band geometry --------------------------------------------


def toroidal_band_boundary_area(
    binary: EffectiveToroidalKerrBinary, rho: float, L_z: Optional[float] = None,
) -> float:
    """Proper area of the (phi, z) 2-torus at fixed rho in the midplane.

    Linear approximation: A = 2 pi rho * L_z * sqrt(1 - 2 Phi(rho)).
    Default L_z = binary.d (use the binary separation as the axial
    extent of the toroidal slab).
    """
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    if L_z is None:
        L_z = binary.d
    if L_z <= 0:
        raise ValueError(f"L_z must be positive, got {L_z}")
    one_minus_two_Phi = 1.0 - 2.0 * binary.phi_newtonian(rho)
    if one_minus_two_Phi <= 0:
        # Inside strong-field core; proper area diverges.
        return float("inf")
    return float(2.0 * math.pi * rho * L_z * math.sqrt(one_minus_two_Phi))


def toroidal_band_volume(
    binary: EffectiveToroidalKerrBinary, L_z: Optional[float] = None,
    n_grid: int = 401,
) -> float:
    """Proper volume of the toroidal CTC band:
        V = int_{rho_in}^{rho_out} 2 pi rho * sqrt(1 - 2 Phi(rho)) drho * L_z.

    Returns 0 if no band exists.
    """
    if L_z is None:
        L_z = binary.d
    if L_z <= 0:
        raise ValueError(f"L_z must be positive, got {L_z}")
    edges = binary.ctc_band_edges(include_phi=False)
    if edges[0] is None:
        return 0.0
    rho_in, rho_out = edges
    rho_grid = np.linspace(rho_in, rho_out, n_grid)
    integrand = np.array([
        2.0 * math.pi * float(r)
        * math.sqrt(max(0.0, 1.0 - 2.0 * binary.phi_newtonian(float(r))))
        for r in rho_grid
    ])
    return float(np.trapezoid(integrand, rho_grid) * L_z)


# ----- (A) LQG area discretization ---------------------------------------


@dataclass(frozen=True)
class ToroidalBandLQG:
    """LQG-discretized boundary-area report for the toroidal CTC band."""
    rho_inner: Optional[float]
    rho_outer: Optional[float]
    A_inner_classical: float
    A_outer_classical: float
    j_inner: float
    j_outer: float
    A_inner_quantized: float
    A_outer_quantized: float
    rel_error_inner: float
    rel_error_outer: float
    # Discrete-gravity protection diagnostic
    band_width_proper: float       # rho_outer - rho_inner (linear LT)
    band_width_in_planck_units: float
    chronology_protected_by_discreteness: bool


def _spin_for_area(A_classical: float) -> tuple[float, float, float]:
    """Solve A_j = A_classical for j, round to half-integer.

    Returns (j_quant, A_quant, rel_err).
    """
    A_unit = planck_area_unit()
    if A_classical <= 0 or not math.isfinite(A_classical):
        return (0.0, 0.0, float("inf"))
    # j^2 + j = (A_classical / A_unit)^2
    disc = 1.0 + 4.0 * (A_classical / A_unit) ** 2
    j_cont = 0.5 * (-1.0 + math.sqrt(disc))
    j_quant = round(j_cont * 2) / 2.0
    A_quant = lqg_area_spectrum_at_j(j_quant)
    rel_err = abs(A_classical - A_quant) / max(A_classical, 1e-30)
    return float(j_quant), float(A_quant), float(rel_err)


def toroidal_band_lqg_discretization(
    binary: EffectiveToroidalKerrBinary, L_z: Optional[float] = None,
    discreteness_threshold_planck: float = 1.0,
) -> ToroidalBandLQG:
    """Apply LQG area quantization to the band's inner/outer boundaries.

    Parameters
    ----------
    discreteness_threshold_planck : float
        Band width (in Planck units) below which the band is judged
        "unresolvable" by the spinfoam, hence chronology-protected by
        gravitational discreteness.

    Returns
    -------
    ToroidalBandLQG with all classical and quantized values, plus the
    discrete-protection verdict.
    """
    edges = binary.ctc_band_edges(include_phi=False)
    if edges[0] is None:
        # No band: trivially "protected" (no band to discretize).
        return ToroidalBandLQG(
            rho_inner=None, rho_outer=None,
            A_inner_classical=0.0, A_outer_classical=0.0,
            j_inner=0.0, j_outer=0.0,
            A_inner_quantized=0.0, A_outer_quantized=0.0,
            rel_error_inner=float("inf"), rel_error_outer=float("inf"),
            band_width_proper=0.0,
            band_width_in_planck_units=0.0,
            chronology_protected_by_discreteness=True,
        )
    rho_in, rho_out = edges
    A_in = toroidal_band_boundary_area(binary, rho_in, L_z=L_z)
    A_out = toroidal_band_boundary_area(binary, rho_out, L_z=L_z)
    j_in, A_in_q, err_in = _spin_for_area(A_in)
    j_out, A_out_q, err_out = _spin_for_area(A_out)
    band_width = rho_out - rho_in
    band_width_planck = band_width / LPLANCK
    protected = band_width_planck < discreteness_threshold_planck
    return ToroidalBandLQG(
        rho_inner=rho_in, rho_outer=rho_out,
        A_inner_classical=A_in, A_outer_classical=A_out,
        j_inner=j_in, j_outer=j_out,
        A_inner_quantized=A_in_q, A_outer_quantized=A_out_q,
        rel_error_inner=err_in, rel_error_outer=err_out,
        band_width_proper=float(band_width),
        band_width_in_planck_units=float(band_width_planck),
        chronology_protected_by_discreteness=protected,
    )


# ----- (B) holographic complexity -----------------------------------------


@dataclass(frozen=True)
class ToroidalBandComplexity:
    """Holographic-complexity report for the toroidal CTC band."""
    band_volume: float
    cv_proxy: float                       # C ~ V / (G ell_AdS)
    ca_proxy: float                       # C ~ I_WdW / pi
    lloyd_growth_rate_max: float          # dC/dt <= 2 E / (pi hbar)
    binary_total_energy: float
    cv_traversal_time: Optional[float]    # tau = C_V / (dC/dt)_max


def volume_complexity_proxy(
    binary: EffectiveToroidalKerrBinary,
    G_newton: float = 1.0,
    ell_AdS: Optional[float] = None,
    L_z: Optional[float] = None,
) -> float:
    """C_V = V_band / (G_newton * ell_AdS).

    Defaults: G_newton = 1 (geometric units), ell_AdS = binary.d
    (use the binary separation as the AdS-like length scale).
    """
    if ell_AdS is None:
        ell_AdS = binary.d
    if ell_AdS <= 0:
        raise ValueError(f"ell_AdS must be positive, got {ell_AdS}")
    V = toroidal_band_volume(binary, L_z=L_z)
    return float(V / (G_newton * ell_AdS))


def action_complexity_proxy(
    binary: EffectiveToroidalKerrBinary,
    L_z: Optional[float] = None,
    n_grid: int = 401,
) -> float:
    """C_A ~ I_WdW / pi.

    Heuristic: I_WdW ~ integral of (Omega_eff(rho))^2 over the band
    volume (gravitomagnetic-energy proxy). This captures the right
    dimensional scaling without invoking the full AdS calculation.
    """
    if L_z is None:
        L_z = binary.d
    edges = binary.ctc_band_edges(include_phi=False)
    if edges[0] is None:
        return 0.0
    rho_in, rho_out = edges
    rho_grid = np.linspace(rho_in, rho_out, n_grid)
    integrand = np.array([
        binary.omega_eff(float(r)) ** 2
        * 2.0 * math.pi * float(r)
        * math.sqrt(max(0.0, 1.0 - 2.0 * binary.phi_newtonian(float(r))))
        for r in rho_grid
    ])
    I_WdW = float(np.trapezoid(integrand, rho_grid) * L_z)
    return float(I_WdW / math.pi)


def lloyd_growth_rate(
    binary: EffectiveToroidalKerrBinary, hbar: float = 1.0,
) -> float:
    """Lloyd bound on complexity growth rate.

        dC/dt  <=  2 E / (pi hbar),       E = 2 M  (total binary energy).
    """
    E_total = 2.0 * binary.M
    return float(2.0 * E_total / (math.pi * hbar))


def toroidal_band_complexity(
    binary: EffectiveToroidalKerrBinary,
    G_newton: float = 1.0,
    ell_AdS: Optional[float] = None,
    L_z: Optional[float] = None,
    hbar: float = 1.0,
) -> ToroidalBandComplexity:
    """Combined CV + CA complexity report for the toroidal CTC band."""
    V = toroidal_band_volume(binary, L_z=L_z)
    cv = volume_complexity_proxy(binary, G_newton=G_newton,
                                 ell_AdS=ell_AdS, L_z=L_z)
    ca = action_complexity_proxy(binary, L_z=L_z)
    dC_dt_max = lloyd_growth_rate(binary, hbar=hbar)
    if dC_dt_max > 0 and cv > 0:
        tau = float(cv / dC_dt_max)
    else:
        tau = None
    return ToroidalBandComplexity(
        band_volume=float(V),
        cv_proxy=float(cv),
        ca_proxy=float(ca),
        lloyd_growth_rate_max=float(dC_dt_max),
        binary_total_energy=float(2.0 * binary.M),
        cv_traversal_time=tau,
    )


# ----- combined diagnostics ----------------------------------------------


@dataclass(frozen=True)
class ToroidalQuantumDiagnostics:
    """LQG + holographic-complexity combined report."""
    lqg: ToroidalBandLQG
    complexity: ToroidalBandComplexity
    has_band: bool


def toroidal_quantum_diagnostics(
    binary: EffectiveToroidalKerrBinary,
    L_z: Optional[float] = None,
    discreteness_threshold_planck: float = 1.0,
    G_newton: float = 1.0,
    ell_AdS: Optional[float] = None,
    hbar: float = 1.0,
) -> ToroidalQuantumDiagnostics:
    """Full LQG + holographic-complexity diagnostic for the band."""
    lqg = toroidal_band_lqg_discretization(
        binary, L_z=L_z,
        discreteness_threshold_planck=discreteness_threshold_planck,
    )
    comp = toroidal_band_complexity(
        binary, G_newton=G_newton, ell_AdS=ell_AdS, L_z=L_z, hbar=hbar,
    )
    return ToroidalQuantumDiagnostics(
        lqg=lqg, complexity=comp,
        has_band=lqg.rho_inner is not None,
    )


def summarise_toroidal_quantum(d: ToroidalQuantumDiagnostics) -> str:
    """Human-readable summary string."""
    if not d.has_band:
        return ("Toroidal CTC band: NONE (subcritical binary)\n"
                "  LQG protection by trivial absence; complexity = 0.")
    lqg = d.lqg
    comp = d.complexity
    lines = [
        f"Toroidal CTC band: rho in [{lqg.rho_inner:.4f}, {lqg.rho_outer:.4f}]",
        f"  Band width:                 {lqg.band_width_proper:.4f}",
        f"  Band width (Planck units):  {lqg.band_width_in_planck_units:.4f}",
        f"  Protected by discreteness?  {lqg.chronology_protected_by_discreteness}",
        "",
        f"LQG boundary discretization:",
        f"  A_inner (classical) = {lqg.A_inner_classical:.4e}",
        f"  A_outer (classical) = {lqg.A_outer_classical:.4e}",
        f"  j_inner (quantized) = {lqg.j_inner:.1f}",
        f"  j_outer (quantized) = {lqg.j_outer:.1f}",
        f"  rel.err inner / outer: {lqg.rel_error_inner:.2e} / "
        f"{lqg.rel_error_outer:.2e}",
        "",
        f"Holographic complexity:",
        f"  V_band              = {comp.band_volume:.4e}",
        f"  C_V (volume proxy)  = {comp.cv_proxy:.4e}",
        f"  C_A (action proxy)  = {comp.ca_proxy:.4e}",
        f"  Lloyd growth bound  = {comp.lloyd_growth_rate_max:.4e}  (dC/dt max)",
        f"  CV-traversal time   = {comp.cv_traversal_time}",
    ]
    return "\n".join(lines)
