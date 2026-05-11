"""5D Kaluza-Klein embedding of the LP cylinder.

Embed the 4D LP metric in 5D with a compact extra dimension xi
of radius L_xi:
   ds_5^2 = ds_LP^2 + L_xi^2 * dxi^2
with xi ~ xi + 2pi. KK reduction yields:
- a 4D gravity sector (the original LP metric);
- a U(1) gauge field A_mu (= g_{xi mu}, zero in this trivial embedding);
- a dilaton phi from g_{xi xi}.

For non-trivial embeddings, allow the cylinder to *thread* the extra
dimension (KK monopole-like) or carry KK charge. This module
explores both cases.

Effects:
- KK tower of modes for any 4D field: each n contributes mass
  m_n = n / L_xi;
- Cylinder mass spectrum shifted; alpha_eff(L_xi) emerges;
- Heavy KK modes thermalize at high T;
- CTCs in 5D may be evaded by moving in xi.

Module provides:
- five_d_metric: 5D LP metric
- kk_tower_masses: m_n = n / L_xi
- kk_reduced_alpha: alpha as function of L_xi
- xi_circulation_evades_CTC: whether motion in xi can avoid CTCs
- kk_monopole_charge: U(1) magnetic charge from non-trivial fiber
- compactification_radius_from_observations: bound L_xi from observation
"""

from __future__ import annotations

import math

import numpy as np

from .vanstockum import VanStockumInterior


def five_d_metric(vs: VanStockumInterior, r: float,
                   xi_radius: float = 1.0) -> np.ndarray:
    """5x5 metric tensor at (r, *, *, xi_anywhere).

    Order: (t, r, phi, z, xi).
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    g = np.zeros((5, 5))
    g[0, 0] = -F
    g[0, 2] = K
    g[2, 0] = K
    g[2, 2] = L
    g[1, 1] = 1.0  # h = 1 placeholder
    g[3, 3] = 1.0
    g[4, 4] = xi_radius * xi_radius
    return g


def kk_tower_masses(L_xi: float, n_max: int = 10) -> np.ndarray:
    """KK mass tower m_n = n / L_xi for n = 1, ..., n_max."""
    if L_xi <= 0:
        raise ValueError("L_xi must be positive")
    n = np.arange(1, n_max + 1)
    return n / L_xi


def kk_reduced_alpha(vs: VanStockumInterior, L_xi: float) -> float:
    """KK reduction shifts effective alpha. Heuristic:
    alpha_eff = alpha * sqrt(1 + R^2/L_xi^2).

    When L_xi >> R, effect is small. When L_xi ~ R, alpha increases
    (cylinder rotates "harder" relative to the KK scale).
    """
    if not vs.is_supercritical():
        return 0.0
    factor = math.sqrt(1.0 + (vs.R / max(L_xi, 1e-30)) ** 2)
    return float(vs.alpha * factor)


def xi_circulation_evades_CTC(
    vs: VanStockumInterior, r_in_CTC: float, L_xi: float,
) -> dict:
    """Test whether motion in xi at constant r can avoid a CTC.

    A worldline that moves rapidly in xi has (dxi/dtau) > 0 contributing
    +L_xi^2 (dxi/dtau)^2 to the 5D line element. For a phi-loop at r,
    the 4D line element contribution is (R) F < 0 (CTC). The 5D
    timelike condition requires that the *full* 5D dot-product be
    negative:
       -F + L_xi^2 * (dxi/d phi)^2 < 0.
    If L_xi^2 * (dxi/d phi)^2 < F (positive), then the 5D path is
    spacelike (avoiding the CTC).

    Returns the minimum dxi/dphi needed to convert the CTC to a
    spacelike path.
    """
    F = float(vs.analytic_exterior_F(np.array([r_in_CTC]))[0])
    if F >= 0:
        return {
            "r": r_in_CTC, "F": F, "regime": "non-CTC",
            "min_dxi_dphi": 0.0,
            "evadable": False,  # no CTC to evade
        }
    # Need L_xi^2 (dxi/dphi)^2 > -F (positive when F<0)
    min_velocity = math.sqrt(-F) / L_xi
    return {
        "r": r_in_CTC,
        "F": F,
        "regime": "CTC",
        "min_dxi_dphi": float(min_velocity),
        "L_xi": L_xi,
        "evadable": True,
    }


def kk_monopole_charge(
    vs: VanStockumInterior, L_xi: float = 1.0, n_twist: int = 1,
) -> dict:
    """Magnetic monopole charge from non-trivial KK fiber bundle.

    A Hopf-style fiber bundle with twist n over the cylinder geometry
    gives a U(1) magnetic charge proportional to n / L_xi.
    """
    if L_xi <= 0:
        raise ValueError("L_xi must be positive")
    charge = n_twist / L_xi
    dirac_quantum = 2 * math.pi  # in natural units
    n_dirac_units = charge * L_xi / dirac_quantum
    return {
        "n_twist": int(n_twist),
        "L_xi": L_xi,
        "magnetic_charge": float(charge),
        "n_dirac_units": float(n_dirac_units),
    }


def compactification_radius_from_observations(
    bound_KK_mass_eV: float = 1e-3,
) -> dict:
    """Compute L_xi compatible with a given KK-tower upper-bound mass.

    If the lightest KK mode m_1 < bound_KK_mass_eV is not observed,
    then m_1 > bound => L_xi < 1/bound.
    """
    if bound_KK_mass_eV <= 0:
        raise ValueError("bound must be positive")
    # m_1 = 1/L_xi (in natural units)
    L_xi_max = 1.0 / bound_KK_mass_eV
    return {
        "bound_KK_mass_eV": bound_KK_mass_eV,
        "L_xi_max_natural_units": float(L_xi_max),
        "L_xi_max_meters_estimate": float(L_xi_max / 5.07e6),  # eV^-1 -> m
    }


def five_d_geodesic_drift_in_xi(
    vs: VanStockumInterior, r: float, L_xi: float,
    dxi_dphi: float,
) -> dict:
    """Net drift in xi per phi-revolution along a closed phi-loop.

    For a particle with constant dxi/dphi, the drift after one
    revolution is 2 pi * dxi/dphi. Returns this and the residual
    proper time per revolution.
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    # 4D contribution: L per dphi^2; 5D adds L_xi^2 (dxi/dphi)^2 per dphi^2
    g_eff_pp = L + L_xi ** 2 * dxi_dphi ** 2
    # Proper time per phi revolution
    if F >= 0:
        # Non-CTC: dtau^2 = g_eff_pp dphi^2 - F (dt/dphi)^2; need (dt/dphi) info
        # Simpler: just report g_eff_pp
        regime = "non-CTC"
    else:
        # CTC: 4D g_pp = L < 0 might still hold; 5D corrects
        regime = "CTC-in-4D"
    drift = 2 * math.pi * dxi_dphi
    return {
        "r": r,
        "L_xi": L_xi,
        "dxi_dphi": dxi_dphi,
        "drift_per_revolution": float(drift),
        "regime": regime,
        "g_eff_pp_5d": float(g_eff_pp),
    }
