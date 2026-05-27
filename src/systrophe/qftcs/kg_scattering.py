"""Klein-Gordon scattering across chronology horizons.

A massive scalar phi on the LP exterior satisfies
   (Box - m^2) phi = 0.
Decomposing as phi = e^{-i omega t} e^{i n phi_var} chi(r), the
radial equation becomes a Schrodinger-like equation:
   chi'' + V_eff(r; omega, n, m) chi = 0,
where V_eff incorporates F, K, L, and the mode quantum numbers.

This module computes:

- effective_potential: V_eff(r) for given (omega, n, m)
- transmission_coefficient: |T|^2 across a chronology horizon
- reflection_coefficient: |R|^2 (= 1 - |T|^2 for unitary scattering)
- transmission_at_multiple_CH: per-horizon T for cascade transmission
- bound_states_between_CHs: discrete spectrum between two chronology
  horizons (analog of "quasi-bound" Tipler states)
- superradiance_test: whether a mode amplifies after CH crossing
  (analog of Kerr superradiance)

The effective potential near a chronology horizon (F -> 0) has a
1/F singularity; we regularize by adding a small imaginary part to
F (analog of the i*epsilon prescription).
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def effective_potential(
    vs: VanStockumInterior, r: float,
    omega: float, n_phi: int = 0, mass: float = 0.0,
) -> float:
    """V_eff(r) for the radial KG equation.

    For the LP metric, the radial equation has the schematic form
       chi'' + (omega^2 / F_eff - mass^2 - n^2 / L_eff) chi = 0
    where F_eff and L_eff are F and L with a small regularization.
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    # Combined effective F including K (frame dragging) shift
    F_combined = F - 2 * omega * K * n_phi / max(omega, 1e-30) - n_phi * n_phi * K * K / max(omega * L, 1e-30)
    F_reg = F_combined if abs(F_combined) > 1e-6 else math.copysign(1e-6, F_combined or 1)
    angular_centrifugal = (n_phi * n_phi) / max(abs(L), 1e-6)
    V = (omega * omega) / F_reg - mass * mass - angular_centrifugal
    return float(V)


def transmission_coefficient(
    vs: VanStockumInterior, r_left: float, r_right: float,
    omega: float, n_phi: int = 0, mass: float = 0.0,
    n_steps: int = 500,
) -> float:
    """WKB-style transmission |T|^2 across (r_left, r_right).

    T ~ exp(-2 integral sqrt|-V_eff| dr) when V_eff is negative
    (classical turning region).
    """
    if r_right <= r_left:
        raise ValueError("r_right must exceed r_left")
    r_grid = np.linspace(r_left, r_right, n_steps)
    dr = (r_right - r_left) / (n_steps - 1)
    log_T = 0.0
    for r in r_grid:
        V = effective_potential(vs, float(r), omega, n_phi, mass)
        if V < 0:
            log_T -= math.sqrt(-V) * dr
    return float(math.exp(2 * log_T))


def reflection_coefficient(
    vs: VanStockumInterior, r_left: float, r_right: float,
    omega: float, n_phi: int = 0, mass: float = 0.0,
) -> float:
    """|R|^2 = 1 - |T|^2 (unitarity, in this WKB approximation)."""
    T = transmission_coefficient(vs, r_left, r_right, omega, n_phi, mass)
    return float(max(0.0, 1.0 - T))


def transmission_at_multiple_CH(
    vs: VanStockumInterior, omega: float,
    n_phi: int = 0, mass: float = 0.0,
    n_horizons: int = 3,
) -> list[dict]:
    """Per-CH transmission coefficient for a cascade.

    Returns transmission across each of the first n_horizons F=0 surfaces.
    """
    if not vs.is_supercritical():
        return []
    R = vs.R
    alpha = vs.alpha
    gamma = math.pi - math.atan(alpha)
    zeros = []
    for n in range(1, 200):
        u_n = (n * math.pi - gamma) / alpha
        if u_n <= 0:
            continue
        r_n = R * math.exp(u_n)
        zeros.append(r_n)
        if len(zeros) >= n_horizons + 1:
            break
    out = []
    cumulative_T = 1.0
    for i in range(min(n_horizons, len(zeros) - 1)):
        # Transmission across the i-th CH at r = zeros[i] (small width around it)
        eps_r = 0.05 * zeros[i]
        T_i = transmission_coefficient(
            vs, zeros[i] - eps_r, zeros[i] + eps_r,
            omega, n_phi, mass,
        )
        cumulative_T *= T_i
        out.append({
            "horizon_index": i,
            "r_horizon": zeros[i],
            "single_horizon_T": T_i,
            "cumulative_T": cumulative_T,
        })
    return out


def bound_states_between_CHs(
    vs: VanStockumInterior, r_left: float, r_right: float,
    omega_range: tuple[float, float] = (0.1, 5.0),
    n_omega: int = 50, n_phi: int = 0, mass: float = 0.5,
) -> list[float]:
    """Find quasi-bound state frequencies in the CTC band (r_left, r_right).

    Heuristic: scan omega; count sign changes of integrated V_eff.
    """
    omegas = np.linspace(omega_range[0], omega_range[1], n_omega)
    r_mid_grid = np.linspace(r_left, r_right, 100)
    bound_omegas = []
    for omega in omegas:
        V_vals = [effective_potential(vs, float(r), float(omega), n_phi, mass)
                  for r in r_mid_grid]
        # Quasi-bound if V<0 over most of (r_left, r_right)
        if sum(1 for v in V_vals if v < 0) > 0.7 * len(V_vals):
            bound_omegas.append(float(omega))
    return bound_omegas


def superradiance_test(
    vs: VanStockumInterior, r_inside_CH: float,
    omega: float, n_phi: int = 1, mass: float = 0.0,
) -> dict:
    """Test whether mode (omega, n_phi) is in the superradiant regime.

    In Kerr, omega < n_phi * Omega_H triggers amplification. For
    LP, the analog uses Omega_H(r) = K(r)/L(r) at the chronology
    horizon.
    """
    K = float(vs.analytic_exterior_K(np.array([r_inside_CH]))[0])
    L = float(vs.analytic_exterior_L(np.array([r_inside_CH]))[0])
    if abs(L) < 1e-30:
        Omega_H = float("inf")
    else:
        Omega_H = K / L
    threshold = n_phi * Omega_H
    is_superradiant = (omega > 0) and (omega < abs(threshold))
    return {
        "r": r_inside_CH,
        "Omega_H": float(Omega_H),
        "n_phi": int(n_phi),
        "threshold": float(threshold),
        "omega": float(omega),
        "is_superradiant": bool(is_superradiant),
    }


def cascade_transmission_total(
    vs: VanStockumInterior, omega: float,
    n_phi: int = 0, mass: float = 0.0,
    n_horizons: int = 3,
) -> float:
    """Cumulative transmission across n_horizons chronology horizons."""
    res = transmission_at_multiple_CH(vs, omega, n_phi, mass, n_horizons)
    if not res:
        return 1.0
    return float(res[-1]["cumulative_T"])
