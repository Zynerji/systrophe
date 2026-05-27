"""Light scalar field coupled to LP geometry as dark-matter analog.

Treat the LP cylinder as immersed in a uniform dark-matter scalar
phi_DM(t, x) with a Lagrangian
   L_DM = -1/2 d_mu phi_DM d^mu phi_DM - 1/2 m_DM^2 phi_DM^2.

For an ultra-light scalar DM (m_DM ~ 10^-22 eV in fuzzy-DM models),
the de Broglie wavelength is comparable to galactic scales. The
*coherent* DM field around a Systrophe cylinder takes the form of a
soliton with profile phi(r) determined by the cylinder's gravity.

Key effects:

- DM-induced shift of orbital frequencies for objects orbiting the
  cylinder (fuzzy-DM "wave drag");
- DM superradiance: amplification of phi_DM modes by the rotating
  cylinder (analog of Kerr black hole superradiance);
- DM-cylinder coupling: phi_DM modifies the effective alpha;
- DM-induced CTC stability: DM pressure can either reinforce or
  evaporate CTCs.

This module provides:
- compton_wavelength: from mass
- DM_density_profile_around_cylinder: phi^2(r) profile
- DM_drag_on_orbits: orbital-frequency shift
- DM_superradiance_growth_rate: mode growth from cylinder rotation
- effective_alpha_with_DM_pressure: corrected alpha
- DM_induced_CTC_lifetime: decoherence time of CTCs from DM noise
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior

# Planck-units conversion constants (rough)
EV_TO_INVERSE_LENGTH = 5.07e6  # 1 eV ~ 5e6 m^-1 (h-bar c)


def compton_wavelength(mass_eV: float) -> float:
    """Compton wavelength in meters from mass in eV."""
    if mass_eV <= 0:
        return float("inf")
    return float(1.0 / (mass_eV * EV_TO_INVERSE_LENGTH))


def DM_density_profile_around_cylinder(
    vs: VanStockumInterior, r_grid: np.ndarray,
    DM_mass_eV: float = 1e-22, central_density: float = 1.0,
) -> dict:
    """Soliton-like DM density profile rho_DM(r) around the cylinder.

    For ultra-light DM, the profile satisfies a Schrodinger-Poisson
    system. We use the analytic fuzzy-DM soliton:
       rho(r) = rho_c / (1 + (r/r_sol)^2)^8
    with r_sol ~ Compton wavelength of the cylinder mass.
    """
    M_cyl = math.pi * vs.R ** 2 * vs.omega ** 2  # rough cylinder energy
    r_sol = compton_wavelength(DM_mass_eV) * (1.0 / max(M_cyl, 1e-30))
    rho = central_density / (1 + (r_grid / max(r_sol, 1e-30)) ** 2) ** 8
    return {
        "r_grid": r_grid,
        "rho_DM": rho,
        "r_soliton": float(r_sol),
        "central_density": central_density,
        "DM_mass_eV": DM_mass_eV,
    }


def DM_drag_on_orbits(
    vs: VanStockumInterior, r_orbit: float,
    DM_mass_eV: float = 1e-22, DM_density: float = 1.0,
) -> dict:
    """Shift to orbital frequency from DM coupling.

    Heuristic: delta Omega ~ -G rho_DM(r_orbit) / Omega_0 for slow
    rotators; for ultra-light DM the shift is at most ~10^{-25} of Omega.
    """
    profile = DM_density_profile_around_cylinder(
        vs, np.array([r_orbit]), DM_mass_eV=DM_mass_eV,
        central_density=DM_density,
    )
    rho = float(profile["rho_DM"][0])
    Omega_0 = vs.omega
    delta_omega = -rho / max(Omega_0, 1e-30)
    return {
        "r_orbit": r_orbit,
        "DM_density_at_orbit": rho,
        "Omega_0": Omega_0,
        "delta_Omega": float(delta_omega),
        "relative_shift": float(abs(delta_omega) / max(Omega_0, 1e-30)),
    }


def DM_superradiance_growth_rate(
    vs: VanStockumInterior, DM_mass_eV: float = 1e-22,
    n_phi: int = 1,
) -> dict:
    """Growth rate of DM superradiant modes around the cylinder.

    For Kerr, the maximum growth rate is ~10^{-7} Omega_H. For LP,
    the analog uses Omega_H = K(R)/L(R) at the cylinder surface.

    The mode condition: omega_mode = m_DM, and superradiance requires
    omega_mode < n_phi * Omega_H.
    """
    K_R = float(vs.analytic_exterior_K(np.array([vs.R * 1.001]))[0])
    L_R = float(vs.analytic_exterior_L(np.array([vs.R * 1.001]))[0])
    Omega_H = K_R / max(abs(L_R), 1e-30)
    omega_mode = DM_mass_eV * EV_TO_INVERSE_LENGTH  # in 1/m
    threshold = n_phi * abs(Omega_H)
    is_superradiant = omega_mode < threshold
    growth_rate_estimate = 1e-7 * abs(Omega_H) if is_superradiant else 0.0
    return {
        "Omega_H": float(Omega_H),
        "n_phi": int(n_phi),
        "omega_mode": float(omega_mode),
        "threshold": float(threshold),
        "is_superradiant": bool(is_superradiant),
        "growth_rate_estimate": float(growth_rate_estimate),
    }


def effective_alpha_with_DM_pressure(
    vs: VanStockumInterior, DM_pressure: float = 0.0,
) -> dict:
    """Modified alpha when DM pressure shifts the dust effective rotation.

    Heuristic: omega_eff = omega * sqrt(1 + P_DM/rho_dust). For
    positive pressure, rotation effectively decreases; negative
    pressure (DM "tension") can increase it.
    """
    rho_dust = vs.omega ** 2  # rough
    factor = math.sqrt(max(1.0 + DM_pressure / max(rho_dust, 1e-30), 1e-30))
    omega_eff = vs.omega * factor
    a_eff = omega_eff * vs.R
    if a_eff <= 0.5:
        return {
            "alpha_bare": vs.alpha if vs.is_supercritical() else 0.0,
            "alpha_corrected": 0.0,
            "still_supercritical": False,
            "a_eff": float(a_eff),
        }
    alpha_corr = float(math.sqrt(4 * a_eff * a_eff - 1))
    return {
        "alpha_bare": vs.alpha if vs.is_supercritical() else 0.0,
        "alpha_corrected": alpha_corr,
        "still_supercritical": True,
        "a_eff": float(a_eff),
    }


def DM_induced_CTC_lifetime(
    vs: VanStockumInterior, DM_density: float = 1.0,
    DM_mass_eV: float = 1e-22,
) -> dict:
    """Decoherence lifetime of a CTC from DM noise.

    Heuristic: tau_decoherence = 1 / (n_DM * sigma_xs * v) where
    sigma_xs is the cross-section for DM-cylinder scattering. For
    ultra-light DM, sigma_xs is tiny and tau is enormous.
    """
    rho = DM_density
    # Cross section ~ (R / Compton)^2 (very small for ultra-light DM)
    lambda_C = compton_wavelength(DM_mass_eV)
    sigma_xs = (vs.R / max(lambda_C, 1e-30)) ** 2
    v_DM = 1e-3  # typical halo velocity ~ 300 km/s in c units
    rate = rho * sigma_xs * v_DM
    tau = 1.0 / max(rate, 1e-30)
    return {
        "DM_density": rho,
        "DM_mass_eV": DM_mass_eV,
        "cross_section_estimate": float(sigma_xs),
        "rate_estimate": float(rate),
        "lifetime_estimate": float(tau),
        "lifetime_exceeds_Hubble": tau > 1e17,  # ~ Hubble time in seconds
    }
