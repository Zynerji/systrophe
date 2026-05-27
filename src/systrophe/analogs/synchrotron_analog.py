"""Synchrotron radiation analog: charged orbits in LP supercritical bands.

A charged particle on a closed phi-orbit in a CTC band r in (r_n,
r_{n+1}) is the relativistic analog of a charged particle in a
synchrotron: it accelerates centripetally and (classically) emits
electromagnetic radiation along its tangent. For the LP background,
the orbital frequency is set by the angular metric structure, and
the analog "Lorentz factor" is built from F, K, L at r.

This module provides:

- orbital_frequency: omega_orbit(r) at a closed phi-orbit
- effective_gamma_factor: relativistic gamma at the orbit
- synchrotron_critical_frequency: omega_c = (3/2) gamma^3 omega_orbit
- emitted_power: P_total from Larmor-like formula
- characteristic_emission_band: spectral peak window
- multi_band_synchrotron_spectrum: emission from all CTC bands
- back_reaction_drag: radiation reaction on orbit
- observable_signature_distant_observer: redshift to far observer
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def orbital_frequency(vs: VanStockumInterior, r: float) -> float:
    """Coordinate orbital frequency d phi / dt for a circular orbit at r.

    For an axisymmetric metric, a circular geodesic satisfies
       Omega^2 = -dF/dr / (dL/dr) approximately (Killing-orbit case),
    or more generally from setting d/dr(g_tt + 2 Omega g_t_phi + Omega^2 g_pp) = 0.
    """
    eps = 1e-4 * max(abs(r), 1.0)
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    F_plus = float(vs.analytic_exterior_F(np.array([r + eps]))[0])
    F_minus = float(vs.analytic_exterior_F(np.array([r - eps]))[0])
    Fp = (F_plus - F_minus) / (2 * eps)

    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    K_plus = float(vs.analytic_exterior_K(np.array([r + eps]))[0])
    K_minus = float(vs.analytic_exterior_K(np.array([r - eps]))[0])
    Kp = (K_plus - K_minus) / (2 * eps)

    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    L_plus = float(vs.analytic_exterior_L(np.array([r + eps]))[0])
    L_minus = float(vs.analytic_exterior_L(np.array([r - eps]))[0])
    Lp = (L_plus - L_minus) / (2 * eps)

    # Geodesic condition: F' + 2 Omega K' + Omega^2 L' = 0
    # Solve quadratic in Omega
    if abs(Lp) < 1e-30:
        # Degenerate; fall back to F'/K' style
        if abs(Kp) < 1e-30:
            return float("nan")
        return float(-Fp / (2 * Kp))
    disc = 4 * Kp * Kp - 4 * Lp * Fp
    if disc < 0:
        return float("nan")
    Omega_plus = (-2 * Kp + math.sqrt(disc)) / (2 * Lp)
    return float(Omega_plus)


def effective_gamma_factor(vs: VanStockumInterior, r: float) -> float:
    """Relativistic gamma at a circular orbit.

    gamma^2 = 1 / (F - 2 Omega K - Omega^2 L) (approximately,
    interpreting -g_tt_effective as the time-time component along the
    orbit).
    """
    Omega = orbital_frequency(vs, r)
    if not math.isfinite(Omega):
        return float("nan")
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    denom = F - 2 * Omega * K - Omega * Omega * L
    if denom <= 0:
        return float("nan")
    return float(1.0 / math.sqrt(denom))


def synchrotron_critical_frequency(vs: VanStockumInterior, r: float) -> float:
    """omega_c = (3/2) gamma^3 omega_orbit (standard synchrotron spectrum peak)."""
    g = effective_gamma_factor(vs, r)
    Om = orbital_frequency(vs, r)
    if not (math.isfinite(g) and math.isfinite(Om)):
        return float("nan")
    return float(1.5 * g ** 3 * abs(Om))


def emitted_power(vs: VanStockumInterior, r: float, charge: float = 1.0) -> float:
    """Larmor-type radiated power: P = (2/3) e^2 a^2 gamma^4.

    Where a ~ Omega^2 r is the centripetal acceleration. Heuristic
    extension to the LP background.
    """
    Om = orbital_frequency(vs, r)
    g = effective_gamma_factor(vs, r)
    if not (math.isfinite(Om) and math.isfinite(g)):
        return float("nan")
    a = Om * Om * r
    P = (2.0 / 3.0) * charge ** 2 * a ** 2 * g ** 4
    return float(P)


def characteristic_emission_band(vs: VanStockumInterior, r: float) -> dict:
    """Spectral window for synchrotron-like emission at r."""
    Om = orbital_frequency(vs, r)
    g = effective_gamma_factor(vs, r)
    if not (math.isfinite(Om) and math.isfinite(g)):
        return {
            "omega_orbit": float("nan"),
            "omega_critical": float("nan"),
            "omega_min": float("nan"),
            "omega_max": float("nan"),
        }
    om_crit = 1.5 * g ** 3 * abs(Om)
    return {
        "omega_orbit": float(abs(Om)),
        "omega_critical": float(om_crit),
        "omega_min": float(abs(Om)),  # fundamental
        "omega_max": float(10.0 * om_crit),  # ~tail to 10 omega_c
        "gamma": float(g),
    }


def multi_band_synchrotron_spectrum(
    vs: VanStockumInterior, r_min: float = None, r_max: float = None,
    n_bands: int = 5,
) -> list[dict]:
    """Synchrotron-like emission per CTC band.

    Samples a representative r within each (r_{n}, r_{n+1}) band
    where F < 0 (CTC) and returns the emission band for each.
    """
    if not vs.is_supercritical():
        return []
    R = vs.R
    alpha = vs.alpha
    gamma_c = math.pi - math.atan(alpha)
    zeros = []
    for n in range(1, 200):
        u_n = (n * math.pi - gamma_c) / alpha
        if u_n <= 0:
            continue
        r_n = R * math.exp(u_n)
        zeros.append(r_n)
        if len(zeros) >= n_bands + 1:
            break
    out = []
    for i in range(min(n_bands, len(zeros) - 1)):
        r_mid = math.sqrt(zeros[i] * zeros[i + 1])
        # Only include if F < 0 (CTC band) at r_mid
        F_mid = float(vs.analytic_exterior_F(np.array([r_mid]))[0])
        if F_mid >= 0:
            continue
        emis = characteristic_emission_band(vs, r_mid)
        emis["band_index"] = i
        emis["r_inner_horizon"] = zeros[i]
        emis["r_outer_horizon"] = zeros[i + 1]
        emis["r_representative"] = r_mid
        out.append(emis)
    return out


def back_reaction_drag(
    vs: VanStockumInterior, r: float, charge: float = 1.0,
    mass: float = 1.0,
) -> float:
    """Radiation-reaction drag dE/dt / (energy).

    Heuristic: tau_brake^{-1} = P / (gamma m c^2). Returns
    1/tau_brake = P/(gamma m).
    """
    P = emitted_power(vs, r, charge)
    g = effective_gamma_factor(vs, r)
    if not (math.isfinite(P) and math.isfinite(g)):
        return float("nan")
    return float(P / max(g * mass, 1e-30))


def observable_signature_distant_observer(
    vs: VanStockumInterior, r_emit: float, r_obs: float = 100.0,
) -> dict:
    """Redshift the synchrotron critical frequency to a distant observer."""
    om_c = synchrotron_critical_frequency(vs, r_emit)
    F_e = float(vs.analytic_exterior_F(np.array([r_emit]))[0])
    F_o = float(vs.analytic_exterior_F(np.array([r_obs]))[0])
    if F_e <= 0 or F_o <= 0:
        return {
            "omega_observed": float("nan"),
            "redshift": float("nan"),
            "behind_CH": True,
        }
    z = math.sqrt(F_e / F_o)
    om_obs = om_c * z if math.isfinite(om_c) else float("nan")
    return {
        "omega_critical_at_source": om_c,
        "omega_observed": float(om_obs),
        "redshift_factor": float(z),
        "behind_CH": False,
    }
