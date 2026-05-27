"""Cosmological embedding: LP cylinder inside FRW background.

The standalone Lewis-Papapetrou exterior is asymptotically *not* flat
(it has log-periodic divergence as r -> infinity). For a physical
cylinder embedded in a cosmological spacetime, we must transition
from the LP interior/exterior to a matching FRW background at some
matching radius r_FRW.

We model this matched-asymptotic construction as:

  r in [0, R]:        van Stockum dust interior
  r in [R, r_match]:  Tipler LP exterior
  r > r_match:        FRW (k=0) with scale factor a_FRW(t)

The matching imposes that the induced 3-metric on r = r_match agrees
between LP and FRW. This is generally over-determined and only
approximately satisfiable; the residual is the "cosmological back-reaction"
on the cylinder.

We provide:
- frw_scale_factor: a_FRW(t) for matter-dominated, radiation-dominated,
  Lambda-dominated cases;
- matching_residual: how much the LP and FRW metrics disagree at r_match;
- cosmological_corrections_to_alpha: leading-order shift in Tipler
  log-frequency from FRW pressure;
- horizon_chirp_rate: how the chronology horizons drift with cosmic time
  due to Hubble expansion;
- cylinder_evaporates_under_expansion: True iff Hubble drag exceeds
  cylinder rotation by some characteristic time;
- inflation_creates_systrophe_cylinders: spontaneous creation rate of
  rotating-dust cylinders during inflation (heuristic).

The matched-asymptotic construction is approximate; numerical-relativity
evolution is the rigorous tool.
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def frw_scale_factor(
    cosmic_time: float,
    H0: float = 0.07,
    equation_of_state: str = "matter",
) -> float:
    """FRW scale factor a(t) for various w (equation-of-state)."""
    if equation_of_state == "matter":
        # a(t) = (3 H0 t / 2)^(2/3) for matter dominated
        if cosmic_time <= 0:
            return 0.0
        return float((1.5 * H0 * cosmic_time) ** (2.0 / 3.0))
    if equation_of_state == "radiation":
        # a(t) = (2 H0 t)^(1/2) for radiation dominated
        if cosmic_time <= 0:
            return 0.0
        return float((2.0 * H0 * cosmic_time) ** 0.5)
    if equation_of_state == "Lambda" or equation_of_state == "deSitter":
        # a(t) = exp(H0 t) for Lambda-dominated
        return float(math.exp(H0 * cosmic_time))
    raise ValueError(f"Unknown equation_of_state: {equation_of_state}")


def hubble_parameter(
    cosmic_time: float,
    H0: float = 0.07,
    equation_of_state: str = "matter",
) -> float:
    """H(t) = a_dot / a for various EoS."""
    if equation_of_state == "matter":
        if cosmic_time <= 0:
            return float("inf")
        return float(2.0 / (3.0 * cosmic_time))
    if equation_of_state == "radiation":
        if cosmic_time <= 0:
            return float("inf")
        return float(1.0 / (2.0 * cosmic_time))
    if equation_of_state in ("Lambda", "deSitter"):
        return float(H0)
    raise ValueError(f"Unknown equation_of_state: {equation_of_state}")


def matching_residual(
    vs: VanStockumInterior, r_match: float,
    cosmic_time: float = 1.0,
    H0: float = 0.07,
    equation_of_state: str = "matter",
) -> dict:
    """Mismatch between LP induced metric and FRW at r = r_match."""
    F_LP = float(vs.analytic_exterior_F(np.array([r_match]))[0])
    L_LP = float(vs.analytic_exterior_L(np.array([r_match]))[0])
    a_t = frw_scale_factor(cosmic_time, H0, equation_of_state)
    F_FRW = 1.0  # FRW has g_tt = -1 in comoving coords
    L_FRW = a_t ** 2 * r_match ** 2  # FRW phi-phi component at r_match

    f_resid = float(abs(F_LP - F_FRW))
    l_resid = float(abs(L_LP - L_FRW))
    return {
        "r_match": r_match,
        "cosmic_time": cosmic_time,
        "F_residual": f_resid,
        "L_residual": l_resid,
        "F_LP": F_LP, "L_LP": L_LP,
        "F_FRW": F_FRW, "L_FRW": L_FRW,
        "scale_factor": a_t,
    }


def cosmological_corrections_to_alpha(
    vs: VanStockumInterior,
    cosmic_time: float = 1.0,
    H0: float = 0.07,
    equation_of_state: str = "matter",
) -> dict:
    """Leading FRW correction to the Tipler log-frequency alpha.

    Heuristic: H(t) acts as an "effective rotation drag":
    omega_eff(t) = omega * (1 - H(t) * tau_drag) where tau_drag is
    set to 1/omega for dimensional consistency.
    The corrected alpha is sqrt(4 (omega_eff R)^2 - 1) if supercritical.
    """
    H_t = hubble_parameter(cosmic_time, H0, equation_of_state)
    tau_drag = 1.0 / max(vs.omega, 1e-30)
    omega_eff = vs.omega * (1.0 - H_t * tau_drag)
    a_eff = omega_eff * vs.R
    if a_eff <= 0.5:
        return {
            "alpha_bare": vs.alpha if vs.is_supercritical() else 0.0,
            "alpha_corrected": 0.0,
            "still_supercritical": False,
            "omega_eff": float(omega_eff),
            "a_eff": float(a_eff),
        }
    alpha_corr = float(math.sqrt(4 * a_eff * a_eff - 1))
    return {
        "alpha_bare": vs.alpha if vs.is_supercritical() else 0.0,
        "alpha_corrected": alpha_corr,
        "still_supercritical": True,
        "omega_eff": float(omega_eff),
        "a_eff": float(a_eff),
    }


def horizon_chirp_rate(
    vs: VanStockumInterior, cosmic_time: float = 1.0,
    H0: float = 0.07, equation_of_state: str = "matter",
) -> dict:
    """How chronology horizons drift in r as cosmic time advances."""
    if not vs.is_supercritical():
        return {
            "first_CH_drift_rate": 0.0,
            "regime": "subcritical",
        }
    # First CH at r_1 = R exp((pi - gamma)/alpha)
    gamma = math.pi - math.atan(vs.alpha)
    r1_bare = vs.R * math.exp((math.pi - gamma) / vs.alpha)
    # Corrected alpha
    cosm = cosmological_corrections_to_alpha(vs, cosmic_time, H0, equation_of_state)
    if not cosm["still_supercritical"]:
        return {
            "first_CH_drift_rate": float("inf"),
            "regime": "evaporated",
            "alpha_corrected": 0.0,
        }
    alpha_c = cosm["alpha_corrected"]
    gamma_c = math.pi - math.atan(alpha_c)
    r1_c = vs.R * math.exp((math.pi - gamma_c) / alpha_c)
    # Time derivative (finite-diff)
    eps = max(1e-3 * cosmic_time, 1e-9)
    cosm_plus = cosmological_corrections_to_alpha(
        vs, cosmic_time + eps, H0, equation_of_state)
    if not cosm_plus["still_supercritical"]:
        return {
            "first_CH_drift_rate": float("inf"),
            "regime": "evaporating",
        }
    alpha_p = cosm_plus["alpha_corrected"]
    gamma_p = math.pi - math.atan(alpha_p)
    r1_p = vs.R * math.exp((math.pi - gamma_p) / alpha_p)
    drift = (r1_p - r1_c) / eps
    return {
        "r1_bare": float(r1_bare),
        "r1_corrected": float(r1_c),
        "first_CH_drift_rate": float(drift),
        "regime": "supercritical",
        "alpha_corrected": alpha_c,
    }


def cylinder_evaporates_under_expansion(
    vs: VanStockumInterior, equation_of_state: str = "matter",
    H0: float = 0.07,
    time_horizon: float = 100.0,
) -> dict:
    """Time scale on which a_eff drops below 0.5 (CTCs extinguish)."""
    if not vs.is_supercritical():
        return {
            "evaporates": False,
            "reason": "already subcritical",
            "time_to_evaporate": None,
        }
    t_grid = np.linspace(0.01, time_horizon, 1000)
    for t in t_grid:
        cosm = cosmological_corrections_to_alpha(vs, float(t), H0, equation_of_state)
        if not cosm["still_supercritical"]:
            return {
                "evaporates": True,
                "time_to_evaporate": float(t),
                "equation_of_state": equation_of_state,
            }
    return {
        "evaporates": False,
        "time_to_evaporate": None,
        "equation_of_state": equation_of_state,
    }


def inflation_creates_systrophe_cylinders(
    H_inflation: float = 1e-6,  # Planck units
    omega_mean: float = 0.5,  # mean rotation, Planck units
    R_mean: float = 1.0,  # mean radius
) -> dict:
    """Heuristic spontaneous creation rate of rotating-dust cylinders during inflation.

    Treats inflation as a Hawking-like vacuum: per-Hubble-volume per
    Hubble-time creation of bound rotating-dust configurations.

    The rate is dimensionally:
       N_create ~ H^4 * exp(-S_E)
    where S_E is the Euclidean action of the formed cylinder ~ M_cyl/H.
    """
    M_cyl = math.pi * R_mean ** 2 * omega_mean ** 2  # rough dust energy
    S_E = M_cyl / max(H_inflation, 1e-30)
    rate = float(H_inflation ** 4 * math.exp(-min(S_E, 700.0)))
    return {
        "H_inflation": H_inflation,
        "M_cylinder_estimate": M_cyl,
        "euclidean_action": float(S_E),
        "creation_rate_per_H4": rate,
        "exponentially_suppressed": S_E > 50,
    }
