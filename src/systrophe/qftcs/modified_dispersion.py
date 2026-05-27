"""Modified dispersion relations on the LP background.

Some Lorentz-violating QG models predict
    E^2 = p^2 c^2 + m^2 c^4 + xi * (p c / E_QG)^n * p^2 c^2
where E_QG is a UV scale (~Planck) and xi, n parameterize the LV.
A non-trivial LV on the LP background would show up as:
- Differential photon arrival times across CTC bands
- Modified Hawking-spectrum cutoff at high frequency
- Velocity-of-light dependence on energy
- Modified threshold reactions (e.g., GZK)

This module computes:
- modified_speed_of_light: v_eff(E) at given UV cutoff
- arrival_time_delay: differential arrival across an LP path
- LP_threshold_modification: shifts in reaction thresholds
- birefringence_test: helicity-dependent v(E)
- vacuum_birefringence_in_supercritical_band
- LV_constraint_from_GRB: bound xi from observations

References
----------
- Amelino-Camelia 2013 (LRR 16, 5): DSR
- Liberati 2013 (CQG 30, 133001): tests of LI
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def modified_speed_of_light(
    energy: float, E_QG: float = 1.22e19,  # Planck in GeV
    xi: float = 1.0, n: int = 1,
) -> float:
    """v_eff = c * (1 - xi/2 * (E/E_QG)^n) for sub-Planckian E."""
    if energy <= 0:
        return 1.0
    correction = xi / 2 * (energy / E_QG) ** n
    return float(1.0 - correction)


def arrival_time_delay(
    energy_low: float, energy_high: float,
    path_length: float,
    E_QG: float = 1.22e19, xi: float = 1.0, n: int = 1,
) -> float:
    """delta_t = path * (1/v_low - 1/v_high).

    For sub-luminal LV (xi>0), high-energy photons are slower; delta_t > 0.
    """
    v_low = modified_speed_of_light(energy_low, E_QG, xi, n)
    v_high = modified_speed_of_light(energy_high, E_QG, xi, n)
    return float(path_length * (1.0 / v_high - 1.0 / v_low))


def lp_path_length_through_ctc_bands(
    vs: VanStockumInterior, r_min: float, r_max: float,
    n_samples: int = 200,
) -> float:
    """Proper radial path length from r_min to r_max."""
    r_grid = np.linspace(r_min, r_max, n_samples)
    F_vals = vs.analytic_exterior_F(r_grid)
    # Proper length: int sqrt(h(r)) dr; h ~ 1 in our convention
    dr = (r_max - r_min) / (n_samples - 1)
    return float(np.sum(np.ones_like(r_grid) * dr))


def LP_threshold_modification(
    reaction_threshold: float, xi: float = 1.0,
    E_QG: float = 1.22e19, n: int = 1,
) -> dict:
    """Shift in a reaction threshold (e.g., GZK cutoff)."""
    delta = xi / 2 * (reaction_threshold / E_QG) ** n
    new_threshold = reaction_threshold * (1.0 + delta)
    return {
        "threshold_standard": reaction_threshold,
        "threshold_LV": float(new_threshold),
        "relative_shift": float(delta),
    }


def birefringence_test(
    energy: float, xi_left: float = 1.0, xi_right: float = -1.0,
    E_QG: float = 1.22e19, n: int = 1,
) -> dict:
    """Compute v_left, v_right for helicity-dependent dispersion."""
    v_L = modified_speed_of_light(energy, E_QG, xi_left, n)
    v_R = modified_speed_of_light(energy, E_QG, xi_right, n)
    return {
        "energy": energy,
        "v_left": float(v_L),
        "v_right": float(v_R),
        "birefringence_amount": float(v_L - v_R),
    }


def vacuum_birefringence_in_supercritical_band(
    vs: VanStockumInterior, energy: float = 1e15,  # GeV (UHECR scale)
    E_QG: float = 1.22e19,
) -> dict:
    """For LP supercritical, does birefringence vary across CTC bands?"""
    if not vs.is_supercritical():
        return {"regime": "subcritical", "results": []}
    R = vs.R
    alpha = vs.alpha
    gamma_c = math.pi - math.atan(alpha)
    zeros = []
    for n in range(1, 50):
        u_n = (n * math.pi - gamma_c) / alpha
        if u_n <= 0:
            continue
        r_n = R * math.exp(u_n)
        zeros.append(r_n)
        if len(zeros) > 4:
            break
    results = []
    for i, r in enumerate(zeros):
        # F value at r (should be ~0 at CH)
        F = float(vs.analytic_exterior_F(np.array([r]))[0])
        # Effective xi from F modulation (heuristic)
        xi_eff = abs(F) + 0.5
        bir = birefringence_test(energy, xi_left=xi_eff, xi_right=-xi_eff,
                                  E_QG=E_QG)
        results.append({
            "horizon_index": i,
            "r": r,
            "F": F,
            "xi_eff": xi_eff,
            "birefringence_amount": bir["birefringence_amount"],
        })
    return {"regime": "supercritical", "results": results}


def LV_constraint_from_GRB(
    observed_delay_seconds: float, energy_low_eV: float = 1e3,
    energy_high_eV: float = 1e9, distance_Mpc: float = 1000.0,
    n: int = 1,
) -> dict:
    """Constrain xi from GRB observed delay between high/low energies.

    Heuristic: |xi| * (E_high - E_low)^n / E_QG^n * (path/c) ~ delay.
    """
    distance_m = distance_Mpc * 3.086e22  # Mpc to meters
    path_per_c = distance_m / 3e8  # seconds
    # delta_t = (xi/2) * (E_high^n - E_low^n) / E_QG^n * path/c
    # xi_max = 2 * delta_t * E_QG^n / ((E_high^n - E_low^n) * path/c)
    E_QG = 1.22e19
    diff_E = (energy_high_eV * 1e-9) ** n - (energy_low_eV * 1e-9) ** n  # GeV^n
    xi_max = abs(2 * observed_delay_seconds * E_QG ** n / (diff_E * path_per_c))
    return {
        "observed_delay_s": observed_delay_seconds,
        "path_per_c_s": path_per_c,
        "xi_upper_bound": float(xi_max),
    }


def lp_specific_birefringence_signature(
    vs: VanStockumInterior, energy: float = 1e15,
) -> dict:
    """Check if LP exterior amplifies birefringence near CHs."""
    if not vs.is_supercritical():
        return {"regime": "subcritical", "max_birefringence": 0.0}
    bif = vacuum_birefringence_in_supercritical_band(vs, energy=energy)
    if not bif["results"]:
        return {"regime": "supercritical", "max_birefringence": 0.0}
    max_b = max(abs(r["birefringence_amount"]) for r in bif["results"])
    return {
        "regime": "supercritical",
        "max_birefringence": float(max_b),
        "max_at_horizon_index": int(np.argmax([
            abs(r["birefringence_amount"]) for r in bif["results"]
        ])),
    }
