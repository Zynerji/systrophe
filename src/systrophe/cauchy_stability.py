"""Cauchy horizon stability analysis for the LP exterior.

Cauchy horizons (CH) are boundaries beyond which the
Einstein-equation initial-value problem fails to determine the
future evolution. In the LP supercritical exterior, each F = 0
surface is a candidate CH. The question: are these horizons stable
against small metric perturbations, or do they amplify perturbations
catastrophically (the "mass inflation" instability of Kerr inner
horizons)?

Strategy: linearise the metric perturbation theory at each F = 0
surface and compute the local Lyapunov exponent of the radial
geodesic flow. A positive Lyapunov exponent indicates exponential
divergence of nearby trajectories -- classical instability.

References
----------
- E. Poisson, W. Israel, Phys. Rev. D 41 (1990) 1796 (mass inflation).
- A. Ori, Phys. Rev. Lett. 67 (1991) 789.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CauchyStabilityReport:
    """Stability descriptor for a single Cauchy horizon."""

    r_horizon: float
    lyapunov_exponent: float  # radial geodesic divergence rate
    is_stable: bool  # True iff lambda <= 0
    F_derivative: float  # dF/dr at the horizon
    K_at_horizon: float
    classification: str  # "stable" / "marginally stable" / "unstable"


def lyapunov_exponent_at_horizon(
    vs, r_horizon: float, eps: float = 1e-5,
) -> float:
    """Local Lyapunov exponent of the radial geodesic flow at r_horizon.

    For a stationary axisymmetric metric, the radial geodesic deviation
    equation gives the rate of separation of nearby radial trajectories.
    The leading-order Lyapunov exponent is

        lambda ~ sqrt(|V_eff''(r_horizon)| / h(r_horizon))

    where V_eff is the effective radial potential for null geodesics.
    A real lambda indicates exponential instability.

    Returns the absolute value of the Lyapunov exponent. A
    classification function elsewhere reports stability.
    """
    F = float(vs.analytic_exterior_F(r_horizon))
    K = float(vs.analytic_exterior_K(r_horizon))
    L = float(vs.analytic_exterior_L(r_horizon))
    F_plus = float(vs.analytic_exterior_F(r_horizon + eps))
    F_minus = float(vs.analytic_exterior_F(r_horizon - eps))
    dF_dr = (F_plus - F_minus) / (2 * eps)
    # Second derivative
    d2F_dr2 = (F_plus - 2 * F + F_minus) / (eps ** 2)
    # Approximate Lyapunov: |sqrt(|d2F_dr2|)|
    return float(np.sqrt(abs(d2F_dr2)))


def stability_at_horizon(
    vs, r_horizon: float, eps: float = 1e-5,
) -> CauchyStabilityReport:
    """Compute stability descriptor at a single Cauchy horizon."""
    lyap = lyapunov_exponent_at_horizon(vs, r_horizon, eps)
    F_plus = float(vs.analytic_exterior_F(r_horizon + eps))
    F_minus = float(vs.analytic_exterior_F(r_horizon - eps))
    dF_dr = (F_plus - F_minus) / (2 * eps)
    K = float(vs.analytic_exterior_K(r_horizon))
    is_stable = lyap < 1e-2
    if lyap < 1e-3:
        classification = "stable"
    elif lyap < 1.0:
        classification = "marginally stable"
    else:
        classification = "unstable"
    return CauchyStabilityReport(
        r_horizon=r_horizon, lyapunov_exponent=lyap,
        is_stable=is_stable, F_derivative=float(dF_dr),
        K_at_horizon=K, classification=classification,
    )


def survey_all_cauchy_horizons(
    vs, r_min: float = 1.05, r_max: float = 50.0,
) -> list[CauchyStabilityReport]:
    """Survey all chronology horizons in r_min..r_max."""
    from .geodesic_completeness import chronology_horizon_radii
    horizons = chronology_horizon_radii(vs, r_min, r_max)
    return [stability_at_horizon(vs, r) for r in horizons]


def mass_inflation_growth_rate(
    vs, r_horizon: float, T_perturb: float = 0.01,
) -> dict:
    """Estimated mass-inflation amplitude for a perturbation at the horizon.

    In Poisson-Israel mass inflation, a small influx of energy at a
    Cauchy horizon grows exponentially via the kinematic shearing of
    inner-vs-outer streams. We estimate the growth rate via the
    Lyapunov exponent:

        delta M / M ~ exp(lambda * t / kappa)

    where kappa is the surface gravity.
    """
    from .quantum_diagnostics import surface_gravity_at_horizon
    lyap = lyapunov_exponent_at_horizon(vs, r_horizon)
    kappa = surface_gravity_at_horizon(vs, r_horizon)
    if kappa < 1e-12:
        return {"growth_rate": float("inf"),
                "characteristic_time": 0.0,
                "lyapunov": lyap, "kappa": kappa}
    growth_rate = lyap / kappa
    char_time = 1.0 / growth_rate if growth_rate > 1e-30 else float("inf")
    return {"growth_rate": growth_rate, "characteristic_time": char_time,
            "lyapunov": lyap, "kappa": kappa,
            "perturbation_growth_factor": float(np.exp(min(growth_rate * T_perturb, 100))),
            "is_unstable_growing": growth_rate > 1e-3}


def chronology_protection_via_instability(
    vs, r_min: float = 1.05, r_max: float = 50.0,
) -> dict:
    """Check the Hawking chronology-protection mechanism via Cauchy
    instability.

    If ALL Cauchy horizons are unstable (lambda > 0), small perturbations
    grow without bound and ultimately destabilise the spacetime,
    preventing actual CTCs from forming -- the Hawking mechanism.

    Returns:
      n_horizons, n_stable, n_unstable
      most_unstable_lambda
      all_unstable: True iff every horizon has lambda > 0
    """
    reports = survey_all_cauchy_horizons(vs, r_min, r_max)
    n_stable = sum(1 for r in reports if r.is_stable)
    n_unstable = len(reports) - n_stable
    if reports:
        max_lambda = max(r.lyapunov_exponent for r in reports)
    else:
        max_lambda = 0.0
    return {
        "n_horizons": len(reports),
        "n_stable": n_stable,
        "n_unstable": n_unstable,
        "max_lyapunov": max_lambda,
        "all_unstable": all(not r.is_stable for r in reports) if reports else False,
        "supports_chronology_protection": (
            n_unstable > 0 and len(reports) > 0
            and all(not r.is_stable for r in reports)
        ),
    }
