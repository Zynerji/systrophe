"""Averaged Null Energy Condition (ANEC) integrals on the LP background.

The pointwise NEC T_{ab} k^a k^b >= 0 is generically violated by quantum
fields (Casimir, squeezed states). The ANEC integrates along a
complete null geodesic:

    int_{-inf}^{+inf} T_{ab} k^a k^b d lambda  >= 0  (claimed bound).

ANEC is conjecturally true for self-consistent semiclassical
spacetimes and has been proven in flat space (Faulkner-Leigh-Parrikar-
Wang 2016) and a few curved cases.

For the LP supercritical exterior, we evaluate the ANEC integral
along radial null geodesics (the simplest null curves). The vacuum
stress-energy is taken from the Hadamard off-trace tensor in
`hadamard_offtrace.py`; non-vacuum states (squeezed, thermal) shift
the integral.

Module functions:
- null_generator_radial: tangent vector of a radial null geodesic
- T_kk_radial: T_{ab} k^a k^b at radius r for a chosen vacuum
- anec_integral_radial: int dr / |k^r| of T_kk over [r_min, r_max]
- anec_integral_circular_null: along a circular null geodesic
- anec_violation_amplitude: how negative the integral can go
- anec_quantum_inequality_bound: Ford-Roman QI lower bound
- pair_extinction_anec_recovery: ANEC restored for Systrophe pair
  at delta = pi (extinction)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class ANECResult:
    r_min: float
    r_max: float
    integral_value: float
    n_samples: int
    violates_ANEC: bool
    quantum_inequality_bound: float


def null_generator_radial(vs: VanStockumInterior, r: float) -> tuple[float, float]:
    """Tangent (k^t, k^r) for a radial null geodesic at radius r.

    From -F (k^t)^2 + h (k^r)^2 = 0 with h = 1: |k^r| = sqrt(|F|) |k^t|.
    Normalise k^t = 1.
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    if F <= 0:
        return (1.0, float("nan"))
    return (1.0, float(math.sqrt(F)))


def T_kk_radial(vs: VanStockumInterior, r: float,
                quantum_state: str = "hartle_hawking") -> float:
    """Effective T_{ab} k^a k^b at r along radial null geodesic.

    For the Hartle-Hawking vacuum on LP supercritical:
        <T_rr> ~ -(1/12 pi) * F'(r)^2 / F(r)^2   (heuristic)
    For pointwise NEC, T_kk = (k^r)^2 <T_rr> + (k^t)^2 <T_tt> .
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    eps = 1e-4 * r
    Fp = (float(vs.analytic_exterior_F(np.array([r + eps]))[0])
          - float(vs.analytic_exterior_F(np.array([r - eps]))[0])) / (2 * eps)
    if abs(F) < 1e-12:
        return float("nan")
    if quantum_state == "hartle_hawking":
        # Heuristic regulated form
        T_rr = -(1.0 / (12 * math.pi)) * (Fp / F) ** 2
        T_tt = (1.0 / (12 * math.pi)) * (Fp / F) ** 2 / 3.0
    elif quantum_state == "boulware":
        # Boulware diverges near horizons (F->0)
        T_rr = -(1.0 / (12 * math.pi)) * (Fp / F) ** 2 * 2.0
        T_tt = (1.0 / (12 * math.pi)) * (Fp / F) ** 2 / 3.0
    else:
        T_rr = 0.0
        T_tt = 0.0
    kt, kr = null_generator_radial(vs, r)
    if not math.isfinite(kr):
        return float("nan")
    return float(F * kt * kt * T_tt + (1.0 / max(F, 1e-30)) * kr * kr * T_rr)


def anec_integral_radial(
    vs: VanStockumInterior,
    r_min: float = None, r_max: float = None,
    n_samples: int = 200,
    quantum_state: str = "hartle_hawking",
) -> ANECResult:
    """int_{r_min}^{r_max} T_{ab} k^a k^b / |k^r| dr along radial null."""
    if r_min is None:
        r_min = vs.R * 1.01
    if r_max is None:
        r_max = vs.R * 10.0
    r_grid = np.linspace(r_min, r_max, n_samples)
    dr = (r_max - r_min) / (n_samples - 1)
    total = 0.0
    for r in r_grid:
        Tkk = T_kk_radial(vs, float(r), quantum_state)
        _, kr = null_generator_radial(vs, float(r))
        if math.isfinite(Tkk) and math.isfinite(kr) and abs(kr) > 0:
            total += Tkk / abs(kr) * dr
    qi_bound = anec_quantum_inequality_bound(vs, r_min, r_max)
    return ANECResult(
        r_min=float(r_min), r_max=float(r_max),
        integral_value=float(total),
        n_samples=int(n_samples),
        violates_ANEC=bool(total < 0),
        quantum_inequality_bound=float(qi_bound),
    )


def anec_quantum_inequality_bound(
    vs: VanStockumInterior, r_min: float, r_max: float,
) -> float:
    """Ford-Roman QI bound: integral >= -C / (r_max - r_min)^4 for a smooth sampling.

    Heuristic in curved space; use the flat-space form as a lower bound.
    """
    if r_max <= r_min:
        return 0.0
    C = 3.0 / (32 * math.pi ** 2)
    return float(-C / (r_max - r_min) ** 4)


def anec_integral_circular_null(
    vs: VanStockumInterior, r: float,
    quantum_state: str = "hartle_hawking",
) -> dict:
    """ANEC along a closed circular null geodesic at radius r.

    Exists where F = 0 (chronology horizon -> null geodesic locus).
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    if F >= 0:
        return {
            "r": r,
            "is_chronology_horizon": False,
            "ANEC_integral_circular": float("nan"),
        }
    # Approximate ANEC along a small phi-loop at r
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    eps = 1e-4 * r
    Fp = (float(vs.analytic_exterior_F(np.array([r + eps]))[0])
          - float(vs.analytic_exterior_F(np.array([r - eps]))[0])) / (2 * eps)
    T_phi_phi = -(1.0 / (12 * math.pi)) * (Fp / max(abs(F), 1e-12)) ** 2
    integrand = T_phi_phi * L  # per unit phi
    integral = integrand * 2 * math.pi
    return {
        "r": r,
        "is_chronology_horizon": True,
        "T_phi_phi": float(T_phi_phi),
        "L": float(L),
        "ANEC_integral_circular": float(integral),
        "violates_ANEC": bool(integral < 0),
    }


def pair_extinction_anec_recovery(
    vs: VanStockumInterior, delta: float,
    r_min: float = None, r_max: float = None,
    n_samples: int = 100,
) -> dict:
    """For a Systrophe pair at offset delta, the effective F-amplitude
    scales by (1 + cos delta)/2. At delta=pi, F=0 -> no CTCs -> ANEC
    integral collapses to ~0 (no chronology-horizon contribution).
    """
    extinction = 0.5 * (1.0 + math.cos(delta))
    base = anec_integral_radial(vs, r_min=r_min, r_max=r_max,
                                 n_samples=n_samples)
    scaled = base.integral_value * extinction ** 2
    return {
        "delta": delta,
        "extinction_factor": extinction,
        "anec_single": base.integral_value,
        "anec_pair_effective": scaled,
        "is_full_extinction": delta == math.pi,
    }


def anec_violation_amplitude(
    vs: VanStockumInterior, r_min: float = None, r_max: float = None,
    n_samples: int = 200,
) -> dict:
    """Maximum negative excursion of T_kk over the integration window."""
    if r_min is None:
        r_min = vs.R * 1.01
    if r_max is None:
        r_max = vs.R * 10.0
    r_grid = np.linspace(r_min, r_max, n_samples)
    Tkk_vals = []
    for r in r_grid:
        v = T_kk_radial(vs, float(r))
        if math.isfinite(v):
            Tkk_vals.append((float(r), v))
    if not Tkk_vals:
        return {"min_T_kk": float("nan"), "max_T_kk": float("nan"),
                "min_at_r": None, "max_at_r": None}
    vals_only = [v for _, v in Tkk_vals]
    min_idx = vals_only.index(min(vals_only))
    max_idx = vals_only.index(max(vals_only))
    return {
        "min_T_kk": float(vals_only[min_idx]),
        "max_T_kk": float(vals_only[max_idx]),
        "min_at_r": float(Tkk_vals[min_idx][0]),
        "max_at_r": float(Tkk_vals[max_idx][0]),
    }
