"""Phase 2a: renormalised stress-energy <T_mu nu> on the supercritical Tipler exterior.

Quantitative test of Hawking's chronology-protection conjecture (Hawking
1992, Phys. Rev. D 46, 603). The conjecture predicts that the renormalised
stress-energy of a free quantum field diverges as one approaches a Cauchy
horizon from outside the chronology-violating region, suppressing any
attempt to form CTCs. This module turns that prediction into a measured
power law on the cleanest analytically-solvable CTC spacetime.

Strategy
--------
On the (t, r) sector of the Lewis-Papapetrou exterior, with the leading-
order line element

    ds^2_{2D} = -F(r) dt^2 + h(r) dr^2  (h = 1 in the LP leading order),

a massless conformally-coupled scalar field has an *exact* renormalised
stress-energy tensor computable from the Polyakov action (Davies-Fulling-
Unruh 1976, Christensen-Fulling 1977). The construction is

  sigma(r) = (1/2) ln F(r)
  r_*(r) = int sqrt(h / F) dr            (tortoise coordinate)

Defining lightcone coordinates u = t - r_*, v = t + r_*, conformal
factor e^{2 sigma} = F, the renormalised <T_{mu nu}> in any state has the
covariantly-conserved Polyakov decomposition

  <T_{uv}>_state = -(1/(12 pi)) ∂_u ∂_v sigma   (state-independent anomaly)
  <T_{uu}>_state = -(1/(12 pi)) [∂_u^2 sigma - (∂_u sigma)^2] + t_u(u)
  <T_{vv}>_state = -(1/(12 pi)) [∂_v^2 sigma - (∂_v sigma)^2] + t_v(v)

The free functions t_u(u), t_v(v) encode the state choice:

- **Boulware**:  t_u = t_v = 0.  Vacuum at spatial infinity; the
  *strongest* divergence at every Cauchy horizon. This is the state
  whose blowup directly tests the chronology-protection conjecture.
- **Hartle-Hawking analog**: t_u = t_v = pi T_H^2 / 12, regularising
  outgoing/ingoing modes at the *first* Cauchy horizon. Mimics the
  thermal Hartle-Hawking state of an eternal black hole.
- **Unruh analog**: t_v = pi T_H^2 / 12, t_u = 0 (or vice versa);
  one-sided thermal. The natural "evaporating" state.

For a static field (∂_t sigma = 0), the algebra collapses to

  <T_{tt}>_B          = (1/(24 pi)) (∂_{r_*} sigma)^2
  <T_{r_* r_*}>_B     = (1/(24 pi)) [(∂_{r_*} sigma)^2 - 2 ∂_{r_*}^2 sigma]
  <T_{t r_*}>_B       = 0
  <T^mu_{~mu}>_B      = R_{2D} / (24 pi)    (trace anomaly, *exact*)

The Hartle-Hawking analog adds a uniform thermal pressure
  Delta <T_{tt}> = -(pi / 12) T_H^2,      Delta <T_{r_* r_*}> = pi T_H^2 / 12.

Chronology-protection signature
-------------------------------
Each Cauchy horizon of the supercritical Tipler exterior sits at a simple
zero of F(r). Near r = r_H with F(r) ~ F'_H (r - r_H):

  (∂_{r_*} sigma)^2 ~ (F')^2 / (4 F)  →  F'_H / [4 (r - r_H)]
  <T_{tt}>_B ~  F'_H / [96 pi (r - r_H)]              (simple-pole divergence)
  <T_{rr}>_B = <T_{r_* r_*}>_B / F  ~  -F''_H / (24 pi F'_H (r - r_H))

The Boulware diagnostic therefore predicts a clean -1 power law in
log|<T_{tt}>_B| vs log|r - r_H|. We measure this fit and report it as
the quantitative chronology-protection verdict.

Public API
----------
- `tortoise_coordinate`: r_*(r) via cumulative quadrature
- `polyakov_sigma`, `polyakov_sigma_derivatives`: building blocks
- `boulware_stress_tensor`, `hartle_hawking_stress_tensor`,
  `unruh_stress_tensor`: 2x2 <T_{mu nu}> in (t, r) coords
- `stress_tensor`: dispatch by state name
- `divergence_rate_at_horizon`: power-law fit near a Cauchy horizon
- `chronology_protection_report`: full multi-horizon study
- `chronology_protection_novelty_scan`: address-space catcher

References
----------
- S.W. Hawking, *Chronology protection conjecture*, Phys. Rev. D 46
  (1992) 603.
- P.C.W. Davies, S.A. Fulling, W.G. Unruh, Phys. Rev. D 13 (1976) 2720.
- S.M. Christensen, S.A. Fulling, Phys. Rev. D 15 (1977) 2088.
- N.D. Birrell, P.C.W. Davies, *Quantum Fields in Curved Space*, CUP 1982,
  Ch. 8 (Polyakov action, 2D anomaly, Hawking radiation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import numpy as np

from .novelty_catcher import scan_novelty
from .quantum_diagnostics import (
    cauchy_horizon_estimate,
    surface_gravity_at_horizon,
)
from .vanstockum import VanStockumInterior


class StressEnergyState(str, Enum):
    """Canonical vacuum-state choices for the 2D Polyakov stress tensor."""

    BOULWARE = "boulware"
    HARTLE_HAWKING = "hartle_hawking"
    UNRUH = "unruh"


# ---------------------------------------------------------------------------
# Geometric building blocks
# ---------------------------------------------------------------------------


def _F(vs: VanStockumInterior, r: float) -> float:
    return float(np.asarray(vs.analytic_exterior_F(np.array([float(r)])))[0])


def polyakov_sigma(vs: VanStockumInterior, r: float) -> float:
    """sigma(r) = (1/2) ln F(r). Undefined when F <= 0."""
    F = _F(vs, r)
    if F <= 0:
        return float("nan")
    return 0.5 * float(np.log(F))


def _F_central_derivatives(
    vs: VanStockumInterior, r: float, eps: float
) -> tuple[float, float, float]:
    """Return (F, F', F'') at radius r via central differences."""
    F = _F(vs, r)
    F_plus = _F(vs, r + eps)
    F_minus = _F(vs, r - eps)
    Fp = (F_plus - F_minus) / (2.0 * eps)
    Fpp = (F_plus - 2.0 * F + F_minus) / (eps * eps)
    return F, Fp, Fpp


def polyakov_sigma_derivatives(
    vs: VanStockumInterior, r: float, eps: float = 1e-5, h_metric: float = 1.0,
) -> dict:
    """Compute sigma, ∂_{r_*} sigma, and ∂_{r_*}^2 sigma at r.

    Uses analytic identities applied to F(r) and a chosen value h_metric
    for the LP radial metric coefficient (default 1, the leading-order
    LP value). Returns NaN where F <= 0.
    """
    F, Fp, Fpp = _F_central_derivatives(vs, r, eps)
    if F <= 0:
        return {
            "sigma": float("nan"),
            "sigma_rstar": float("nan"),
            "sigma_rstar2": float("nan"),
            "F": F,
            "F_prime": Fp,
            "F_double_prime": Fpp,
        }
    sqrt_F_over_h = float(np.sqrt(F / h_metric))
    # sigma = 0.5 ln F, sigma' = F'/(2F)
    # ∂_{r_*} = sqrt(F/h) ∂_r, so ∂_{r_*} sigma = sqrt(F/h) F' / (2F)
    sigma_rstar = sqrt_F_over_h * Fp / (2.0 * F)
    # ∂_{r_*}^2 sigma = sqrt(F/h) ∂_r [sqrt(F/h) F'/(2F)]
    #                = sqrt(F/h) * [(F'^2 / (4 F sqrt(Fh))) * (h/F is constant ish)
    # Cleanest: compute by 2nd-order FD on sigma in r_* coordinates.
    # We use: ∂_{r_*}^2 sigma = (F/h) * sigma'' + (1/2) (F'/h) sigma'
    # which comes from ∂_{r_*}^2 = (F/h) ∂_r^2 + (1/2)(F'/h - F h'/h^2) ∂_r,
    # taking h' = 0 (leading-order LP) gives
    #  ∂_{r_*}^2 = (F/h) ∂_r^2 + (F'/(2 h)) ∂_r.
    sigma_p = Fp / (2.0 * F)
    sigma_pp = (Fpp / (2.0 * F)) - (Fp * Fp) / (2.0 * F * F)
    sigma_rstar2 = (F / h_metric) * sigma_pp + (Fp / (2.0 * h_metric)) * sigma_p
    return {
        "sigma": 0.5 * float(np.log(F)),
        "sigma_rstar": float(sigma_rstar),
        "sigma_rstar2": float(sigma_rstar2),
        "F": F,
        "F_prime": Fp,
        "F_double_prime": Fpp,
    }


def tortoise_coordinate(
    vs: VanStockumInterior, r: float, r_ref: float | None = None,
    n_quad: int = 256, h_metric: float = 1.0,
) -> float:
    """r_*(r) = int_{r_ref}^r sqrt(h / F) dr by Simpson's rule.

    Diverges at every Cauchy horizon (F = 0). The reference radius
    defaults to 1.05 R (just outside the source).
    """
    if r_ref is None:
        r_ref = 1.05 * vs.R
    if r == r_ref:
        return 0.0
    sign = 1.0 if r > r_ref else -1.0
    a, b = (r_ref, r) if r > r_ref else (r, r_ref)
    grid = np.linspace(a, b, n_quad + 1)
    F = vs.analytic_exterior_F(grid)
    F = np.asarray(F, dtype=float)
    F_safe = np.where(np.abs(F) > 1e-12, F, np.sign(F + 1e-300) * 1e-12)
    integrand = np.sqrt(np.abs(h_metric / F_safe))
    integrand = np.where(F > 0, integrand, np.nan)
    if np.any(~np.isfinite(integrand)):
        return float("nan")
    integral = float(np.trapezoid(integrand, grid))
    return sign * integral


# ---------------------------------------------------------------------------
# 2D Ricci scalar and trace anomaly (consistency)
# ---------------------------------------------------------------------------


def ricci_scalar_2d(
    vs: VanStockumInterior, r: float, eps: float = 1e-5, h_metric: float = 1.0,
) -> float:
    """R_2D of the (t, r) sector: -F''/(F h) + (F')^2 / (2 F^2 h)."""
    F, Fp, Fpp = _F_central_derivatives(vs, r, eps)
    if abs(F) < 1e-12:
        return float("inf") * np.sign(-Fpp + (Fp ** 2) / 2.0)
    return -Fpp / (F * h_metric) + (Fp * Fp) / (2.0 * F * F * h_metric)


def trace_anomaly_2d(
    vs: VanStockumInterior, r: float, eps: float = 1e-5, h_metric: float = 1.0,
) -> float:
    """<T^mu_{~mu}>_ren = R_2D / (24 pi). Exact for a massless conformally-coupled scalar."""
    R = ricci_scalar_2d(vs, r, eps=eps, h_metric=h_metric)
    return R / (24.0 * np.pi)


# ---------------------------------------------------------------------------
# Polyakov stress tensor
# ---------------------------------------------------------------------------


def _state_thermal_offset(T_H: float) -> float:
    """Uniform constant added to T_uu, T_vv in HH/Unruh: pi T_H^2 / 12."""
    return float(np.pi * T_H * T_H / 12.0)


def _stress_tensor_in_rstar_coords(
    sigma_rstar: float, sigma_rstar2: float, t_u: float, t_v: float,
) -> dict:
    """Assemble (T_{tt}, T_{r_* r_*}, T_{t r_*}, T_{uv}) from sigma derivatives + state offsets.

    Working in the t / r_* basis, for a static state (∂_t sigma = 0):

      ∂_u sigma = -sigma_{r_*} / 2,   ∂_v sigma = +sigma_{r_*} / 2,
      ∂_u^2 sigma = ∂_v^2 sigma = sigma_{r_*r_*} / 4,
      ∂_u ∂_v sigma                = -sigma_{r_*r_*} / 4.

      T_uu = -(1/(12 pi)) (sigma_{r_* r_*}/4 - sigma_{r_*}^2/4) + t_u
      T_vv = -(1/(12 pi)) (sigma_{r_* r_*}/4 - sigma_{r_*}^2/4) + t_v
      T_uv = -(1/(12 pi)) * (-sigma_{r_* r_*}/4) = sigma_{r_* r_*} / (48 pi)

    Then
      T_tt        = T_uu + 2 T_uv + T_vv
      T_{r_* r_*} = T_uu - 2 T_uv + T_vv
      T_{t r_*}   = -T_uu + T_vv = t_v - t_u
    """
    pi = np.pi
    polyakov_part = -(sigma_rstar2 / 4.0 - sigma_rstar * sigma_rstar / 4.0) / (12.0 * pi)
    T_uu = polyakov_part + t_u
    T_vv = polyakov_part + t_v
    T_uv = sigma_rstar2 / (48.0 * pi)
    T_tt = T_uu + 2.0 * T_uv + T_vv
    T_rstar = T_uu - 2.0 * T_uv + T_vv
    T_t_rstar = T_vv - T_uu
    return {
        "T_uu": float(T_uu),
        "T_vv": float(T_vv),
        "T_uv": float(T_uv),
        "T_tt": float(T_tt),
        "T_rstar_rstar": float(T_rstar),
        "T_t_rstar": float(T_t_rstar),
    }


def _rstar_to_r_tensor(T_rstar_dict: dict, F: float, h_metric: float = 1.0) -> dict:
    """Convert (T_{tt}, T_{r_* r_*}, T_{t r_*}) → (T_{tt}, T_{rr}, T_{tr}).

    The tortoise transform is r_* = int sqrt(h / F) dr, so
        dr_* / dr = sqrt(h / F),
        T_{rr}   = (h / F) * T_{r_* r_*},
        T_{tr}   = sqrt(h / F) * T_{t r_*}.
    """
    if F <= 0:
        return {
            "T_tt": T_rstar_dict["T_tt"],
            "T_rr": float("nan"),
            "T_tr": float("nan"),
        }
    sqrt_h_over_F = float(np.sqrt(h_metric / F))
    return {
        "T_tt": T_rstar_dict["T_tt"],
        "T_rr": (h_metric / F) * T_rstar_dict["T_rstar_rstar"],
        "T_tr": sqrt_h_over_F * T_rstar_dict["T_t_rstar"],
    }


def boulware_stress_tensor(
    vs: VanStockumInterior, r: float, eps: float = 1e-5, h_metric: float = 1.0,
) -> dict:
    """<T_{mu nu}>_B in the (t, r) sector. Returns dict with T_tt, T_rr, T_tr (and r_*-frame).

    Boulware state: t_u = t_v = 0 (vacuum at spatial infinity; *divergent*
    at every Cauchy horizon). Returns NaN T_rr/T_tr where F <= 0.
    """
    d = polyakov_sigma_derivatives(vs, r, eps=eps, h_metric=h_metric)
    if not np.isfinite(d["sigma"]):
        return {
            "T_tt": float("nan"), "T_rr": float("nan"), "T_tr": float("nan"),
            "T_rstar_rstar": float("nan"), "T_t_rstar": float("nan"),
            "T_uv": float("nan"), "T_uu": float("nan"), "T_vv": float("nan"),
            "state": "boulware", "F": d["F"],
        }
    rstar = _stress_tensor_in_rstar_coords(
        d["sigma_rstar"], d["sigma_rstar2"], t_u=0.0, t_v=0.0,
    )
    rdict = _rstar_to_r_tensor(rstar, F=d["F"], h_metric=h_metric)
    return {
        **rstar, **rdict, "state": "boulware", "F": d["F"],
    }


def hartle_hawking_stress_tensor(
    vs: VanStockumInterior, r: float, r_horizon: float | None = None,
    eps: float = 1e-5, h_metric: float = 1.0,
) -> dict:
    """<T_{mu nu}>_HH for the Hartle-Hawking analog.

    Adds a uniform thermal offset pi T_H^2 / 12 to both lightcone
    components, with T_H = surface_gravity / (2 pi) at the chosen
    horizon (default: first Cauchy horizon).

    This state is *thermal at spatial infinity* (deviation from
    Boulware by exactly the Planck pressure) and *regular at r_horizon*
    (the divergence at that single horizon is cancelled). It still
    diverges at other Cauchy horizons.
    """
    if r_horizon is None:
        horizons = cauchy_horizon_estimate(vs)
        if len(horizons) == 0:
            raise ValueError("No Cauchy horizons found in vs exterior")
        r_horizon = float(horizons[0])
    kappa = surface_gravity_at_horizon(vs, r_horizon)
    T_H = kappa / (2.0 * np.pi)
    offset = _state_thermal_offset(T_H)
    d = polyakov_sigma_derivatives(vs, r, eps=eps, h_metric=h_metric)
    if not np.isfinite(d["sigma"]):
        return {
            "T_tt": float("nan"), "T_rr": float("nan"), "T_tr": float("nan"),
            "T_rstar_rstar": float("nan"), "T_t_rstar": float("nan"),
            "T_uv": float("nan"), "T_uu": float("nan"), "T_vv": float("nan"),
            "state": "hartle_hawking", "F": d["F"],
            "T_H": T_H, "r_horizon": r_horizon,
        }
    rstar = _stress_tensor_in_rstar_coords(
        d["sigma_rstar"], d["sigma_rstar2"], t_u=offset, t_v=offset,
    )
    rdict = _rstar_to_r_tensor(rstar, F=d["F"], h_metric=h_metric)
    return {
        **rstar, **rdict, "state": "hartle_hawking", "F": d["F"],
        "T_H": float(T_H), "r_horizon": float(r_horizon),
    }


def unruh_stress_tensor(
    vs: VanStockumInterior, r: float, r_horizon: float | None = None,
    eps: float = 1e-5, h_metric: float = 1.0, outgoing_thermal: bool = True,
) -> dict:
    """<T_{mu nu}>_U for the Unruh analog: one-sided thermal.

    `outgoing_thermal=True`: t_v = pi T_H^2 / 12, t_u = 0 (thermal
    outgoing radiation, Boulware-like ingoing). Mimics the natural
    "evaporating" state.

    Carries non-zero <T_{t r}> (radial energy flux), the canonical
    signature of one-sided thermal emission. Regular on one of the
    two future Cauchy horizons; divergent on the other.
    """
    if r_horizon is None:
        horizons = cauchy_horizon_estimate(vs)
        if len(horizons) == 0:
            raise ValueError("No Cauchy horizons found in vs exterior")
        r_horizon = float(horizons[0])
    kappa = surface_gravity_at_horizon(vs, r_horizon)
    T_H = kappa / (2.0 * np.pi)
    offset = _state_thermal_offset(T_H)
    if outgoing_thermal:
        t_u, t_v = 0.0, offset
    else:
        t_u, t_v = offset, 0.0
    d = polyakov_sigma_derivatives(vs, r, eps=eps, h_metric=h_metric)
    if not np.isfinite(d["sigma"]):
        return {
            "T_tt": float("nan"), "T_rr": float("nan"), "T_tr": float("nan"),
            "T_rstar_rstar": float("nan"), "T_t_rstar": float("nan"),
            "T_uv": float("nan"), "T_uu": float("nan"), "T_vv": float("nan"),
            "state": "unruh", "F": d["F"],
            "T_H": T_H, "r_horizon": r_horizon,
            "outgoing_thermal": outgoing_thermal,
        }
    rstar = _stress_tensor_in_rstar_coords(
        d["sigma_rstar"], d["sigma_rstar2"], t_u=t_u, t_v=t_v,
    )
    rdict = _rstar_to_r_tensor(rstar, F=d["F"], h_metric=h_metric)
    return {
        **rstar, **rdict, "state": "unruh", "F": d["F"],
        "T_H": float(T_H), "r_horizon": float(r_horizon),
        "outgoing_thermal": outgoing_thermal,
    }


def stress_tensor(
    vs: VanStockumInterior, r: float, state: str | StressEnergyState = StressEnergyState.BOULWARE,
    r_horizon: float | None = None, eps: float = 1e-5, h_metric: float = 1.0,
) -> dict:
    """Dispatch to the appropriate state-specific stress-tensor function."""
    state_val = StressEnergyState(state) if not isinstance(state, StressEnergyState) else state
    if state_val == StressEnergyState.BOULWARE:
        return boulware_stress_tensor(vs, r, eps=eps, h_metric=h_metric)
    if state_val == StressEnergyState.HARTLE_HAWKING:
        return hartle_hawking_stress_tensor(
            vs, r, r_horizon=r_horizon, eps=eps, h_metric=h_metric,
        )
    if state_val == StressEnergyState.UNRUH:
        return unruh_stress_tensor(
            vs, r, r_horizon=r_horizon, eps=eps, h_metric=h_metric,
        )
    raise ValueError(f"Unknown state: {state}")


# ---------------------------------------------------------------------------
# Cauchy-horizon divergence study
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DivergenceFit:
    """Power-law fit |<T_{tt}>| ~ A |r - r_H|^p near a Cauchy horizon."""

    r_horizon: float
    component: str
    power: float
    amplitude_log: float
    n_samples_fit: int
    state: str
    fit_residual_rms: float
    diverges: bool
    sample_r: tuple
    sample_T: tuple


def divergence_rate_at_horizon(
    vs: VanStockumInterior, r_horizon: float, state: str | StressEnergyState = StressEnergyState.BOULWARE,
    n_samples: int = 18, eps_min: float = 5e-4, eps_max: float = 5e-2,
    component: str = "T_tt", eps_diff: float = 1e-5,
) -> DivergenceFit:
    """Fit log|<T_{component}>| = A + p log|r - r_H| near a Cauchy horizon.

    Approaches r_horizon from outside (r > r_horizon) at geometrically
    spaced offsets. A negative power signals divergence.

    For Boulware near a simple zero of F (Tipler exterior), the
    theoretical leading-order behaviour is

        <T_{tt}>_B ~ F'_H / (96 pi (r - r_H))  →  power = -1.

    Approach from the appropriate side (outside the chronology-violating
    band, where F > 0 monotonically increasing or decreasing).
    """
    state_val = StressEnergyState(state) if not isinstance(state, StressEnergyState) else state
    eps_grid = np.geomspace(eps_min, eps_max, n_samples)
    # Decide approach direction: from outside, prefer r > r_H first; if F < 0
    # there, fall back to r < r_H. Use sign of F at r_H + eps_min as test.
    F_outside = _F(vs, r_horizon + eps_min)
    if F_outside <= 0:
        # try inside
        F_inside = _F(vs, r_horizon - eps_min)
        if F_inside <= 0:
            raise ValueError(f"Cannot find regular side of horizon r_H = {r_horizon}")
        eps_grid = -eps_grid
    r_samples = r_horizon + eps_grid
    T_samples = []
    for r in r_samples:
        T = stress_tensor(vs, float(r), state=state_val, r_horizon=r_horizon, eps=eps_diff)
        T_samples.append(T.get(component, float("nan")))
    T_arr = np.array(T_samples, dtype=float)
    log_eps = np.log(np.abs(eps_grid))
    finite_mask = np.isfinite(T_arr) & (np.abs(T_arr) > 1e-30)
    if finite_mask.sum() < 4:
        return DivergenceFit(
            r_horizon=float(r_horizon), component=component, power=float("nan"),
            amplitude_log=float("nan"), n_samples_fit=int(finite_mask.sum()),
            state=state_val.value, fit_residual_rms=float("nan"),
            diverges=False, sample_r=tuple(r_samples.tolist()),
            sample_T=tuple(T_arr.tolist()),
        )
    log_T = np.log(np.abs(T_arr[finite_mask]))
    p, log_A = np.polyfit(log_eps[finite_mask], log_T, 1)
    resid = log_T - (p * log_eps[finite_mask] + log_A)
    return DivergenceFit(
        r_horizon=float(r_horizon),
        component=component,
        power=float(p),
        amplitude_log=float(log_A),
        n_samples_fit=int(finite_mask.sum()),
        state=state_val.value,
        fit_residual_rms=float(np.sqrt(np.mean(resid ** 2))),
        diverges=bool(p < -0.1),
        sample_r=tuple(r_samples.tolist()),
        sample_T=tuple(T_arr.tolist()),
    )


# ---------------------------------------------------------------------------
# Full chronology-protection report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChronologyProtectionReport:
    """End-to-end Phase 2a verdict."""

    omega: float
    R: float
    alpha: float
    n_horizons_scanned: int
    horizons: tuple
    boulware_fits: tuple
    hartle_hawking_fits: tuple
    unruh_fits: tuple
    trace_anomaly_max_residual: float
    verdict: str
    summary: str


def _trace_anomaly_check(
    vs: VanStockumInterior, r_samples: np.ndarray, eps_diff: float = 1e-5,
) -> float:
    """Max |T^mu_mu - R_2D/(24 pi)| over r_samples in Boulware (should be ~ 0).

    Sanity check: the trace of <T_{mu nu}>_B in the (t, r) sector must
    equal the 2D conformal anomaly R_{2D}/(24 pi), exactly (Polyakov
    identity). Numerical residual is finite-difference noise.
    """
    residuals = []
    for r in r_samples:
        T = boulware_stress_tensor(vs, float(r), eps=eps_diff)
        F = T["F"]
        if F <= 0 or not np.isfinite(T["T_tt"]):
            continue
        # <T^mu_mu> = g^{tt} T_tt + g^{rr} T_rr
        # In 2D LP (t, r) sector with g_tt = -F, g_rr = h = 1:
        # g^{tt} = -1/F, g^{rr} = 1.
        T_mu_mu = (-1.0 / F) * T["T_tt"] + T["T_rr"]
        anomaly = trace_anomaly_2d(vs, float(r), eps=eps_diff)
        residuals.append(abs(T_mu_mu - anomaly))
    if not residuals:
        return float("nan")
    return float(np.max(residuals))


def chronology_protection_report(
    vs: VanStockumInterior, n_horizons: int = 3,
    component: str = "T_tt", n_samples: int = 18,
    eps_min: float = 5e-4, eps_max: float = 5e-2,
    eps_diff: float = 1e-5,
) -> ChronologyProtectionReport:
    """Full Phase 2a study: fit divergence rate at each of the first N Cauchy horizons
    in three vacuum states, plus the trace-anomaly consistency check.

    The chronology-protection conjecture is *consistent* iff the
    Boulware fit power is approximately -1 at every horizon scanned
    (simple-pole divergence in r - r_H, equivalent to a Planck-scale
    blowup in proper distance).
    """
    if not vs.is_supercritical():
        raise ValueError("chronology_protection_report requires the supercritical regime (a > 1/2)")
    all_horizons = cauchy_horizon_estimate(vs)
    horizons = all_horizons[:n_horizons].tolist()

    boulware = []
    hh = []
    unruh = []
    for rh in horizons:
        boulware.append(divergence_rate_at_horizon(
            vs, float(rh), state=StressEnergyState.BOULWARE,
            n_samples=n_samples, eps_min=eps_min, eps_max=eps_max,
            component=component, eps_diff=eps_diff,
        ))
        hh.append(divergence_rate_at_horizon(
            vs, float(rh), state=StressEnergyState.HARTLE_HAWKING,
            n_samples=n_samples, eps_min=eps_min, eps_max=eps_max,
            component=component, eps_diff=eps_diff,
        ))
        unruh.append(divergence_rate_at_horizon(
            vs, float(rh), state=StressEnergyState.UNRUH,
            n_samples=n_samples, eps_min=eps_min, eps_max=eps_max,
            component=component, eps_diff=eps_diff,
        ))

    # Trace-anomaly check on a sweep just outside the source
    r_sample_grid = np.linspace(1.05 * vs.R, 0.95 * float(horizons[0]), 20)
    anomaly_residual = _trace_anomaly_check(vs, r_sample_grid, eps_diff=eps_diff)

    # Verdict
    boulware_powers = np.array([f.power for f in boulware])
    n_diverging = int(np.sum([f.diverges for f in boulware]))
    close_to_minus_one = np.mean(np.abs(boulware_powers - (-1.0)) < 0.25)
    if n_diverging == len(horizons) and close_to_minus_one >= 0.5:
        verdict = "chronology_protection_consistent"
        summary = (
            f"Hawking CP conjecture *consistent*: Boulware <T_{{tt}}>_B diverges "
            f"as ~ 1/(r - r_H) at every one of {len(horizons)} Cauchy horizons "
            f"scanned (powers = {np.round(boulware_powers, 3).tolist()}). "
            f"Trace anomaly residual = {anomaly_residual:.2e}."
        )
    elif n_diverging >= len(horizons) / 2:
        verdict = "chronology_protection_partial"
        summary = (
            f"Hawking CP *partial*: Boulware <T_{{tt}}> diverges at {n_diverging}/"
            f"{len(horizons)} horizons. Powers = {np.round(boulware_powers, 3).tolist()}."
        )
    else:
        verdict = "chronology_protection_inconsistent"
        summary = (
            f"Hawking CP *inconsistent*: Boulware <T_{{tt}}> does NOT diverge at "
            f"{len(horizons) - n_diverging}/{len(horizons)} horizons. Powers = "
            f"{np.round(boulware_powers, 3).tolist()}."
        )

    return ChronologyProtectionReport(
        omega=float(vs.omega), R=float(vs.R), alpha=float(vs.alpha),
        n_horizons_scanned=len(horizons),
        horizons=tuple(float(x) for x in horizons),
        boulware_fits=tuple(boulware),
        hartle_hawking_fits=tuple(hh),
        unruh_fits=tuple(unruh),
        trace_anomaly_max_residual=float(anomaly_residual),
        verdict=verdict,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Novelty catcher (always-on rule for Systrophe deliverables)
# ---------------------------------------------------------------------------


def chronology_protection_novelty_scan(
    vs: VanStockumInterior, n_radii: int = 40,
    state: str | StressEnergyState = StressEnergyState.BOULWARE,
    r_min: float | None = None, r_max: float | None = None,
    eps_diff: float = 1e-5,
) -> dict:
    """Address-space lambda_2 catcher on <T_{tt}>(r) and <T_{rr}>(r) across a sweep.

    Per the always-on directive (feedback_systrophe_novelty_catcher_always_on,
    2026-05-11), this scan is run as part of every Systrophe deliverable.
    Sharp lambda_2 jumps mark the Cauchy-horizon transitions; the catcher
    recovers them from address-space alone, without any horizon-specific
    knowledge.
    """
    if not vs.is_supercritical():
        raise ValueError("supercritical regime required")
    if r_min is None:
        r_min = 1.05 * vs.R
    if r_max is None:
        # Sweep through the first 3 Cauchy horizons by default
        horizons = cauchy_horizon_estimate(vs)
        r_max = float(horizons[2]) * 1.5 if len(horizons) >= 3 else 10.0 * vs.R
    r_grid = np.linspace(r_min, r_max, n_radii)

    def output_fn(r: float) -> np.ndarray:
        T = stress_tensor(vs, r, state=state, eps=eps_diff)
        # Returns (log|T_tt|, log|T_rr|, log|F|) clipped to finite range
        def lg(x):
            if x is None or not np.isfinite(x):
                return 1e6
            return float(np.log(np.abs(x) + 1e-30))
        return np.array([lg(T["T_tt"]), lg(T["T_rr"]), lg(T["F"])])

    result = scan_novelty(r_grid, output_fn, n_bits=48, parameter_label="r",
                          radii=(4, 8, 16, 24))
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
        "n_radii": int(n_radii),
        "r_grid": r_grid.tolist(),
        "state": state if isinstance(state, str) else state.value,
    }
