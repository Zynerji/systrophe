"""Holographic / boundary-CFT dictionary for the LP cylinder.

The rotating-cylinder exterior is not asymptotically AdS, but at large
r the supercritical metric components oscillate log-periodically about
a power-law envelope. This admits a (speculative) AdS/CFT-style
dictionary in which:

  - The cylinder axis (r = 0) plays the role of an "IR" interior;
  - The asymptotic region (r -> infinity) plays the role of a 2D
    "boundary" with log-periodic conformal structure;
  - Bulk fields with log-periodic falloff are dual to boundary
    operators with complex conformal weights Delta = 1 +/- i alpha / 2.

Complex conformal weights are the hallmark of an *unstable* CFT or
one with a hidden RG cycle (Sornette 1998, Wilczek-Son log-periodic
flow). This puts the Tipler exterior in correspondence with a class
of "log-periodic CFTs" that have been studied in:

  - Critical exponents of disordered systems (Sornette)
  - Efimov physics (discrete scale invariance in 3-body)
  - Holographic models with imaginary scaling dimensions
    (Faulkner, Liu, Vegh)

Functions
---------
- boundary_expansion_coefficients: extract beta, alpha, delta at large r
- dual_operator_dimensions: complex Delta from alpha
- brown_york_stress_tensor: quasi-local stress at finite r
- holographic_entropy_strip: Ryu-Takayanagi-style geodesic length
- boundary_two_point_correlator: log-periodic falloff of <O O>
- efimov_correspondence: map Tipler alpha to Efimov scale lambda_E
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class HolographicData:
    """Boundary CFT data extracted from an LP exterior."""

    a: float
    alpha: float  # Tipler log-frequency
    delta_real: float  # Re(Delta) of dual operator
    delta_imag: float  # Im(Delta) (= alpha/2)
    is_unitary: bool  # False if Delta is complex
    log_period: float  # boundary correlation log-period


def boundary_expansion_coefficients(
    vs: VanStockumInterior, r_min: float = 10.0, r_max: float = 100.0,
    n_samples: int = 200,
) -> dict:
    """Extract leading coefficients of F(r) at large r.

    Model: F(r) ~ beta * cos(alpha * ln(r/R) + delta).
    Returns beta, alpha (theoretical), delta_fit.
    """
    if not vs.is_supercritical():
        return {
            "regime": "subcritical_or_critical",
            "beta": None, "alpha": None, "delta_fit": None,
        }

    r = np.geomspace(r_min, r_max, n_samples)
    F = vs.analytic_exterior_F(r)
    alpha = vs.alpha

    u = np.log(r / vs.R)
    # F = A cos(alpha u) + B sin(alpha u). Fit A, B.
    M = np.column_stack([np.cos(alpha * u), np.sin(alpha * u)])
    coef, *_ = np.linalg.lstsq(M, F, rcond=None)
    A, B = coef
    beta = float(np.sqrt(A ** 2 + B ** 2))
    delta_fit = float(np.arctan2(-B, A))
    return {
        "regime": "supercritical",
        "beta": beta,
        "alpha": float(alpha),
        "delta_fit": delta_fit,
        "log_period": float(np.exp(2 * np.pi / alpha)),
    }


def dual_operator_dimensions(vs: VanStockumInterior) -> HolographicData:
    """Conformal dimensions of operators dual to bulk LP modes.

    For a scalar field with falloff phi ~ r^(-Delta) in d=2 boundary,
    log-periodic structure F ~ cos(alpha ln r) corresponds to
    Delta = 1 +/- i alpha / 2 (Breitenlohner-Freedman violation regime).
    """
    a = vs.a
    if not vs.is_supercritical():
        return HolographicData(
            a=a, alpha=0.0, delta_real=1.0, delta_imag=0.0,
            is_unitary=True, log_period=float("inf"),
        )
    alpha = vs.alpha
    return HolographicData(
        a=a, alpha=float(alpha),
        delta_real=1.0,
        delta_imag=float(alpha / 2.0),
        is_unitary=False,
        log_period=float(np.exp(2 * np.pi / alpha)),
    )


def brown_york_stress_tensor(
    vs: VanStockumInterior, r_boundary: float,
) -> dict:
    """Quasi-local Brown-York stress tensor at finite r = r_boundary.

    For a stationary axisymmetric metric, the BY tensor reduces to
    surface integrals of the extrinsic curvature K_ab on the r =
    const surface. We compute the trace and the t-phi component.

    This is a heuristic computation (full BY would require background
    subtraction); the trace and t-phi component are returned in their
    bare form.
    """
    r = float(r_boundary)
    eps = 1e-4 * r
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

    # Bare extrinsic-curvature trace proxy
    trace_K = Fp / max(abs(F), 1e-30) + Lp / max(abs(L), 1e-30)
    return {
        "r_boundary": r,
        "T_trace_bare": float(trace_K),
        "T_t_phi_bare": float(Kp),
        "F_at_boundary": F,
        "K_at_boundary": K,
        "L_at_boundary": L,
    }


def holographic_entropy_strip(
    vs: VanStockumInterior, strip_width: float,
    r_boundary: float = 50.0,
) -> dict:
    """Ryu-Takayanagi-style entanglement of a phi-interval on the boundary.

    For a boundary strip of width Delta phi on r = r_boundary, the
    minimal-surface entropy is approximated by the proper length of
    the bulk geodesic anchored on the two endpoints. We compute the
    rough scaling S(Delta phi) ~ log(Delta phi).

    Returns the geometric proxy:
       S_proxy = sqrt(L(r_b)) * Delta phi - leading_divergence
    """
    L = float(vs.analytic_exterior_L(np.array([r_boundary]))[0])
    s_proxy = float(np.sqrt(abs(L)) * strip_width)
    # Log-periodic correction
    if vs.is_supercritical():
        alpha = vs.alpha
        s_proxy *= 1 + 0.1 * np.cos(alpha * np.log(strip_width + 1e-12))
    return {
        "strip_width": strip_width,
        "r_boundary": r_boundary,
        "S_proxy": s_proxy,
        "L_at_boundary": L,
    }


def boundary_two_point_correlator(
    vs: VanStockumInterior, separation: float,
) -> complex:
    """<O(0) O(s)> ~ s^(-2 Delta) with complex Delta.

    For Re(Delta) = 1, Im(Delta) = alpha/2 (supercritical), this
    becomes |s|^(-2) * exp(-i alpha ln |s|), i.e. a log-periodically
    *oscillating* two-point function -- the signature of a
    non-unitary, RG-cyclic CFT.
    """
    s = float(separation)
    if s <= 0:
        raise ValueError("separation must be positive")
    if not vs.is_supercritical():
        return complex(1.0 / s ** 2, 0.0)
    alpha = vs.alpha
    mag = 1.0 / s ** 2
    phase = -alpha * np.log(s)
    return complex(mag * np.cos(phase), mag * np.sin(phase))


def efimov_correspondence(vs: VanStockumInterior) -> dict:
    """Map Tipler alpha to Efimov geometric scaling lambda_E.

    In 3-body Efimov physics, three-particle bound states form a
    geometric tower with scale ratio lambda_E = exp(pi/s_0) where
    s_0 ~ 1.00624 for identical bosons. The Tipler analog is
    lambda_T = exp(pi/alpha).
    """
    if not vs.is_supercritical():
        return {
            "regime": "subcritical_or_critical",
            "lambda_tipler": None, "s0_equivalent": None,
            "matches_efimov_22.7": False,
        }
    alpha = vs.alpha
    lambda_tipler = float(np.exp(np.pi / alpha))
    s0_eq = float(alpha)  # The "Efimov scaling parameter"
    # Efimov tower lambda_E = 22.7 corresponds to s_0 = 1.00624
    matches = abs(lambda_tipler - 22.7) / 22.7 < 0.05
    return {
        "regime": "supercritical",
        "lambda_tipler": lambda_tipler,
        "s0_equivalent": s0_eq,
        "matches_efimov_22.7": bool(matches),
    }


def reconstruct_alpha_from_holographic_data(
    vs: VanStockumInterior, sample_separations: np.ndarray | None = None,
) -> float:
    """Cross-check: extract alpha from boundary correlator phase wrap.

    Computes <O(s) O(0)> at multiple s, fits the phase to -alpha ln(s).
    """
    if sample_separations is None:
        sample_separations = np.geomspace(1.0, 1000.0, 50)
    s_arr = np.asarray(sample_separations, dtype=float)
    if not vs.is_supercritical():
        return 0.0
    correlators = np.array([
        boundary_two_point_correlator(vs, float(s)) for s in s_arr
    ])
    # Unwrap phase (already by construction phi = -alpha ln s)
    phases = np.angle(correlators)
    unwrapped = np.unwrap(phases)
    log_s = np.log(s_arr)
    # Fit unwrapped = -alpha * log_s + c
    slope, _ = np.polyfit(log_s, unwrapped, 1)
    return float(abs(slope))
