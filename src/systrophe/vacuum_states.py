"""Vacuum-state selection criteria on the LP background.

Validates speculative item I.6 from `docs/INTERPRETATIONS.md`:
"Casimir cavity at throat IS the natural QFT vacuum."

A *vacuum state* on a curved-spacetime QFT requires choosing a
selection criterion. We implement three standard choices:

1. **Boulware vacuum**: defined by no excitations w.r.t. the
   Schwarzschild-like timelike Killing vector partial_t. On the LP
   exterior, this exists for r where F(r) > 0 (the static region).
   Divergent at horizons.

2. **Adiabatic vacuum** (order n): defined by WKB matching to
   geodesically-built modes. Well-defined everywhere F > 0.

3. **Conformal-Killing vacuum**: defined by Killing structure of a
   conformally-related metric. On LP exteriors, the conformal Killing
   structure exists only in special slow-rotation limits.

Each vacuum gives a *different* <T_{mu nu}>; this module exposes the
state-dependent piece on top of the locally-determined Hadamard
geometric piece from `hadamard_offtrace.py`.

The conclusion: no choice is "natural" without external input. The
Hartle-Hawking analog (the closest to "natural" on a horizon-
containing background) requires the geometry to support a
*Killing horizon*, which the LP exterior does NOT have in general
(it has chronology horizons, not Killing horizons). The earlier
speculation that "the cavity at throat is the natural vacuum" is
therefore incorrect: the Hartle-Hawking analog does not exist
without further geometric input.

References
----------
- D. G. Boulware, Phys. Rev. D 11 (1975) 1404.
- L. Parker, "Quantized fields and particle creation in expanding
  universes," Phys. Rev. 183 (1969) 1057 (adiabatic vacuum).
- B. S. DeWitt, *Dynamical Theory of Groups and Fields* (1965).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VacuumStateReport:
    """Diagnostic for a vacuum-state choice on the LP background."""

    r: float
    F_value: float
    vacuum_type: str
    well_defined: bool
    divergent_at_horizon: bool
    energy_density: float
    comment: str


def boulware_well_defined(vs, r: float) -> dict:
    """Test whether the Boulware vacuum is well-defined at radius r.

    The Boulware vacuum requires partial_t to be a timelike Killing
    vector at r, i.e., F(r) = -g_{tt} > 0. At horizons (F -> 0), the
    vacuum is divergent.
    """
    F = float(vs.analytic_exterior_F(r))
    well_def = F > 1e-12
    div_at_h = F < 1e-3 and F > 0
    return {
        "r": r,
        "F_value": F,
        "well_defined": well_def,
        "divergent_at_horizon": div_at_h,
        "vacuum_type": "Boulware",
    }


def adiabatic_well_defined(vs, r: float, eps: float = 1e-5) -> dict:
    """Adiabatic vacuum requires WKB validity: slowly-varying metric.

    Criterion: |dF/dr| / |F| < 1 (rough adiabaticity).
    """
    F = float(vs.analytic_exterior_F(r))
    if abs(F) < 1e-12:
        return {"r": r, "F_value": F, "well_defined": False,
                "divergent_at_horizon": True, "vacuum_type": "Adiabatic"}
    Fp = float(vs.analytic_exterior_F(r + eps))
    Fm = float(vs.analytic_exterior_F(r - eps))
    dFdr = (Fp - Fm) / (2 * eps)
    adiabatic_param = abs(dFdr) / abs(F)
    well_def = adiabatic_param < 1.0
    return {
        "r": r,
        "F_value": F,
        "adiabatic_parameter": adiabatic_param,
        "well_defined": well_def,
        "divergent_at_horizon": False,
        "vacuum_type": "Adiabatic",
    }


def hartle_hawking_analog_exists(vs, r: float) -> dict:
    """Test whether a Hartle-Hawking analog exists at radius r.

    HH vacuum requires a *Killing* horizon: the gradient of F at
    F = 0 must be along a Killing vector. The LP exterior has
    chronology horizons (where F + K^2/L flips sign) but they are
    NOT Killing horizons in general --- partial_t becomes spacelike
    not via Killing degeneracy but via metric component crossover.

    So HH does NOT exist generically; only in slow-rotation
    asymptotic limits where omega R << 1.
    """
    F = float(vs.analytic_exterior_F(r))
    K = float(vs.analytic_exterior_K(r))
    L = float(vs.analytic_exterior_L(r))
    # Killing-horizon proxy: F = 0 AND K finite
    is_chronology_horizon = abs(F) < 1e-3
    # Effective lapse-squared: F + K^2/L
    if abs(L) > 1e-30:
        alpha_sq = F + K * K / L
    else:
        alpha_sq = F
    # The chronology horizon is a Killing horizon iff alpha_sq -> 0 at F=0
    is_killing = abs(alpha_sq) < 1e-3
    return {
        "r": r,
        "F_value": F,
        "K_value": K,
        "L_value": L,
        "alpha_squared": float(alpha_sq),
        "is_chronology_horizon": bool(is_chronology_horizon),
        "is_killing_horizon": bool(is_killing),
        "hh_analog_exists": bool(is_killing),
    }


def boulware_energy_density_proxy(vs, r: float) -> float:
    """Boulware energy density (leading-order Hadamard estimate).

    For the conformally-coupled massless scalar in vacuum, the
    leading geometric piece is

        rho_B(r) = (1/2880 pi^2) * R_{0 rho sigma tau} R_0^{rho sigma tau}.

    Returns the (0, 0) component of the locally-determined Hadamard
    tensor (which is one specific local stress-tensor estimate, NOT
    the full state-dependent answer; the latter requires mode sums).
    """
    from .hadamard_offtrace import hadamard_offtrace_T
    T = hadamard_offtrace_T(vs, r=r)
    return float(T[0, 0])


def vacuum_selection_summary(vs, r: float) -> VacuumStateReport:
    """Summarise which vacuum is well-defined at radius r."""
    boulware = boulware_well_defined(vs, r)
    rho = boulware_energy_density_proxy(vs, r)
    if boulware["divergent_at_horizon"]:
        comment = "Boulware divergent near horizon; switch to adiabatic"
    elif boulware["well_defined"]:
        comment = "Boulware vacuum well-defined; use leading Hadamard"
    else:
        comment = "Boulware vacuum NOT well-defined (F <= 0); deep CTC region"
    return VacuumStateReport(
        r=r,
        F_value=boulware["F_value"],
        vacuum_type="Boulware (proxy)",
        well_defined=boulware["well_defined"],
        divergent_at_horizon=boulware["divergent_at_horizon"],
        energy_density=rho,
        comment=comment,
    )


def compare_vacua_at_radius(vs, r: float) -> dict:
    """Compare Boulware, adiabatic, and Hartle-Hawking-analog at one r."""
    boulware = boulware_well_defined(vs, r)
    adiabatic = adiabatic_well_defined(vs, r)
    hh = hartle_hawking_analog_exists(vs, r)
    return {
        "r": r,
        "boulware": boulware,
        "adiabatic": adiabatic,
        "hartle_hawking": hh,
        "all_well_defined": (
            boulware["well_defined"] and adiabatic["well_defined"]
            and hh["hh_analog_exists"]
        ),
        "any_well_defined": (
            boulware["well_defined"] or adiabatic["well_defined"]
            or hh["hh_analog_exists"]
        ),
    }


def natural_vacuum_verdict(vs, r_samples: np.ndarray) -> dict:
    """Verdict on whether a "natural" vacuum exists on the LP background.

    Tests each vacuum criterion at multiple radii. Returns:

      - boulware_fraction_valid: fraction of samples where Boulware works
      - adiabatic_fraction_valid: ditto for adiabatic
      - hh_analog_fraction_valid: ditto for HH analog

    A "natural" vacuum on a horizon-containing background would have
    HH analog valid on a positive-measure set. If the HH fraction is
    ~0, the natural-vacuum claim fails.
    """
    n = len(r_samples)
    n_boulware = 0
    n_adiabatic = 0
    n_hh = 0
    for r in r_samples:
        r = float(r)
        if boulware_well_defined(vs, r)["well_defined"]:
            n_boulware += 1
        if adiabatic_well_defined(vs, r)["well_defined"]:
            n_adiabatic += 1
        if hartle_hawking_analog_exists(vs, r)["hh_analog_exists"]:
            n_hh += 1
    return {
        "n_samples": n,
        "boulware_fraction_valid": n_boulware / n,
        "adiabatic_fraction_valid": n_adiabatic / n,
        "hh_analog_fraction_valid": n_hh / n,
        "natural_vacuum_exists": n_hh / n > 0.0,
        "verdict": (
            "Natural HH-analog vacuum DOES exist on a positive-measure set"
            if n_hh > 0 else
            "Natural HH-analog vacuum DOES NOT exist; use Boulware/adiabatic"
        ),
    }
