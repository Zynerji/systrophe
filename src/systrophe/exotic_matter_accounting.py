"""Conditional exotic-matter accounting (Morris-Thorne vs. Casimir).

Validates speculative item I.2 from `docs/INTERPRETATIONS.md`:
"The Casimir negative energy replaces the exotic matter that Tipler
/ Morris-Thorne wormholes require."

Important framing
-----------------
The Systrophe LP exterior at L = 0 is **vacuum** (R_{mu nu} = 0)
and is NOT a Morris-Thorne wormhole junction (see the framing
docstring in `wormhole_throat.py`). The LP "throat" stays open
without local exotic matter because no local stress-energy is
required at the L = 0 locus --- the geometry is sourced remotely by
the rotating cylinder interior.

This module therefore answers a *conditional* question:

  "If one tried to interpret the L = 0 locus as a genuine Morris-
   Thorne wormhole junction (against the actual topology), how much
   local exotic matter would Morris-Thorne require, and could the
   Casimir vacuum supply it?"

The answer is quantitative NO: even under that hypothetical
interpretation, the Casimir budget is far too small.

What this module computes
-------------------------
For a Morris-Thorne wormhole with throat radius r_t and shape
function b(r), NEC violation defines the exotic-matter budget

    E_exotic = integral_(throat) [rho + p_r] dV.

Morris-Thorne (1988) Eq. (32): at the throat, the leading-order
density that must violate NEC is

    |rho_exotic|(r_t) = b'(r_t) / (8 pi r_t^2),

with b'(r_t) in (0, 1).

The Casimir vacuum gives rho_C = -pi^2 / (720 d^4) in a parallel-
plate cavity of separation d. Total Casimir energy in a cylindrical
cavity of radius r_t and length d:

    E_C = rho_C * (pi r_t^2 d) = -pi^3 r_t^2 / (720 d^3).

The required plate separation to match exotic-matter budget is

    d_req = (pi^3 r_t^2 / (90 b'(r_t)))^{1/4}.

Verdict
-------
For r_t ~ 1 in natural units (b' = 0.5), d_req / r_t ~ 0.7 --- the
Casimir cavity does not geometrically fit within the throat. The
hypothetical Morris-Thorne interpretation requires either sub-
Planckian plate separation or super-Planckian throat radius. The
Casimir vacuum is NOT sufficient under the Morris-Thorne ansatz.

This is *separate* from the question of whether the LP "throat"
actually needs exotic matter: it does not, because it is vacuum.

References
----------
- M. S. Morris, K. S. Thorne, Am. J. Phys. 56 (1988) 395.
- M. Visser, *Lorentzian Wormholes*, AIP Press 1996.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from .casimir import standard_casimir_energy_density


def morris_thorne_exotic_density(
    r_throat: float, db_dr_at_throat: float = 0.5
) -> float:
    """Effective exotic-matter density required at a Morris-Thorne throat.

    Morris-Thorne (1988) Eq. (32): at the throat,
        8 pi rho(r_t) = b'(r_t) / r_t^2
    so the violation magnitude (assumed entirely in rho) is
        |rho_exotic| = b'(r_t) / (8 pi r_t^2).

    Since b'(r_t) < 1 (flaring-out), |rho_exotic| < 1 / (8 pi r_t^2).
    """
    if r_throat <= 0:
        raise ValueError("r_throat must be positive")
    if not (0 < db_dr_at_throat < 1):
        raise ValueError("db_dr_at_throat must be in (0, 1) for flaring-out")
    return db_dr_at_throat / (8 * pi * r_throat ** 2)


def morris_thorne_exotic_budget(
    r_throat: float, cavity_length: float, db_dr_at_throat: float = 0.5
) -> float:
    """Total exotic-matter energy in a cylindrical cavity at the throat.

    E_exotic = rho_exotic * volume = rho_exotic * (pi r_t^2 * cavity_length).
    """
    rho = morris_thorne_exotic_density(r_throat, db_dr_at_throat)
    return rho * pi * r_throat ** 2 * cavity_length


def casimir_cavity_energy(
    r_throat: float, plate_separation: float
) -> float:
    """Total Casimir negative energy in a cylindrical cavity.

    E_C = rho_C * volume = -(pi^2 / 720 d^4) * (pi r_t^2 d)
        = -pi^3 r_t^2 / (720 d^3).
    """
    if r_throat <= 0 or plate_separation <= 0:
        raise ValueError("r_throat and plate_separation must be positive")
    rho_C = float(standard_casimir_energy_density(plate_separation))
    volume = pi * r_throat ** 2 * plate_separation
    return rho_C * volume


@dataclass(frozen=True)
class ExoticMatterReport:
    """Comparison of exotic-matter requirement vs Casimir budget."""

    r_throat: float
    plate_separation: float
    cavity_length: float
    rho_exotic: float
    E_exotic: float
    rho_casimir: float
    E_casimir: float
    ratio: float
    casimir_sufficient: bool


def exotic_vs_casimir_comparison(
    r_throat: float,
    plate_separation: float,
    cavity_length: float | None = None,
    db_dr_at_throat: float = 0.5,
) -> ExoticMatterReport:
    """Compare exotic-matter requirement vs Casimir-vacuum budget.

    cavity_length defaults to plate_separation (a unit-aspect cylinder).
    Returns a ExoticMatterReport.

    casimir_sufficient is True iff |E_casimir| > E_exotic
    (the Casimir negative energy magnitude exceeds the exotic-matter
    requirement at the throat).
    """
    if cavity_length is None:
        cavity_length = plate_separation
    rho_ex = morris_thorne_exotic_density(r_throat, db_dr_at_throat)
    E_ex = rho_ex * pi * r_throat ** 2 * cavity_length
    rho_C = float(standard_casimir_energy_density(plate_separation))
    E_C = rho_C * pi * r_throat ** 2 * cavity_length
    abs_ratio = abs(E_C) / E_ex if E_ex > 0 else float("inf")
    return ExoticMatterReport(
        r_throat=r_throat,
        plate_separation=plate_separation,
        cavity_length=cavity_length,
        rho_exotic=rho_ex,
        E_exotic=E_ex,
        rho_casimir=rho_C,
        E_casimir=E_C,
        ratio=abs_ratio,
        casimir_sufficient=bool(abs_ratio > 1.0),
    )


def required_plate_separation(
    r_throat: float, db_dr_at_throat: float = 0.5
) -> float:
    """Plate separation d such that |E_Casimir| = E_exotic exactly.

    Solving |pi^2 / (720 d^4)| = b'(r_t) / (8 pi r_t^2):

        d^4 = (pi^2) * 8 pi r_t^2 / (720 b'(r_t))
            = 8 pi^3 r_t^2 / (720 b'(r_t))
            = pi^3 r_t^2 / (90 b'(r_t)).

        d = (pi^3 r_t^2 / (90 b'(r_t)))^(1/4)
    """
    if r_throat <= 0:
        raise ValueError("r_throat must be positive")
    if not (0 < db_dr_at_throat < 1):
        raise ValueError("db_dr_at_throat must be in (0, 1)")
    return (pi ** 3 * r_throat ** 2 / (90 * db_dr_at_throat)) ** 0.25


def is_casimir_route_physically_plausible(
    r_throat: float, max_d_ratio: float = 0.1
) -> dict:
    """Diagnose whether the Casimir-replaces-exotic-matter route is plausible.

    The required plate separation d is compared to a fraction of the
    throat radius. If d <= max_d_ratio * r_throat, the cavity fits
    inside the throat geometrically; if d >> r_throat, the cavity is
    geometrically impossible.
    """
    d_req = required_plate_separation(r_throat)
    geometric_fit = d_req <= max_d_ratio * r_throat
    return {
        "r_throat": r_throat,
        "d_required": d_req,
        "d_to_r_ratio": d_req / r_throat,
        "geometric_fit": geometric_fit,
        "verdict": (
            "PLAUSIBLE: Casimir cavity fits within throat geometry"
            if geometric_fit else
            "NOT PLAUSIBLE: required cavity geometrically too large"
        ),
    }
