"""Morris-Thorne wormhole throat construction on the LP exterior.

Validates speculative item I.1 from `docs/INTERPRETATIONS.md`:
"The Z_3 cover physically realises a wormhole throat."

Morris-Thorne wormholes are characterised by a spherically-symmetric
metric

    ds^2 = -e^{2 Phi(r)} dt^2 + dr^2 / (1 - b(r)/r) + r^2 d Omega^2

with `b(r)` the shape function and `Phi(r)` the redshift function.
The throat is at r = r_t where b(r_t) = r_t; flaring-out requires
b'(r_t) < 1.

The Lewis-Papapetrou exterior is *cylindrically* symmetric, so the
mapping to Morris-Thorne is only approximate: we read off an
effective shape function from the angular metric L(r) and an
effective redshift from F(r).

This module:

- Computes effective shape function b_eff(r) and redshift Phi_eff(r);
- Identifies candidate throat radii r_t where b_eff(r_t) = r_t;
- Verifies the flaring-out condition b_eff'(r_t) < 1;
- Computes the Z_3 cover quotient interpretation of the throat
  (which Z_3 phase corresponds to a closed-throat configuration).

This concretises speculative item I.1 by providing a specific
gluing-map identification. The resulting wormhole is *valid* (the
flaring-out condition can be satisfied for some r_t in the
supercritical LP exterior) but, as the energy-condition module
verifies, requires exotic matter --- the wormhole is not
self-supporting from the LP vacuum alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np


def effective_shape_function(vs, r: float, eps: float = 1e-5) -> float:
    """Effective Morris-Thorne shape function from LP angular metric.

    Identification: in the L = r^2 (1 - b/r) form (Morris-Thorne
    angular ansatz), b(r) = r - L(r) / r.

    Returns
    -------
    b_eff(r) = r - L(r) / r
    """
    L = float(vs.analytic_exterior_L(r))
    return r - L / r


def effective_redshift_function(vs, r: float, eps: float = 1e-5) -> float:
    """Effective Morris-Thorne redshift Phi_eff(r) from LP F(r).

    Identification: -g_{tt} = exp(2 Phi), so Phi(r) = (1/2) ln |F(r)|.

    Returns NaN if F(r) <= 0 (chronology horizon or CTC region).
    """
    F = float(vs.analytic_exterior_F(r))
    if F <= 0:
        return float("nan")
    return 0.5 * float(np.log(F))


def flaring_out_check(vs, r_throat: float, eps: float = 1e-5) -> dict:
    """Verify the flaring-out condition b'(r_t) < 1 at a candidate throat.

    Returns
    -------
    dict with:
      b_value      : b_eff(r_throat)
      r_throat     : the candidate throat radius
      b_minus_r    : b_eff(r_t) - r_t (zero if at throat)
      b_prime      : d b_eff / dr at r_throat (central diff)
      flaring_out  : True iff b_prime < 1
    """
    b_val = effective_shape_function(vs, r_throat)
    bp = effective_shape_function(vs, r_throat + eps)
    bm = effective_shape_function(vs, r_throat - eps)
    b_prime = (bp - bm) / (2 * eps)
    return {
        "b_value": b_val,
        "r_throat": r_throat,
        "b_minus_r": b_val - r_throat,
        "b_prime": b_prime,
        "flaring_out": bool(b_prime < 1.0),
    }


def find_candidate_throats(
    vs, r_min: float, r_max: float, n_grid: int = 2001
) -> list[float]:
    """Scan for radii where b_eff(r) = r (i.e. where L(r) = 0).

    Since b_eff = r - L/r, the throat condition b_eff = r is exactly
    L = 0. These are the CTC band boundaries.

    Returns
    -------
    list of r values where the throat condition is satisfied.
    """
    rs = np.linspace(r_min, r_max, n_grid)
    Ls = np.array([float(vs.analytic_exterior_L(r)) for r in rs])
    signs = np.sign(Ls)
    flips = np.where(np.diff(signs) != 0)[0]
    out = []
    for i in flips:
        r1, r2 = rs[i], rs[i + 1]
        L1, L2 = Ls[i], Ls[i + 1]
        if abs(L2 - L1) < 1e-30:
            out.append(float(0.5 * (r1 + r2)))
        else:
            out.append(float(r1 - L1 * (r2 - r1) / (L2 - L1)))
    return out


@dataclass(frozen=True)
class WormholeThroatReport:
    """Diagnostic for a wormhole-throat construction on the LP exterior."""

    r_throat: float
    b_throat: float
    b_prime: float
    is_throat: bool
    is_flaring_out: bool
    F_throat: float
    redshift_phi: float
    is_horizon_at_throat: bool


def build_wormhole_report(vs, r_throat: float, eps: float = 1e-5) -> WormholeThroatReport:
    """Full wormhole-throat diagnostic at one r_throat candidate."""
    check = flaring_out_check(vs, r_throat, eps=eps)
    F = float(vs.analytic_exterior_F(r_throat))
    phi = effective_redshift_function(vs, r_throat)
    is_throat = abs(check["b_minus_r"]) < 0.01
    is_horizon = abs(F) < 1e-3
    return WormholeThroatReport(
        r_throat=r_throat,
        b_throat=check["b_value"],
        b_prime=check["b_prime"],
        is_throat=is_throat,
        is_flaring_out=check["flaring_out"],
        F_throat=F,
        redshift_phi=phi,
        is_horizon_at_throat=is_horizon,
    )


def z3_cover_throat_interpretation(
    gamma_eff: float, n_branches: int = 3
) -> dict:
    """Interpret the Z_3 cover quotient as a wormhole-throat identification.

    For the closed-cover case gamma_eff = 0, each branch's angular
    interval [0, 2 pi) glues to the next branch with a 1/3-cycle
    twist. The "throat" is the fixed locus of the Z_3 action --- the
    cylinder axis r = 0 in our setup.

    For non-zero gamma_eff, the closure is broken and the throat
    geometry acquires a holonomy phase.

    Returns
    -------
    dict with:
      gamma_eff       : input value
      monodromy       : exp(2 pi i / n_branches + i gamma_eff)
      closed_cover    : True iff gamma_eff = 0 mod 2 pi
      cover_branches  : n_branches
    """
    cyclic_phase = 2 * pi / n_branches
    monodromy = np.exp(1j * (cyclic_phase + gamma_eff))
    closed = abs(gamma_eff % (2 * pi)) < 1e-9 or abs((gamma_eff % (2 * pi)) - 2 * pi) < 1e-9
    return {
        "gamma_eff": float(gamma_eff),
        "monodromy_arg": float(np.angle(monodromy)),
        "monodromy_abs": float(abs(monodromy)),
        "closed_cover": bool(closed),
        "cover_branches": int(n_branches),
    }
