"""PocketGeometry: trapped-null pocket descriptor and closure residuals.

The pocket is a closed null geodesic on the hybrid Schwarzschild + van
Stockum cylinder exterior. We carve it at a prescribed radius
``r_target`` by solving for the Schwarzschild mass M that places the
photon sphere there, then report:

* the impact parameter b(r_ph) of the closed orbit,
* the orbital angular velocity Ω,
* the stability indicator d²b/dr²(r_ph) > 0,
* closure residuals (how flat db/dr is at r_ph, how symmetric b is
  around r_ph).
"""

from __future__ import annotations

from dataclasses import dataclass

from systrophe.geometry.photon_sphere import (
    _db_dr_hybrid,
    _d2b_dr2_hybrid,
    effective_b_hybrid,
    engineer_photon_sphere_via_mass,
    hybrid_photon_sphere_omega,
    hybrid_photon_sphere_stability,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class PocketGeometry:
    """Closed-null-pocket descriptor.

    Attributes
    ----------
    r_target : float
        Requested pocket radius.
    M_engineered : float | None
        Schwarzschild mass that places a photon sphere at r_target,
        or None if no such M exists in the search range.
    branch : str
        "prograde" or "retrograde".
    impact_parameter : float
        b(r_ph) of the closed orbit.
    omega_orbit : float
        Orbital angular velocity Ω.
    is_stable : bool
        True iff d²b/dr²(r_ph) > 0 — i.e. trapped (a small perturbation
        away from r_ph experiences a restoring force).
    closure_residual_dbdr : float
        |db/dr|(r_ph). Should be ~ 0 if carving was successful.
    closure_residual_symmetry : float
        |b(r_ph+δ) - 2 b(r_ph) + b(r_ph-δ)| / |b(r_ph)| — measures
        symmetry of b around r_ph (proportional to |d²b/dr²|).
    schwarzschild_limit_M : float
        r_target / 3, the M that would carve r_target in pure
        Schwarzschild. Use as a sanity reference.
    """

    r_target: float
    M_engineered: float | None
    branch: str
    impact_parameter: float
    omega_orbit: float
    is_stable: bool
    closure_residual_dbdr: float
    closure_residual_symmetry: float
    schwarzschild_limit_M: float

    @property
    def is_carved(self) -> bool:
        """True iff an M_engineered exists and the residual is tight."""
        return (
            self.M_engineered is not None
            and abs(self.closure_residual_dbdr) < 1e-3
        )


def closure_residual(vs: VanStockumInterior, r_target: float, M: float,
                     branch: str = "prograde", delta: float = 1e-3) -> dict:
    """Closure residuals of the trapped-null pocket at r_target.

    Parameters
    ----------
    vs : VanStockumInterior
    r_target : float
        The carved pocket radius.
    M : float
        Schwarzschild mass placing the photon sphere at r_target.
    branch : {"prograde", "retrograde"}
    delta : float
        Finite-difference step for symmetry residual.

    Returns
    -------
    dict
        ``dbdr`` (should be ~ 0), ``symmetry`` (|2nd derivative| /
        |b(r_ph)|), and ``stability`` (sign of d²b/dr² — True =
        trapped).
    """
    dbdr = _db_dr_hybrid(vs, r_target, M, branch=branch)
    b0 = effective_b_hybrid(vs, r_target, M, branch=branch)
    b_plus = effective_b_hybrid(vs, r_target + delta, M, branch=branch)
    b_minus = effective_b_hybrid(vs, r_target - delta, M, branch=branch)
    second_deriv = (b_plus - 2.0 * b0 + b_minus) / (delta ** 2)
    sym = abs(second_deriv) / max(abs(b0), 1e-12)
    return {
        "dbdr": float(dbdr),
        "symmetry": float(sym),
        "stability": bool(second_deriv > 0),
    }


def carve_photon_pocket(vs: VanStockumInterior, r_target: float,
                          branch: str = "prograde",
                          M_range: tuple[float, float] = (0.01, 5.0),
                          delta: float = 1e-3) -> PocketGeometry:
    """Carve a closed null pocket at r_target on the hybrid spacetime.

    Steps:
      1. Solve for Schwarzschild mass M that places a photon sphere at
         r_target (Brent search in M_range).
      2. Measure the impact parameter, orbital Ω, and stability there.
      3. Report closure residuals.

    Returns a PocketGeometry. If no M exists in range, returns a
    descriptor with ``M_engineered=None`` and NaN residuals.
    """
    M = engineer_photon_sphere_via_mass(
        vs, r_target=r_target, branch=branch, M_range=M_range,
    )
    schw_limit = r_target / 3.0
    if M is None:
        return PocketGeometry(
            r_target=float(r_target),
            M_engineered=None,
            branch=branch,
            impact_parameter=float("nan"),
            omega_orbit=float("nan"),
            is_stable=False,
            closure_residual_dbdr=float("nan"),
            closure_residual_symmetry=float("nan"),
            schwarzschild_limit_M=float(schw_limit),
        )
    b = effective_b_hybrid(vs, r_target, M, branch=branch)
    om = hybrid_photon_sphere_omega(vs, r_target, M, branch=branch)
    stable = hybrid_photon_sphere_stability(vs, r_target, M, branch=branch)
    res = closure_residual(vs, r_target, M, branch=branch, delta=delta)
    return PocketGeometry(
        r_target=float(r_target),
        M_engineered=float(M),
        branch=branch,
        impact_parameter=float(b),
        omega_orbit=float(om),
        is_stable=stable,
        closure_residual_dbdr=res["dbdr"],
        closure_residual_symmetry=res["symmetry"],
        schwarzschild_limit_M=float(schw_limit),
    )
