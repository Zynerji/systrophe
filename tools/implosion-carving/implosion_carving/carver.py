"""ImplosionCarver: wrap a van Stockum exterior and carve a trapped-null pocket.

Top-level API for the implosion-carving tool. Models the LPAnalyser
pattern: one object per spacetime, methods carve the pocket at
requested radii, monodromy is computed once.
"""

from __future__ import annotations

from dataclasses import dataclass

from systrophe.vanstockum import VanStockumInterior

from .monodromy import (
    Z3MonodromySignature,
    closure_on_cover_residual,
    compute_z3_signature,
)
from .pocket import PocketGeometry, carve_photon_pocket


@dataclass
class ImplosionSummary:
    """Flat one-shot summary of a carved pocket + its Z_3 cover."""
    omega: float
    R: float
    r_target: float
    branch: str
    M_engineered: float | None
    schwarzschild_limit_M: float
    impact_parameter: float
    omega_orbit: float
    is_stable: bool
    closure_residual_dbdr: float
    closure_residual_symmetry: float
    is_carved: bool
    z3_N: int
    z3_triplet_convergence_error: float
    z3_closure_phase_residual: float

    def __repr__(self) -> str:
        return (
            f"ImplosionSummary(r_target={self.r_target}, "
            f"M_engineered={self.M_engineered}, is_carved={self.is_carved}, "
            f"is_stable={self.is_stable}, "
            f"closure_dbdr={self.closure_residual_dbdr:.2e})"
        )


class ImplosionCarver:
    """Carve trapped-null pockets into a Schwarzschild + van Stockum hybrid.

    The carver wraps a :class:`VanStockumInterior` and exposes methods
    for placing closed null geodesics at requested radii. A Z_3
    monodromy signature is computed lazily.

    Parameters
    ----------
    omega : float
        Rotation rate of the underlying cylinder (1/length; c=G=1).
    R : float
        Cylinder radius.
    M_range : tuple[float, float], optional
        Schwarzschild mass search range for the engineering step.
        Default (0.01, 5.0).
    """

    def __init__(self, omega: float, R: float,
                 M_range: tuple[float, float] = (0.01, 5.0)) -> None:
        self.vs = VanStockumInterior(omega=float(omega), R=float(R))
        self.M_range = (float(M_range[0]), float(M_range[1]))
        self._z3_cache: dict[int, Z3MonodromySignature] = {}

    @property
    def omega(self) -> float:
        return float(self.vs.omega)

    @property
    def R(self) -> float:
        return float(self.vs.R)

    @property
    def a(self) -> float:
        """Dimensionless rotation parameter a = omega * R."""
        return float(self.vs.a)

    def carve(self, r_target: float, branch: str = "prograde",
              delta: float = 1e-3) -> PocketGeometry:
        """Carve a closed null pocket at r_target."""
        return carve_photon_pocket(
            self.vs, r_target=r_target, branch=branch,
            M_range=self.M_range, delta=delta,
        )

    def z3_signature(self, N: int = 256) -> Z3MonodromySignature:
        """Z_3 monodromy signature on N nodes. Cached per N."""
        if N not in self._z3_cache:
            self._z3_cache[N] = compute_z3_signature(N=N)
        return self._z3_cache[N]

    def summary(self, r_target: float, branch: str = "prograde",
                z3_N: int = 256) -> ImplosionSummary:
        """One-shot summary of carve(r_target) + Z_3 signature.

        Fast (~ms): one Brent search + one finite-difference for the
        pocket, one closed-form spectrum for Z_3.
        """
        pocket = self.carve(r_target=r_target, branch=branch)
        sig = self.z3_signature(N=z3_N)
        return ImplosionSummary(
            omega=self.omega,
            R=self.R,
            r_target=float(r_target),
            branch=branch,
            M_engineered=pocket.M_engineered,
            schwarzschild_limit_M=pocket.schwarzschild_limit_M,
            impact_parameter=pocket.impact_parameter,
            omega_orbit=pocket.omega_orbit,
            is_stable=pocket.is_stable,
            closure_residual_dbdr=pocket.closure_residual_dbdr,
            closure_residual_symmetry=pocket.closure_residual_symmetry,
            is_carved=pocket.is_carved,
            z3_N=sig.N,
            z3_triplet_convergence_error=sig.triplet_convergence_error,
            z3_closure_phase_residual=closure_on_cover_residual(sig),
        )
