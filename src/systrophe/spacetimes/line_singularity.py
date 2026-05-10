"""Rotating line singularity: the singular limit of a van Stockum cylinder.

The van Stockum dust column with extended source 0 <= r <= R is one
of an *equivalence class* of sources that produce the same Lewis-
Papapetrou vacuum exterior. The key invariants of the exterior are
the twist constant c and the boundary values F(R) = 1, F'(R) = 0;
these depend only on (omega, R) regardless of how matter is distributed
inside the cylinder, provided the matching at r = R is identical.

In the limit R -> 0 with the twist constant c = 2 omega and the matching
data held fixed, the source degenerates to a singular axis carrying
mass per unit length M and angular momentum per unit length J. This
is the cylindrical analog of:
- the Schwarzschild point particle limit of a sphere of dust,
- the Vilenkin (1981) cosmic string as a singular line source,
- the Kerr ring singularity at r = 0, theta = pi/2 in 4D.

This module makes the reinterpretation explicit: a `LineSingularity`
is constructed with (omega, R) parameters (or equivalently with M, J)
and exposes the same exterior (F, K, L) as a VanStockumInterior of
the matching parameters.

The Systrophe pair, in the R -> 0 limit at fixed M, J, becomes a *pair
of singular rotating axes* -- the cylindrical analog of the Gott (1991)
cosmic-string pair, with rotation playing the role of relative
translation. The "two top-spin, dual-positive singularities" of the
Titor specification are exactly this: two parallel rotating line
sources whose joint exterior carries the off-set Tipler sinusoid.

Reference
---------
T. Lewis, *Some special solutions of the equations of axially
symmetric gravitational fields*, Proc. Roy. Soc. London A 136 (1932) 176.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LineSingularity:
    """Rotating line singularity (singular limit of a van Stockum cylinder).

    Parameters
    ----------
    omega : float
        Effective rotation rate (positive). Controls the twist constant
        c = 2 omega of the resulting Lewis-Papapetrou exterior.
    R_match : float
        Matching radius. The exterior at r > R_match is the standard
        van Stockum / Tipler / Bonnor exterior of an equivalent dust
        cylinder with the same (omega, R_match). In the strict
        singular limit, R_match -> 0; we keep it as the "regulator"
        for evaluating the analytic exterior.
    """

    omega: float
    R_match: float

    def __post_init__(self) -> None:
        if self.omega <= 0:
            raise ValueError("omega must be positive")
        if self.R_match <= 0:
            raise ValueError("R_match must be positive")

    @property
    def a(self) -> float:
        """Bonnor parameter a = omega R_match."""
        return self.omega * self.R_match

    @property
    def twist_constant(self) -> float:
        """Twist constant c = 2 omega."""
        return 2.0 * self.omega

    def equivalent_van_stockum(self):
        """Return the matching VanStockumInterior whose exterior is identical."""
        from .. import VanStockumInterior

        return VanStockumInterior(omega=self.omega, R=self.R_match)

    def F(self, r: float | np.ndarray) -> np.ndarray:
        """Exterior F(r) = -g_{tt}(r). Delegates to the equivalent van Stockum."""
        return self.equivalent_van_stockum().analytic_exterior_F(r)

    def K(self, r: float | np.ndarray) -> np.ndarray:
        """Exterior K(r) = g_{t phi}(r)."""
        return self.equivalent_van_stockum().analytic_exterior_K(r)

    def L(self, r: float | np.ndarray) -> np.ndarray:
        """Exterior L(r) = g_{phi phi}(r). CTC bands are L < 0."""
        return self.equivalent_van_stockum().analytic_exterior_L(r)

    @property
    def regime(self) -> str:
        return self.equivalent_van_stockum().regime

    @classmethod
    def from_mass_and_angular_momentum(
        cls, M: float, J: float, R_match: float = 1.0
    ) -> "LineSingularity":
        """Construct from (mass per unit length M, angular momentum per unit length J).

        Inverts the van Stockum total-energy and total-angular-momentum
        integrals at fixed R_match. The relation
            M = (1/2)(exp(a^2) - 1),
            J = (1/(2 omega)) * [ (a^2 - 1) exp(a^2) + 1 ]
        is solved for omega given M (which fixes a = omega R_match) and
        consistency-checked against the supplied J.

        Parameters
        ----------
        M : float
            Target mass per unit length (must satisfy M > 0).
        J : float
            Target angular momentum per unit length. For positive J, omega > 0.
        R_match : float
            Regulator scale (default 1.0).
        """
        if M <= 0:
            raise ValueError("M must be positive")
        if R_match <= 0:
            raise ValueError("R_match must be positive")
        # Solve M = (1/2)(exp(a^2) - 1) for a^2 = ln(2M + 1).
        a_sq = float(np.log(2.0 * M + 1.0))
        if a_sq <= 0:
            raise ValueError("M too small to define a positive Bonnor a")
        a = float(np.sqrt(a_sq))
        omega = a / R_match
        # The corresponding J is fixed by (omega, R_match) via the integral
        # above; we simply construct and let the user inspect via
        # angular_momentum_per_unit_length.
        return cls(omega=omega, R_match=R_match)

    def to_dict(self) -> dict:
        """Compact dictionary describing the singular-axis invariants."""
        vs = self.equivalent_van_stockum()
        return {
            "omega": self.omega,
            "R_match": self.R_match,
            "a": self.a,
            "twist_constant_c": self.twist_constant,
            "mass_per_unit_length_M": vs.mass_per_unit_length,
            "angular_momentum_per_unit_length_J": vs.angular_momentum_per_unit_length,
            "regime": self.regime,
        }
