"""Vilenkin (1981) cosmic string — non-rotating line singularity.

A cosmic string is a one-dimensional topological defect in spacetime,
mathematically a delta-function source on its 2D worldsheet. The exterior
metric of a straight static cosmic string of mass per unit length mu is
locally Minkowski with a *conical deficit*:

    ds^2 = -dt^2 + dr^2 + (1 - 4 G mu)^2 r^2 dphi^2 + dz^2,
    deficit angle Delta = 8 pi G mu.

In c = G = 1 geometrized units, the conical factor is 1 - 4 mu, valid
for mu in (0, 1/4) (sub-extremal, deficit < 2 pi).

A single static cosmic string contains *no* closed timelike curves: it
is locally Minkowski everywhere. CTCs require relative motion between
two strings (the Gott pair, see `gott.py`) or rotation (see
`line_singularity.py`).

This module exposes the conical metric, the deficit angle, and a
`compose_with_gott` convenience that hands two cosmic strings to a
`GottPair` constructor at a chosen relative velocity. This makes the
construction "two cosmic strings -> Gott time machine" explicit.

Reference
---------
A. Vilenkin, *Gravitational field of vacuum domain walls and strings*,
Phys. Rev. D 23 (1981) 852.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CosmicString:
    """Static, straight, non-rotating Vilenkin cosmic string.

    Parameters
    ----------
    mu : float
        Mass per unit length (geometrized G = c = 1). Required:
        0 < mu < 1/4 (sub-extremal, conical deficit < 2 pi).
    """

    mu: float

    def __post_init__(self) -> None:
        if not (0.0 < self.mu < 0.25):
            raise ValueError("mu must lie in (0, 1/4) for a sub-extremal cosmic string")

    @property
    def conical_factor(self) -> float:
        """1 - 4 mu (the prefactor of r dphi in the line element)."""
        return 1.0 - 4.0 * self.mu

    @property
    def deficit_angle(self) -> float:
        """Deficit angle Delta = 8 pi mu (in radians)."""
        return 8.0 * np.pi * self.mu

    def line_element_factor(self, r: float | np.ndarray) -> np.ndarray:
        """g_{phi phi}(r) = (1 - 4 mu)^2 r^2 (positive everywhere -- no CTCs)."""
        r_arr = np.asarray(r, dtype=float)
        return self.conical_factor ** 2 * r_arr ** 2

    def has_local_ctc(self, r: float, phi: float = 0.0) -> bool:
        """A static cosmic string has no CTCs anywhere."""
        return False

    def compose_with_gott(self, other: "CosmicString", v: float):
        """Form a Gott pair (relative velocity v) of this string and `other`.

        Both strings must have the same mass per unit length (the
        symmetric Gott construction; see `gott.py`).
        """
        from .gott import GottPair

        if not isinstance(other, CosmicString):
            raise TypeError("other must be a CosmicString")
        if abs(self.mu - other.mu) > 1e-12:
            raise ValueError(
                "symmetric Gott pair requires matched masses; mu1 != mu2"
            )
        return GottPair(mu=self.mu, v=v)
