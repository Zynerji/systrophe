"""Gott pair — closed timelike curves from a pair of moving cosmic strings.

Gott (1991) showed that two parallel cosmic strings of mass per unit
length mu, passing each other in their respective spacetimes with
relative velocity v perpendicular to the strings, generate closed
timelike curves *outside* both strings (encircling both) when the
boost is large enough relative to each string's conical deficit.

Geometry
--------
A cosmic string (Vilenkin 1981) has a conical exterior:

    ds^2 = -dt^2 + dr^2 + (1 - 4 G mu)^2 r^2 dphi^2 + dz^2,

with deficit angle Delta = 8 pi G mu around the string. Two parallel
strings boost in opposite directions in their relative-rest frame; the
total spacetime is the union of two such conical exteriors with a
coordinate identification.

Gott CTC condition (symmetric pair, mass per unit length mu each,
relative velocity v in the COM frame of the pair):

    gamma * v > tan(4 pi G mu)             (Gott 1991, eq. 13)

equivalently  gamma > sec(4 pi G mu)
or v_crit > sin(4 pi G mu)  for the threshold velocity.

Working in geometrised units (G = c = 1), define:

    alpha = 4 pi mu       (each string's deficit fraction per pi)

valid for 0 < alpha < pi / 2 (sub-extremal cosmic strings). The CTC
condition becomes simply  gamma * v > tan(alpha).

Reference: J. R. Gott III, *Closed timelike curves produced by pairs of
moving cosmic strings: exact solutions*, Phys. Rev. Lett. 66 (1991)
1126.

Implementation note
-------------------
This module implements only the *symmetric* Gott pair (equal masses,
equal-and-opposite velocities). The asymmetric case requires the more
general analysis of Cutler (1992); we leave that out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, pi, sin, sqrt, tan

import numpy as np


@dataclass(frozen=True)
class GottPair:
    """Symmetric pair of parallel cosmic strings (Gott 1991).

    Parameters
    ----------
    mu : float
        Mass per unit length of each string (geometrised units G = c = 1).
        The deficit angle of each string is 8 pi mu; the conical factor
        is 1 - 4 mu. Required: 0 < mu < 1/8 (so 4 pi mu < pi / 2,
        the sub-extremal regime).
    v : float
        Relative velocity between the two strings in their COM frame
        (each string moves at v/2 in opposite directions in some other
        common frame, but we parameterise by the relative speed v
        directly). Required: 0 <= v < 1.
    """

    mu: float
    v: float

    def __post_init__(self) -> None:
        if not (0.0 < self.mu < 0.125):
            raise ValueError(
                "mu must lie in (0, 1/8) for a sub-extremal cosmic string "
                "(deficit < pi)"
            )
        if not (0.0 <= self.v < 1.0):
            raise ValueError("v must lie in [0, 1) (subluminal relative motion)")

    @property
    def alpha(self) -> float:
        """Per-string angular parameter alpha = 4 pi mu (in radians)."""
        return 4.0 * pi * self.mu

    @property
    def deficit_angle(self) -> float:
        """Conical deficit angle of each string: Delta = 8 pi mu."""
        return 8.0 * pi * self.mu

    @property
    def gamma(self) -> float:
        """Lorentz factor for the relative velocity."""
        return 1.0 / sqrt(1.0 - self.v * self.v)

    @property
    def critical_velocity(self) -> float:
        """v at which the CTC condition becomes marginal: v_crit = sin(alpha).

        Above v_crit, gamma * v > tan(alpha) and CTCs exist.
        """
        return sin(self.alpha)

    @property
    def critical_gamma(self) -> float:
        """gamma at which the CTC condition becomes marginal: gamma = sec(alpha)."""
        return 1.0 / cos(self.alpha)

    def has_ctc(self) -> bool:
        """True iff Gott's symmetric-pair CTC condition is satisfied."""
        return self.gamma * self.v > tan(self.alpha)

    def ctc_margin(self) -> float:
        """gamma * v - tan(alpha). Positive iff CTCs exist; magnitude
        gives the margin above (or below) threshold."""
        return self.gamma * self.v - tan(self.alpha)

    def critical_velocity_for_mu(self, mu_target: float) -> float:
        """For a given mu_target, return the threshold v above which CTCs form."""
        if not (0.0 < mu_target < 0.125):
            raise ValueError("mu_target must lie in (0, 1/8)")
        alpha_t = 4.0 * pi * mu_target
        return sin(alpha_t)


def gott_critical_velocity(mu: float) -> float:
    """Threshold velocity v_crit = sin(4 pi mu) above which CTCs form.

    This is the standalone function form of GottPair(mu, v).critical_velocity.
    """
    if not (0.0 < mu < 0.125):
        raise ValueError("mu must lie in (0, 1/8)")
    return sin(4.0 * pi * mu)


def gott_critical_mu(v: float) -> float:
    """Threshold mass per unit length above which a given v admits CTCs.

    Inverts v = sin(4 pi mu) at the CTC threshold; mu_crit = arcsin(v) / (4 pi).
    """
    if not (0.0 < v < 1.0):
        raise ValueError("v must lie in (0, 1)")
    from math import asin
    return asin(v) / (4.0 * pi)
