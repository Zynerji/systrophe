"""Kerr (1963) rotating black hole — analytic extension contains CTCs.

The Kerr metric in Boyer-Lindquist coordinates (t, r, theta, phi):

    ds^2 = -(1 - 2 M r / Sigma) dt^2
           - (4 M a r sin^2 theta / Sigma) dt dphi
           + (Sigma / Delta) dr^2
           + Sigma dtheta^2
           + (r^2 + a^2 + 2 M a^2 r sin^2 theta / Sigma) sin^2 theta dphi^2

where  Sigma = r^2 + a^2 cos^2 theta,  Delta = r^2 - 2 M r + a^2.

Closed timelike curves
----------------------
g_{phi phi} = sin^2 theta [ r^2 + a^2 + 2 M a^2 r sin^2 theta / Sigma ].

For sin^2 theta > 0, the bracket is negative iff
    r^2 + a^2 + 2 M a^2 r sin^2 theta / Sigma < 0.

For r > 0 every term is non-negative -- no CTCs.

For r < 0 (the analytic-maximal extension through the ring singularity
at r = 0, theta = pi/2), the cross term 2 M a^2 r sin^2 theta / Sigma is
negative, and as r approaches the ring singularity it dominates,
producing CTCs. In the equatorial plane (theta = pi/2)
    g_{phi phi} = r^2 + a^2 + 2 M a^2 / r,
which is negative when (multiplying by r < 0 flips the sign)
    r^3 + a^2 r + 2 M a^2 > 0.
The boundary is the unique real root r_CTC < 0; the CTC region is
the interval (r_CTC, 0), bounded above by the ring singularity.

Sub-extremal Kerr (a < M) has horizons at r_+- = M +- sqrt(M^2 - a^2),
hiding the CTC region behind the inner horizon. Over-extremal Kerr
(a > M) has no horizons; the ring singularity and the CTC region are
exposed (a "naked time machine"). Extremal Kerr (a = M) is the
boundary case.

Reference
---------
R. P. Kerr, *Gravitational field of a spinning mass as an example of
algebraically special metrics*, Phys. Rev. Lett. 11 (1963) 237.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KerrSpacetime:
    """Kerr black hole / naked-singularity spacetime in Boyer-Lindquist coords.

    Parameters
    ----------
    M : float
        Mass parameter (positive).
    a : float
        Spin parameter (angular momentum per unit mass). May be any real;
        |a| > M corresponds to an over-extremal naked singularity.
    """

    M: float
    a: float

    def __post_init__(self) -> None:
        if self.M <= 0:
            raise ValueError("M must be positive")

    def Sigma(self, r: float | np.ndarray, theta: float | np.ndarray) -> np.ndarray:
        r_a = np.asarray(r, dtype=float)
        t_a = np.asarray(theta, dtype=float)
        return r_a * r_a + self.a * self.a * np.cos(t_a) ** 2

    def Delta(self, r: float | np.ndarray) -> np.ndarray:
        r_a = np.asarray(r, dtype=float)
        return r_a * r_a - 2.0 * self.M * r_a + self.a * self.a

    def gtt(self, r: float | np.ndarray, theta: float | np.ndarray) -> np.ndarray:
        r_a = np.asarray(r, dtype=float)
        return -(1.0 - 2.0 * self.M * r_a / self.Sigma(r, theta))

    def gtphi(self, r: float | np.ndarray, theta: float | np.ndarray) -> np.ndarray:
        r_a = np.asarray(r, dtype=float)
        t_a = np.asarray(theta, dtype=float)
        return -2.0 * self.M * self.a * r_a * np.sin(t_a) ** 2 / self.Sigma(r, theta)

    def gphiphi(self, r: float | np.ndarray, theta: float | np.ndarray) -> np.ndarray:
        r_a = np.asarray(r, dtype=float)
        t_a = np.asarray(theta, dtype=float)
        s2 = np.sin(t_a) ** 2
        return s2 * (
            r_a * r_a + self.a * self.a
            + 2.0 * self.M * self.a * self.a * r_a * s2 / self.Sigma(r, theta)
        )

    @property
    def is_overextremal(self) -> bool:
        """True iff |a| > M (naked singularity, no horizons)."""
        return abs(self.a) > self.M

    @property
    def is_extremal(self) -> bool:
        return abs(self.a) == self.M

    @property
    def event_horizons(self) -> Optional[tuple[float, float]]:
        """Outer/inner horizons (r_+, r_-) for sub-extremal Kerr.

        Returns None if over-extremal (no horizons).
        """
        disc = self.M ** 2 - self.a ** 2
        if disc < 0:
            return None
        d = sqrt(disc)
        return (self.M + d, self.M - d)

    def has_local_ctc(self, r: float, theta: float) -> bool:
        """True iff a closed phi-orbit at fixed (t, r, theta) is timelike."""
        return float(self.gphiphi(r, theta)) < 0.0

    def equatorial_ctc_radius(self) -> Optional[float]:
        """Equatorial-plane CTC threshold: largest r < 0 root of r^3 + a^2 r + 2 M a^2 = 0.

        The equatorial CTC region is the interval (r_CTC, 0): for r < r_CTC
        the spacetime is normal, and for r_CTC < r < 0 the closed
        phi-orbit is timelike (g_{phi phi} < 0). The ring singularity
        bounds the region above.
        Returns None if no real negative root exists (Schwarzschild a = 0).
        """
        if self.a == 0.0:
            return None
        # Cubic r^3 + a^2 r + 2 M a^2 = 0
        coeffs = [1.0, 0.0, self.a ** 2, 2.0 * self.M * self.a ** 2]
        roots = np.roots(coeffs)
        real_neg = [float(z.real) for z in roots if abs(z.imag) < 1e-9 and z.real < 0]
        if not real_neg:
            return None
        return max(real_neg)  # closest to zero (the CTC boundary)

    def schwarzschild_limit(self) -> bool:
        """True iff a = 0 (no rotation, no CTCs anywhere)."""
        return self.a == 0.0
