"""Goedel rotating-dust universe (Goedel 1949) — closed-form CTC spacetime.

The Goedel universe is the simplest exact solution of Einstein's
equations containing closed timelike curves. It is a homogeneous
rotating dust spacetime with cosmological constant; CTCs exist through
every point.

Metric in cylindrical-type coordinates (t, r, phi, z) with
fundamental length scale `a`:

    ds^2 = a^2 [ -(dt + Sigma(r) dphi)^2 + dr^2 + dz^2 + D(r)^2 dphi^2 ]

where Sigma(r) = sqrt(2) sinh^2(r) and D(r) = sinh(r) cosh(r).

Equivalently, expanding:

    g_tt    = -a^2
    g_tphi  = -a^2 Sigma(r) = -a^2 sqrt(2) sinh^2(r)
    g_phiphi = a^2 (D(r)^2 - Sigma(r)^2) = a^2 sinh^2(r) (cosh^2(r) - 2 sinh^2(r))
            = a^2 sinh^2(r) (1 - sinh^2(r))    (using cosh^2 = 1 + sinh^2)
    g_rr    = a^2
    g_zz    = a^2

CTC condition: g_phiphi < 0 iff sinh^2(r) > 1, i.e.,
    r > r_CTC = arcsinh(1) = ln(1 + sqrt(2)) ≈ 0.8814.

Stress-energy: dust + cosmological constant Lambda, with the dust
4-velocity being the timelike Killing vector partial_t. Density
rho = 1/(8 pi a^2) and Lambda = -1/(2 a^2) in c = G = 1 units.

Reference: K. Goedel, *An example of a new type of cosmological
solutions of Einstein's field equations of gravitation*, Rev. Mod.
Phys. 21 (1949) 447.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import asinh, log, sqrt

import numpy as np


def godel_ctc_radius() -> float:
    """The Goedel CTC threshold radius r_CTC = arcsinh(1) = ln(1 + sqrt(2))."""
    return asinh(1.0)


@dataclass(frozen=True)
class GodelUniverse:
    """Goedel rotating-dust universe.

    Parameters
    ----------
    a : float
        Fundamental length scale of the rotating universe (positive).
        Sets the time scale, the CTC radius (in geometrical units), and
        the dust density rho = 1/(8 pi a^2).
    """

    a: float = 1.0

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("a must be positive")

    @property
    def dust_density(self) -> float:
        """rho = 1/(8 pi a^2) (c = G = 1)."""
        return 1.0 / (8.0 * np.pi * self.a * self.a)

    @property
    def cosmological_constant(self) -> float:
        """Lambda = -1/(2 a^2) (c = G = 1)."""
        return -1.0 / (2.0 * self.a * self.a)

    @property
    def angular_velocity(self) -> float:
        """The Goedel rotation rate omega = 1/a (c = G = 1)."""
        return 1.0 / self.a

    def gphiphi(self, r: float | np.ndarray) -> np.ndarray:
        """g_{phi phi}(r) = a^2 sinh^2(r) (1 - sinh^2(r)).

        Negative for r > arcsinh(1), where closed phi-orbits become CTCs.
        """
        r_arr = np.asarray(r, dtype=float)
        s = np.sinh(r_arr)
        return self.a * self.a * s * s * (1.0 - s * s)

    def gtphi(self, r: float | np.ndarray) -> np.ndarray:
        """g_{t phi}(r) = -a^2 sqrt(2) sinh^2(r)."""
        r_arr = np.asarray(r, dtype=float)
        return -self.a * self.a * sqrt(2.0) * np.sinh(r_arr) ** 2

    def gtt(self, r: float | np.ndarray = 0.0) -> np.ndarray:
        """g_{tt} = -a^2 (independent of r in Goedel coordinates)."""
        r_arr = np.asarray(r, dtype=float)
        return np.full_like(r_arr, -self.a * self.a)

    def grr(self, r: float | np.ndarray = 0.0) -> np.ndarray:
        """g_{rr} = a^2."""
        r_arr = np.asarray(r, dtype=float)
        return np.full_like(r_arr, self.a * self.a)

    def gzz(self, r: float | np.ndarray = 0.0) -> np.ndarray:
        """g_{zz} = a^2."""
        r_arr = np.asarray(r, dtype=float)
        return np.full_like(r_arr, self.a * self.a)

    @property
    def ctc_threshold_radius(self) -> float:
        """r_CTC = arcsinh(1) ~ 0.8814 (in units where the radial coord is dimensionless)."""
        return asinh(1.0)

    def has_local_ctc(self, r: float) -> bool:
        """True iff a closed phi-orbit at fixed (t, r, z) is timelike."""
        return float(self.gphiphi(r)) < 0.0

    def find_ctc_radial_interval(self, r_max: float) -> tuple[float, float] | None:
        """Return (r_inner, r_outer) of the CTC region [r_CTC, r_max] (one band).

        In Goedel, the CTC region is a single half-open ray r > r_CTC; we
        truncate to [r_CTC, r_max] for finite reporting. Returns None if
        r_max <= r_CTC.
        """
        if r_max <= self.ctc_threshold_radius:
            return None
        return (self.ctc_threshold_radius, float(r_max))

    def proper_phi_circumference(self, r: float) -> float:
        """Proper length of a closed phi-orbit at fixed (t, r, z), 2 pi sqrt(|g_phiphi|).

        Real for r < r_CTC (positive g_phiphi), imaginary (CTC sign) for r > r_CTC.
        Returns the magnitude in either case; the sign of g_phiphi is the
        diagnostic for causality character.
        """
        return 2.0 * np.pi * float(np.sqrt(abs(self.gphiphi(r))))
