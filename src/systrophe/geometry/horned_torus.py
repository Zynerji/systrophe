"""Horned-torus topology on the LP angular structure.

The Lewis-Papapetrou exterior has angular metric g_phi_phi(r) = L(r) per
(r, z) slice. A "horned torus" quotient introduces a z-dependent
modulation:

    L_horned(r, z) = h(z) * L(r)

with a *horn profile* h(z). Two physically distinct families:

    - regular (pinch): h(z) = h_min + (1 - h_min) * tanh^2(z / sigma).
      The phi-circle is pinched at z = 0 (h -> h_min); far away, h -> 1.

    - inverted (bulge): h(z) = 1 + (h_max - 1) * exp(-(z/sigma)^2).
      The phi-circle is bulged at z = 0 (h -> h_max); far away, h -> 1.

In both cases h(z) > 0 everywhere, so the *sign* of L_horned matches
L(r) at every (r, z): the r-CTC bands are unchanged. What changes
is the *proper length* of each CTC revolution. The regular horn
*shortens* CTCs at the pinch; the inverted horn *lengthens* them at
the bulge.

The proper-length diagnostic

    ctc_traversal_proper_area = integral_{L_horned < 0} sqrt(|L_horned|) dr dz

is a single-number figure of merit for the effective CTC budget of the
horned geometry.

Topology
--------
For h_min > 0, the regular horn is a *thinned* torus, same genus as
T^2. The inverted horn is a *fattened* torus, also same genus. In the
limit h_min -> 0, the regular horn becomes a *true* horned torus
(topology change: the phi-fibre degenerates to a point at z = 0); in
the limit h_max -> infinity, the inverted horn becomes asymptotically
flat in the z direction (no topology change).

This module exposes both modes and a clean diagnostic comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Callable, Literal

import numpy as np


def regular_horn_profile(
    z: float | np.ndarray, h_min: float = 0.1, sigma: float = 1.0
) -> np.ndarray:
    """Regular horn: h(z) = h_min + (1 - h_min) tanh^2(z / sigma).

    h(0) = h_min (pinch), h(|z| >> sigma) -> 1.
    """
    if not (0.0 < h_min <= 1.0):
        raise ValueError("h_min must be in (0, 1]")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = np.asarray(z, dtype=float)
    return h_min + (1.0 - h_min) * np.tanh(z / sigma) ** 2


def inverted_horn_profile(
    z: float | np.ndarray, h_max: float = 2.0, sigma: float = 1.0
) -> np.ndarray:
    """Inverted horn (bulge): h(z) = 1 + (h_max - 1) exp(-(z/sigma)^2).

    h(0) = h_max (bulge), h(|z| >> sigma) -> 1.
    """
    if h_max < 1.0:
        raise ValueError("h_max must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z = np.asarray(z, dtype=float)
    return 1.0 + (h_max - 1.0) * np.exp(-((z / sigma) ** 2))


@dataclass(frozen=True)
class HornedTorus:
    """Horned-torus modulation of an LP angular structure.

    Parameters
    ----------
    L_func : Callable[[np.ndarray], np.ndarray]
        Base L(r) = g_phi_phi(r) from the LP exterior (e.g. a
        SystrophePair.L or TiplerSinusoid.L).
    mode : {"regular", "inverted"}
        Horn type (pinch or bulge at z = 0).
    h_param : float
        For regular: h_min (in (0, 1]); for inverted: h_max (>= 1).
    sigma : float
        Horn width in z.
    """

    L_func: Callable
    mode: Literal["regular", "inverted"] = "regular"
    h_param: float = 0.1
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("regular", "inverted"):
            raise ValueError(f"mode must be 'regular' or 'inverted', got {self.mode!r}")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.mode == "regular" and not (0.0 < self.h_param <= 1.0):
            raise ValueError("regular horn requires h_param in (0, 1]")
        if self.mode == "inverted" and self.h_param < 1.0:
            raise ValueError("inverted horn requires h_param >= 1")

    def h(self, z: float | np.ndarray) -> np.ndarray:
        if self.mode == "regular":
            return regular_horn_profile(z, h_min=self.h_param, sigma=self.sigma)
        return inverted_horn_profile(z, h_max=self.h_param, sigma=self.sigma)

    def L_horned(self, r: float | np.ndarray, z: float | np.ndarray) -> np.ndarray:
        """L_horned(r, z) = h(z) * L(r)."""
        return np.asarray(self.h(z)) * np.asarray(self.L_func(r))

    def ctc_indicator(
        self, r: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Boolean indicator of CTC region on the (r, z) grid.

        Since h(z) > 0 everywhere, this matches the L(r) sign and is
        constant in z.
        """
        r_g, z_g = np.meshgrid(r, z, indexing="ij")
        return self.L_horned(r_g, z_g) < 0

    def ctc_traversal_proper_area(
        self,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
        n_r: int = 401,
        n_z: int = 201,
    ) -> float:
        """integral_{L<0} sqrt(|L_horned|) dr dz over the box.

        A single-number figure of merit for the effective CTC budget.
        Regular horn (pinch) decreases this vs h = 1; inverted horn
        (bulge) increases it.
        """
        r = np.linspace(r_min, r_max, n_r)
        z = np.linspace(z_min, z_max, n_z)
        r_g, z_g = np.meshgrid(r, z, indexing="ij")
        L_h = self.L_horned(r_g, z_g)
        mask = L_h < 0
        integrand = np.zeros_like(L_h)
        integrand[mask] = np.sqrt(-L_h[mask])
        # Trapezoid integration
        dr = (r_max - r_min) / (n_r - 1)
        dz = (z_max - z_min) / (n_z - 1)
        return float(np.sum(integrand) * dr * dz)

    def topology_class(self) -> str:
        """Return a short topology classification string.

        Modes:
          - 'pinch_h_min_0': regular horn with h_param == 0 (true horn,
                            topology change)
          - 'thinned_T2'   : regular horn with h_param > 0
          - 'fattened_T2'  : inverted horn (h_param > 1)
          - 'flat_T2'      : h_param exactly 1 (no modulation)
        """
        if self.mode == "regular":
            if self.h_param <= 1e-12:
                return "pinch_h_min_0"
            if abs(self.h_param - 1.0) < 1e-12:
                return "flat_T2"
            return "thinned_T2"
        # inverted
        if abs(self.h_param - 1.0) < 1e-12:
            return "flat_T2"
        return "fattened_T2"


def compare_horn_modes(
    L_func: Callable,
    h_min: float = 0.1,
    h_max: float = 2.0,
    sigma: float = 1.0,
    r_min: float = 1.0,
    r_max: float = 10.0,
    z_min: float = -3.0,
    z_max: float = 3.0,
    n_r: int = 401,
    n_z: int = 201,
) -> dict:
    """Compare regular vs inverted horns on the same base L(r).

    Returns:
      - regular_proper_area : CTC proper area for the pinch
      - inverted_proper_area: CTC proper area for the bulge
      - flat_proper_area    : reference value with h = 1 (no horn)
      - regular_fraction    : regular_proper_area / flat_proper_area
      - inverted_fraction   : inverted_proper_area / flat_proper_area
      - ratio_inv_over_reg  : inverted / regular
    """
    regular = HornedTorus(L_func=L_func, mode="regular",
                          h_param=h_min, sigma=sigma)
    inverted = HornedTorus(L_func=L_func, mode="inverted",
                           h_param=h_max, sigma=sigma)
    flat = HornedTorus(L_func=L_func, mode="regular",
                       h_param=1.0, sigma=sigma)

    A_reg = regular.ctc_traversal_proper_area(r_min, r_max, z_min, z_max, n_r, n_z)
    A_inv = inverted.ctc_traversal_proper_area(r_min, r_max, z_min, z_max, n_r, n_z)
    A_flat = flat.ctc_traversal_proper_area(r_min, r_max, z_min, z_max, n_r, n_z)

    return {
        "regular_proper_area": A_reg,
        "inverted_proper_area": A_inv,
        "flat_proper_area": A_flat,
        "regular_fraction": A_reg / A_flat if A_flat > 0 else float("nan"),
        "inverted_fraction": A_inv / A_flat if A_flat > 0 else float("nan"),
        "ratio_inv_over_reg": A_inv / A_reg if A_reg > 0 else float("nan"),
    }


def horn_circumference_at_z(L_at_r_band: float, z: float, horn: HornedTorus) -> float:
    """Proper circumference of the phi-circle at (r, z) for a given band L(r).

    For L(r) > 0 (timelike phi-orbit allowed), proper circumference at
    height z is 2 pi sqrt(h(z) L(r)). For L(r) < 0 (CTC region), proper
    length is 2 pi sqrt(|h(z) L(r)|) but the orbit is closed-timelike.
    """
    h_at_z = float(horn.h(z))
    return 2 * pi * float(np.sqrt(abs(h_at_z * L_at_r_band)))
