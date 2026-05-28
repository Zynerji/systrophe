"""Multi-binary "necklace" configuration for the Toroidal Knopp Drive.

Replaces a single counter-rotating Kerr binary with N identical binaries
arranged on a ring of radius R_ring. Each binary contributes its own
toroidal frame-dragging; the superposition extends the effective band
coverage along the ring.

Geometry
--------
N binaries at angular positions phi_k = 2 pi k / N (k = 0, ..., N-1) on
a ring of radius R_ring in the (x, y) plane. Each binary is a tight,
counter-rotating Kerr pair (oriented with its axis tangent to the ring,
say). The local toroidal CTC band of each binary lives in a small
neighbourhood of its location.

Effective frame-dragging
------------------------
At a test point (rho_ring, phi_test) on the ring at the same radius as
the binaries, the dominant frame-dragging contribution is from the
nearest binary. Far from any binary the LT field decays as 1/r^3, so
the necklace's effective tilt at angular position phi_test is

    T_eff^necklace(phi_test, R_ring)  =  sum_k T_eff_local(d_k(phi_test))

where d_k = 2 R_ring sin(|phi_test - phi_k|/2) is the chord distance
to binary k.

Findings
--------
- For sufficiently large N (close-packed binaries), the necklace
  T_eff becomes approximately uniform along the ring -- the band
  closes into a continuous "necklace band."
- The packing threshold N_min for continuous-band coverage scales
  with binary separation d and ring radius R_ring.
- The lifetime issue (each individual binary still merges in << 1
  orbit) is unchanged: the necklace inherits the SAME falsification
  per-binary. Only collective stabilising effects (mutual angular
  momentum exchange, ring-mode tidal coupling) would help, and those
  are not in this leading-order construction.

This is a CLASSICAL kinematic exploration -- the dynamical viability
of the necklace as a whole is its own open problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from systrophe.knopp.knopp_toroidal import EffectiveToroidalKerrBinary
from systrophe.knopp.knopp_toroidal_stability import time_to_merger


@dataclass(frozen=True)
class KnoppNecklace:
    """A ring of N identical counter-rotating Kerr binaries.

    Parameters
    ----------
    n_binaries : int
        Number of binaries on the ring (N >= 1).
    M : float
        Per-BH mass (each binary has 2 holes of mass M).
    d : float
        Per-binary separation.
    chi : float
        Per-binary spin magnitude.
    R_ring : float
        Radius of the ring on which the binaries sit.

    The N binaries are placed at uniform angular positions
    phi_k = 2 pi k / N on the ring.
    """

    n_binaries: int = 6
    M: float = 1.0
    d: float = 2.0
    chi: float = 1.0
    R_ring: float = 10.0

    def __post_init__(self) -> None:
        if self.n_binaries < 1:
            raise ValueError("n_binaries must be >= 1")
        if self.M <= 0 or self.d <= 0 or self.R_ring <= 0:
            raise ValueError("M, d, R_ring must be positive")
        if not 0.0 <= self.chi <= 1.0:
            raise ValueError("chi must be in [0, 1]")

    # ---- geometry ----

    def binary_angular_positions(self) -> list[float]:
        """phi_k = 2 pi k / N."""
        return [2.0 * math.pi * k / self.n_binaries
                for k in range(self.n_binaries)]

    def chord_to_kth_binary(self, phi_test: float, k: int) -> float:
        """Chord distance on the ring from phi_test to the k-th binary."""
        phi_k = 2.0 * math.pi * k / self.n_binaries
        delta = abs(phi_test - phi_k)
        delta = min(delta, 2.0 * math.pi - delta)
        return 2.0 * self.R_ring * math.sin(delta / 2.0)

    def nearest_binary_distance(self, phi_test: float) -> float:
        """Distance from phi_test to the nearest binary on the ring."""
        return min(
            self.chord_to_kth_binary(phi_test, k)
            for k in range(self.n_binaries)
        )

    def nearest_binary_angular_separation(self) -> float:
        """The angular gap between adjacent binaries: 2 pi / N."""
        return 2.0 * math.pi / self.n_binaries

    def adjacent_binary_chord(self) -> float:
        """Chord distance between two ADJACENT binaries on the ring."""
        return 2.0 * self.R_ring * math.sin(math.pi / self.n_binaries)

    # ---- frame-dragging (additive over binaries) ----

    def t_eff_local(self, distance_from_binary: float) -> float:
        """Per-binary T_eff at distance d_local from one binary's centre.

        Uses the linear LT formula of EffectiveToroidalKerrBinary:
            T_eff_local(d_local) = 4 chi M^2 (d_local)^2
                                   / [(d/2)^2 + d_local^2]^(3/2).
        """
        return float(
            4.0 * self.chi * self.M ** 2 * distance_from_binary ** 2
            / ((self.d / 2.0) ** 2 + distance_from_binary ** 2) ** 1.5
        )

    def t_eff_necklace(self, phi_test: float) -> float:
        """Total T_eff at angular position phi_test on the ring, summed
        over all N binaries."""
        return float(sum(
            self.t_eff_local(self.chord_to_kth_binary(phi_test, k))
            for k in range(self.n_binaries)
        ))

    # ---- continuous-coverage test ----

    def is_continuously_banded(self, n_test: int = 256) -> bool:
        """True iff T_eff_necklace(phi) >= 1 for *all* phi on the ring,
        meaning the CTC band wraps continuously around the necklace."""
        phis = np.linspace(0.0, 2.0 * math.pi, n_test, endpoint=False)
        return bool(all(
            self.t_eff_necklace(float(p)) >= 1.0 for p in phis
        ))

    def min_t_eff_along_ring(self, n_test: int = 256) -> float:
        """Minimum of T_eff_necklace around the ring (the bottleneck)."""
        phis = np.linspace(0.0, 2.0 * math.pi, n_test, endpoint=False)
        return float(min(
            self.t_eff_necklace(float(p)) for p in phis
        ))

    def max_t_eff_along_ring(self, n_test: int = 256) -> float:
        """Maximum of T_eff_necklace around the ring (at binary centres)."""
        phis = np.linspace(0.0, 2.0 * math.pi, n_test, endpoint=False)
        return float(max(
            self.t_eff_necklace(float(p)) for p in phis
        ))

    # ---- per-binary stability ----

    def per_binary_n_orbits(self) -> float:
        """Per-binary orbital lifetime: each binary on the necklace
        inherits the same Peters merger problem as the single-binary
        case. The necklace doesn't fix this."""
        b = EffectiveToroidalKerrBinary(M=self.M, d=self.d, chi=self.chi)
        t_merge = time_to_merger(b)
        Omega = math.sqrt(2.0 * self.M / self.d ** 3)
        T_orb = 2.0 * math.pi / Omega
        return float(t_merge / T_orb)


def packing_threshold_for_continuous_band(
    M: float, d: float, R_ring: float, chi: float = 1.0,
    n_max: int = 200,
) -> Optional[int]:
    """Smallest N such that T_eff_necklace stays >= 1 everywhere on the
    ring (continuous band). Returns None if no N <= n_max suffices.
    """
    for N in range(1, n_max + 1):
        nec = KnoppNecklace(
            n_binaries=N, M=M, d=d, chi=chi, R_ring=R_ring,
        )
        if nec.is_continuously_banded():
            return N
    return None


@dataclass(frozen=True)
class NecklaceReport:
    necklace: KnoppNecklace
    per_binary_band_edges: tuple[Optional[float], Optional[float]]
    per_binary_n_orbits: float
    adjacent_binary_chord: float
    t_eff_min_along_ring: float
    t_eff_max_along_ring: float
    has_continuous_band: bool
    packing_threshold: Optional[int]
    inherits_falsification: bool


def necklace_report(necklace: KnoppNecklace) -> NecklaceReport:
    """Combined diagnostic for a necklace configuration."""
    single_binary = EffectiveToroidalKerrBinary(
        M=necklace.M, d=necklace.d, chi=necklace.chi,
    )
    edges = single_binary.ctc_band_edges(include_phi=False)
    n_orb = necklace.per_binary_n_orbits()
    return NecklaceReport(
        necklace=necklace,
        per_binary_band_edges=edges,
        per_binary_n_orbits=float(n_orb),
        adjacent_binary_chord=float(necklace.adjacent_binary_chord()),
        t_eff_min_along_ring=float(necklace.min_t_eff_along_ring()),
        t_eff_max_along_ring=float(necklace.max_t_eff_along_ring()),
        has_continuous_band=bool(necklace.is_continuously_banded()),
        packing_threshold=packing_threshold_for_continuous_band(
            necklace.M, necklace.d, necklace.R_ring, necklace.chi,
        ),
        inherits_falsification=bool(n_orb < 1.0),
    )


def summarise_necklace(r: NecklaceReport) -> str:
    nec = r.necklace
    lines = [
        f"Knopp necklace: N = {nec.n_binaries}, M = {nec.M}, "
        f"d = {nec.d}, R_ring = {nec.R_ring}, chi = {nec.chi}",
        f"  per-binary band edges:    {r.per_binary_band_edges}",
        f"  per-binary n_orbits:      {r.per_binary_n_orbits:.4e}",
        f"  adjacent chord:           {r.adjacent_binary_chord:.4f}",
        f"  T_eff range on ring:      "
        f"[{r.t_eff_min_along_ring:.4f}, {r.t_eff_max_along_ring:.4f}]",
        f"  continuous band?          {r.has_continuous_band}",
        f"  packing N for continuity: {r.packing_threshold}",
        f"  inherits falsification?   {r.inherits_falsification}",
    ]
    return "\n".join(lines)
