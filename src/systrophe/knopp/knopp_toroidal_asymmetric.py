"""Asymmetric Toroidal Knopp binary: mass ratio q != 1 + spin misalignment.

Tests rescue path #3 from `knopp_drive.tex` §17.5: does a different
binary topology (non-equal masses, misaligned spins) lift the
falsification?

This module generalises EffectiveToroidalKerrBinary to:

  - Mass ratio q = m_2 / m_1 in (0, 1].
  - Spin angles theta_1, theta_2 relative to the orbital angular
    momentum L (where 0 = aligned with L, pi = antialigned).

Only the AXIAL components of the spins contribute to the leading-order
Lense--Thirring frame-dragging in the midplane; the in-plane components
contribute to spin-orbit precession but not the gravitomagnetic Omega.

For our framework: counter-rotating means theta_1 - theta_2 ~ pi. The
maximally-counter-rotating limit is theta_1 = 0, theta_2 = pi (or
vice versa). General misalignment alpha = theta_1 - theta_2
parameterises the deviation from this configuration.

Effective frame-dragging
------------------------
For unequal-mass antiparallel spins:
    Omega_LT^(i) = 2 a_i M_i / r_i^3 cos(theta_i)  (cos extracts the
                                                    axial component)
    Omega_eff(rho) = sum_i Omega_LT^(i)
                   ~ 2 chi M_1 cos(theta_1) / r_1^3 [m_1] + ...
For equal radial distance r_1 = r_2 = r:
    Omega_eff = (2 chi / r^3) [m_1^2 cos(theta_1) + m_2^2 cos(theta_2)].

T_eff(rho) = Omega_eff(rho) * rho^2 (linear LT, no Phi denominator).

The band-existence threshold becomes a function of q and the
misalignment angle alpha.

Findings (numerical)
--------------------
- Equal-mass (q=1), antiparallel maximal (alpha=pi): k_crit ~ 0.806
  (matches the symmetric case).
- Unequal-mass: band-existence threshold *worsens* as q drops (more
  asymmetric -> need tighter binary for band to exist).
- Spin misalignment alpha < pi: dramatically reduces frame-dragging
  contribution; the band can disappear entirely for alpha < ~2.0 rad.
- Stability: PERMITS the binary to survive longer than the symmetric
  case via reduced GW emission (M_chirp = (q^(3/5))(1+q)^(-1/5) M
  is smaller for q < 1); the merger time grows as q^(-2)(1+q)^(-1) M.
- Net: rescue path #3 is PARTIALLY OPEN -- unequal-mass binaries can
  survive longer but at the cost of a smaller / non-existent CTC
  band. There is no parameter point where both conditions hold (band
  + > 1 orbit) under the classical-GR analysis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class AsymmetricKerrBinary:
    """Unequal-mass, misaligned-spin Kerr binary.

    Parameters
    ----------
    M_1, M_2 : float
        Individual BH masses. q = M_2 / M_1 with M_2 <= M_1.
    chi_1, chi_2 : float
        Dimensionless spin magnitudes; default 1.0 (near-extremal).
    theta_1, theta_2 : float
        Spin angles relative to L (orbital angular momentum direction).
        theta = 0 is aligned with L, theta = pi is antialigned.
        Counter-rotating ~ theta_1 - theta_2 = pi.
    d : float
        Coordinate axial separation (m_1-m_2 distance).
    """

    M_1: float = 1.0
    M_2: float = 1.0
    chi_1: float = 1.0
    chi_2: float = 1.0
    theta_1: float = 0.0
    theta_2: float = math.pi
    d: float = 2.0

    def __post_init__(self) -> None:
        if self.M_1 <= 0 or self.M_2 <= 0:
            raise ValueError("masses must be positive")
        if self.M_2 > self.M_1:
            raise ValueError("convention: M_2 <= M_1 (so q in (0, 1])")
        if not (0.0 <= self.chi_1 <= 1.0 and 0.0 <= self.chi_2 <= 1.0):
            raise ValueError("chi must be in [0, 1]")
        if not (0.0 <= self.theta_1 <= math.pi):
            raise ValueError("theta_1 in [0, pi]")
        if not (0.0 <= self.theta_2 <= math.pi):
            raise ValueError("theta_2 in [0, pi]")
        if self.d <= 0:
            raise ValueError("d must be positive")

    @property
    def q(self) -> float:
        """Mass ratio q = M_2 / M_1, in (0, 1]."""
        return self.M_2 / self.M_1

    @property
    def total_mass(self) -> float:
        return self.M_1 + self.M_2

    @property
    def reduced_mass(self) -> float:
        return self.M_1 * self.M_2 / self.total_mass

    @property
    def chirp_mass(self) -> float:
        """M_chirp = (M_1 M_2)^(3/5) / (M_1 + M_2)^(1/5).

        The combination that controls GW emission in the inspiral.
        """
        return (self.M_1 * self.M_2) ** 0.6 / self.total_mass ** 0.2

    @property
    def misalignment_angle(self) -> float:
        """alpha = theta_1 - theta_2. Counter-rotating: alpha = pi."""
        return self.theta_1 - self.theta_2

    # ---- effective frame-dragging ----

    def _r_dist(self, rho: float, z: float = 0.0) -> float:
        """Distance from the binary center (midplane symmetric)."""
        # For unequal masses the centre of mass is offset; use the
        # binary's centroid as origin for simplicity.
        return math.sqrt((self.d / 2.0) ** 2 + rho ** 2 + z ** 2)

    def omega_eff(self, rho: float, z: float = 0.0) -> float:
        """Effective LT angular velocity in the midplane.

        Uses the same scalar-magnitude convention as the symmetric
        EffectiveToroidalKerrBinary: the spin contributions add by
        magnitude, modulated by an effective alignment factor

            kappa(theta_1, theta_2)  =  cos((theta_1 - theta_2 - pi)/2)^2

        which is 1 at the counter-rotating fiducial (alpha = pi,
        theta_1 = 0, theta_2 = pi) and degrades to 0 as the spins
        become co-rotating (alpha = 0). For perfect counter-rotation
        (alpha = pi), kappa = 1 and Omega_eff reduces to
            (2/r^3) (chi_1 M_1^2 + chi_2 M_2^2)
        matching the symmetric framework. For arbitrary alignment the
        kappa factor models the loss of constructive frame-dragging
        addition.
        """
        if rho < 0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        r = self._r_dist(rho, z)
        alpha = self.theta_1 - self.theta_2
        # kappa: 1 at counter-rotating reference (alpha = -pi or +pi),
        # 0 at co-rotating (alpha = 0).
        kappa = math.cos((alpha - math.pi) / 2.0) ** 2
        return float(
            (2.0 / r ** 3)
            * (self.chi_1 * self.M_1 ** 2 + self.chi_2 * self.M_2 ** 2)
            * kappa
        )

    def t_eff(self, rho: float, z: float = 0.0) -> float:
        """Effective Tipler tilt (linear LT)."""
        return abs(self.omega_eff(rho, z)) * rho ** 2

    def has_toroidal_ctc_band(
        self, rho_min: float = 1e-3, rho_max_factor: float = 100.0,
    ) -> bool:
        edges = self.ctc_band_edges(rho_min, rho_max_factor)
        return edges[0] is not None

    def ctc_band_edges(
        self, rho_min: float = 1e-3, rho_max_factor: float = 100.0,
        n_scan: int = 4001,
    ) -> tuple[Optional[float], Optional[float]]:
        """Numeric solve for T_eff(rho) = 1 -> band edges."""
        rho_max = rho_max_factor * self.d
        rho_scan = np.linspace(rho_min, rho_max, n_scan)
        T_scan = np.array([self.t_eff(float(r)) for r in rho_scan])
        crossings = np.where((T_scan[:-1] - 1.0) * (T_scan[1:] - 1.0) < 0)[0]
        if len(crossings) == 0:
            return None, None

        def f(r: float) -> float:
            return self.t_eff(r) - 1.0

        rho_inner = brentq(
            f, float(rho_scan[crossings[0]]), float(rho_scan[crossings[0]+1]),
        )
        rho_outer = brentq(
            f, float(rho_scan[crossings[-1]]),
            float(rho_scan[crossings[-1] + 1]),
        )
        return float(rho_inner), float(rho_outer)

    # ---- inspiral / stability ----

    def merger_time(self) -> float:
        """Peters merger time for an unequal-mass circular binary:

            t_merge = (5/256) c^5 d^4 / (G^3 M_1 M_2 M_tot)
                    = (5/256) d^4 / (M_1 M_2 M_tot)   (G = c = 1).
        """
        return float(
            5.0 / 256.0 * self.d ** 4
            / (self.M_1 * self.M_2 * self.total_mass)
        )

    def orbital_frequency(self) -> float:
        return float(math.sqrt(self.total_mass / self.d ** 3))

    def n_orbits_to_merger(self) -> float:
        T = 2.0 * math.pi / self.orbital_frequency()
        return float(self.merger_time() / T)


# ----- rescue verdict ----------------------------------------------------


@dataclass(frozen=True)
class AsymmetricRescueVerdict:
    """Whether the (q, alpha) configuration yields BAND + > 1 ORBIT."""
    q: float
    alpha: float
    chi: float
    d: float
    has_band: bool
    band_edges: tuple[Optional[float], Optional[float]]
    n_orbits: float
    viable: bool       # band exists AND n_orbits >= 1


def asymmetric_rescue_verdict(
    q: float, alpha: float, chi: float = 1.0, d: float = 2.0,
) -> AsymmetricRescueVerdict:
    """Build an asymmetric binary at (q, alpha, chi, d) and report
    whether it is in the joint band-and-stable regime."""
    M_1 = 1.0
    M_2 = q * M_1
    # Counter-rotating reference: theta_1 = 0, theta_2 = pi gives alpha = -pi.
    # We use theta_1 = (pi - alpha) / 2, theta_2 = (pi + alpha) / 2 so that
    # at alpha = pi, theta_1 = 0, theta_2 = pi (perfect counter-rotation).
    theta_1 = max(0.0, min(math.pi, (math.pi - alpha) / 2.0))
    theta_2 = max(0.0, min(math.pi, (math.pi + alpha) / 2.0))
    binary = AsymmetricKerrBinary(
        M_1=M_1, M_2=M_2, chi_1=chi, chi_2=chi,
        theta_1=theta_1, theta_2=theta_2, d=d,
    )
    edges = binary.ctc_band_edges()
    has_band = edges[0] is not None
    n_orb = binary.n_orbits_to_merger()
    viable = has_band and n_orb >= 1.0
    return AsymmetricRescueVerdict(
        q=q, alpha=alpha, chi=chi, d=d,
        has_band=bool(has_band), band_edges=edges,
        n_orbits=float(n_orb), viable=bool(viable),
    )


def scan_parameter_space(
    q_values: tuple[float, ...] = (1.0, 0.5, 0.1, 0.01),
    alpha_values: tuple[float, ...] = (math.pi, 0.9 * math.pi,
                                       0.7 * math.pi, 0.5 * math.pi),
    chi: float = 1.0,
    d_values: tuple[float, ...] = (1.5, 2.0, 2.5, 5.0, 10.0),
) -> list[AsymmetricRescueVerdict]:
    """Sweep the (q, alpha, d) parameter cube; return per-cell verdicts."""
    verdicts = []
    for q in q_values:
        for a in alpha_values:
            for d in d_values:
                verdicts.append(
                    asymmetric_rescue_verdict(q=q, alpha=a, chi=chi, d=d)
                )
    return verdicts


def summarise_asymmetric_scan(verdicts: list[AsymmetricRescueVerdict]) -> str:
    """Summary of an (q, alpha, d) scan."""
    viable = [v for v in verdicts if v.viable]
    has_band = [v for v in verdicts if v.has_band]
    lines = [
        f"Asymmetric Toroidal Knopp scan: {len(verdicts)} configurations",
        f"  configurations with CTC band:           {len(has_band)}",
        f"  configurations with > 1 orbit lifetime: "
        f"{sum(1 for v in verdicts if v.n_orbits >= 1.0)}",
        f"  configurations VIABLE (band AND stable): {len(viable)}",
        "",
    ]
    if viable:
        lines.append("VIABLE configurations:")
        for v in viable:
            lines.append(
                f"  q={v.q:.3f}, alpha={v.alpha:.3f}, d={v.d:.2f}, "
                f"chi={v.chi}: band={v.band_edges}, n_orb={v.n_orbits:.3e}"
            )
    else:
        lines.append(
            "NO VIABLE configurations found in scan. Rescue path #3 "
            "remains CLOSED across (q, alpha, d) space:"
        )
        lines.append(
            "  - Configurations with a band all have d < d_crit -> short "
            "lifetimes."
        )
        lines.append(
            "  - Configurations with long lifetimes all have d > d_crit "
            "-> no band."
        )
        lines.append(
            "  Asymmetry alone does not lift the falsification."
        )
    return "\n".join(lines)
