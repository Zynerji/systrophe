"""Analog acoustic-horizon analyser on a van Stockum exterior.

Wraps systrophe.analogs.acoustic_metric for ergonomic access:

  - find the acoustic horizon (locus c² = v² ⇔ F = 0)
  - compute its surface gravity, acoustic Hawking T
  - audit CTC ↔ supersonic equivalence at sampled radii
  - report acoustic vs gravitational Hawking temperatures (the two
    coincide by the Unruh identification, used here as a self-check)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.analogs.acoustic_metric import (
    acoustic_horizon_radius,
    acoustic_hawking_temperature,
    acoustic_line_element_at_radius,
    acoustic_surface_gravity,
    compare_acoustic_vs_gravitational_T_H,
    ctc_region_is_supersonic,
)
from systrophe.geometry.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class AnalogHorizonReport:
    """Compact report on a van Stockum analog acoustic horizon."""
    omega: float
    R: float
    horizon_r: float | None
    surface_gravity: float | None
    T_hawking_acoustic: float | None
    T_hawking_gravitational: float | None
    ctc_supersonic_consistent: bool
    n_subsonic: int
    n_supersonic: int
    n_sonic: int

    def __repr__(self) -> str:
        return (
            f"AnalogHorizonReport(omega={self.omega}, R={self.R}, "
            f"r_H={self.horizon_r}, T_H={self.T_hawking_acoustic}, "
            f"consistent={self.ctc_supersonic_consistent})"
        )


class AnalogHorizonAnalyser:
    """One-object-per-spacetime analog-acoustic-horizon analyser.

    Parameters
    ----------
    omega, R : float
        Underlying van Stockum cylinder parameters.
    r_search_min, r_search_max : float
        Range to scan for the acoustic horizon (default 1.05R .. 20R).
    """

    def __init__(self, omega: float, R: float,
                 r_search_min: float | None = None,
                 r_search_max: float | None = None) -> None:
        self.vs = VanStockumInterior(omega=float(omega), R=float(R))
        self.r_search_min = (1.05 * R) if r_search_min is None else float(r_search_min)
        self.r_search_max = (20.0 * R) if r_search_max is None else float(r_search_max)

    @property
    def omega(self) -> float:
        return float(self.vs.omega)

    @property
    def R(self) -> float:
        return float(self.vs.R)

    def horizon(self) -> float | None:
        """Return the first acoustic horizon (F=0) in the search range, or None."""
        return acoustic_horizon_radius(
            self.vs, r_min=self.r_search_min, r_max=self.r_search_max,
        )

    def surface_gravity(self, r_horizon: float | None = None) -> float:
        r_h = r_horizon if r_horizon is not None else self.horizon()
        if r_h is None:
            raise ValueError("no acoustic horizon found in search range")
        return float(acoustic_surface_gravity(self.vs, r_horizon=r_h))

    def T_hawking_acoustic(self, r_horizon: float | None = None) -> float:
        r_h = r_horizon if r_horizon is not None else self.horizon()
        if r_h is None:
            raise ValueError("no acoustic horizon found in search range")
        return float(acoustic_hawking_temperature(self.vs, r_horizon=r_h))

    def compare_acoustic_vs_gravitational(self, r_horizon: float | None = None
                                           ) -> dict:
        r_h = r_horizon if r_horizon is not None else self.horizon()
        if r_h is None:
            raise ValueError("no acoustic horizon found in search range")
        return compare_acoustic_vs_gravitational_T_H(self.vs, r_horizon=r_h)

    def line_element_at(self, r: float) -> dict:
        """Acoustic line-element components at a single radius."""
        return acoustic_line_element_at_radius(self.vs, float(r))

    def audit_ctc_supersonic(self, n_samples: int = 100) -> dict:
        """Verify F < 0 ⇔ supersonic across n_samples radii in the search range."""
        rs = np.linspace(self.r_search_min, self.r_search_max, n_samples)
        return ctc_region_is_supersonic(self.vs, rs)

    def report(self, n_audit_samples: int = 100) -> AnalogHorizonReport:
        """One-shot characterisation."""
        r_h = self.horizon()
        if r_h is None:
            audit = self.audit_ctc_supersonic(n_samples=n_audit_samples)
            return AnalogHorizonReport(
                omega=self.omega, R=self.R, horizon_r=None,
                surface_gravity=None,
                T_hawking_acoustic=None, T_hawking_gravitational=None,
                ctc_supersonic_consistent=bool(audit["ctc_supersonic_consistency"]),
                n_subsonic=int(audit["n_subsonic"]),
                n_supersonic=int(audit["n_supersonic"]),
                n_sonic=int(audit["n_sonic"]),
            )
        kappa = self.surface_gravity(r_horizon=r_h)
        T_ac = self.T_hawking_acoustic(r_horizon=r_h)
        cmp_ = self.compare_acoustic_vs_gravitational(r_horizon=r_h)
        audit = self.audit_ctc_supersonic(n_samples=n_audit_samples)
        return AnalogHorizonReport(
            omega=self.omega, R=self.R, horizon_r=float(r_h),
            surface_gravity=float(kappa),
            T_hawking_acoustic=float(T_ac),
            T_hawking_gravitational=float(cmp_["T_gravitational"]),
            ctc_supersonic_consistent=bool(audit["ctc_supersonic_consistency"]),
            n_subsonic=int(audit["n_subsonic"]),
            n_supersonic=int(audit["n_supersonic"]),
            n_sonic=int(audit["n_sonic"]),
        )
