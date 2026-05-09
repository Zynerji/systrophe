"""Time-machine harness: identify CTC bands and tune time-travel orbits.

This module is the "engineering" layer of Systrophe. Given a single
cylinder or a SystrophePair, it answers:

1. **Where are the CTC bands?** Radial intervals where g_{phi phi}(r) < 0.
   These are the "time-travel windows": only inside such bands can a
   closed phi-orbit be timelike (Tipler 1974; Bonnor 1980).

2. **What orbital angular velocity Omega tunes the coordinate-time
   advance per revolution to a target value Delta t?**
   At fixed r in a CTC band, an orbit with angular velocity Omega has
       Delta t = 2 pi / Omega                         (coordinate time)
       Delta tau = sqrt(F - 2 K Omega - L Omega^2) * |Delta t|.
   For Omega < 0 the coordinate-time advance is negative — the particle
   has moved into its own past while advancing its proper time.

3. **Tunable windows via the off-set.** Varying the SystrophePair phase
   offset shifts CTC bands along the r-axis. The user can therefore
   "engineer" which radius supports the desired time-travel orbit.

This is a *theoretical* framework. The cylinders are idealised
infinite-length van Stockum dust sources; the result is a vacuum
Einstein solution that admits CTCs. Whether the configuration could be
physically realised is a separate (open) question, addressed by the
chronology-protection literature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .geodesic import (
    CircularOrbit,
    is_omega_timelike,
    omega_for_target_coord_time,
    timelike_omega_bounds,
)


@dataclass(frozen=True)
class TimeMachineWindow:
    """A CTC band: a radial interval where g_{phi phi} < 0 and timelike
    closed phi-orbits exist.

    Attributes
    ----------
    r_inner, r_outer : float
        Radial boundaries of the band (where L = 0).
    r_min_L : float
        Radius of minimum L (deepest CTC point in the band).
    L_min : float
        Value of L at r_min_L (most negative).
    F_at_min, K_at_min : float
        Metric components at r_min_L (used for orbit tuning).
    omega_bounds_at_min : tuple[float, float]
        (Omega_-, Omega_+) — boundaries of the timelike-orbit Omega
        sector at r_min_L. Inside the band, timelike orbits require
        Omega < Omega_- or Omega > Omega_+.
    """

    r_inner: float
    r_outer: float
    r_min_L: float
    L_min: float
    F_at_min: float
    K_at_min: float
    omega_bounds_at_min: tuple[float, float]

    def log_span(self) -> float:
        return float(np.log(self.r_outer / self.r_inner))

    def orbit_at_r_min(self, Omega: float) -> CircularOrbit:
        """Construct a circular orbit at r_min_L with given Omega."""
        return CircularOrbit(
            r=self.r_min_L,
            F=self.F_at_min,
            K=self.K_at_min,
            L=self.L_min,
            Omega=Omega,
        )

    def orbit_for_target_dt(self, target_dt: float) -> CircularOrbit:
        """Return the orbit at r_min_L whose Delta t per revolution = target_dt.

        Negative target_dt selects a backward-time-travel orbit (provided
        the resulting Omega lies in the timelike sector). Raises
        ValueError if the target is unattainable.
        """
        if target_dt == 0.0:
            raise ValueError("target_dt = 0 corresponds to Omega = +infinity")
        Omega = 2.0 * np.pi / target_dt
        if not is_omega_timelike(self.F_at_min, self.K_at_min, self.L_min, self.r_min_L, Omega):
            ob_lo, ob_hi = self.omega_bounds_at_min
            raise ValueError(
                f"Omega = {Omega:.4f} (target_dt = {target_dt:.4f}) is in the "
                f"spacelike sector [{ob_lo:.4f}, {ob_hi:.4f}]; choose "
                f"Omega outside this interval (CTC region admits Omega in "
                f"(-inf, {ob_lo:.4f}) U ({ob_hi:.4f}, +inf))."
            )
        return self.orbit_at_r_min(Omega)


def find_time_machine_windows(
    L_fn: Callable[[float | np.ndarray], np.ndarray],
    F_fn: Callable[[float | np.ndarray], np.ndarray] | None,
    K_fn: Callable[[float | np.ndarray], np.ndarray] | None,
    r_min: float,
    r_max: float,
    n_grid: int = 4001,
) -> list[TimeMachineWindow]:
    """Identify all CTC bands of L_fn in [r_min, r_max] and characterize each.

    For each CTC band [r_a, r_b] (where L < 0 between two consecutive
    L = 0 zeros), compute:
      - the radius of minimum L within the band,
      - the metric components F, K at that radius,
      - the timelike-orbit Omega bounds.
    """
    from .ctc import find_ctc_intervals

    bands = find_ctc_intervals(L_fn, r_min=r_min, r_max=r_max, n_grid=n_grid)
    windows: list[TimeMachineWindow] = []
    for r_a, r_b in bands:
        r_grid = np.linspace(r_a, r_b, 1001)
        L_vals = np.asarray(L_fn(r_grid), dtype=float)
        i = int(np.argmin(L_vals))
        r_min_L = float(r_grid[i])
        L_min = float(L_vals[i])
        F_at = float(F_fn(r_min_L)) if F_fn is not None else 1.0
        K_at = float(K_fn(r_min_L)) if K_fn is not None else 0.0
        ob = timelike_omega_bounds(F_at, K_at, L_min, r_min_L)
        windows.append(
            TimeMachineWindow(
                r_inner=float(r_a),
                r_outer=float(r_b),
                r_min_L=r_min_L,
                L_min=L_min,
                F_at_min=F_at,
                K_at_min=K_at,
                omega_bounds_at_min=ob,
            )
        )
    return windows


def find_single_cylinder_windows(vs, r_min: float, r_max: float, n_grid: int = 4001):
    """Find CTC windows of a single VanStockumInterior using analytic L, F, K."""
    if not vs.is_supercritical():
        raise ValueError("Single-cylinder time machine requires a > 1/2 (Case III).")
    return find_time_machine_windows(
        L_fn=lambda r: vs.analytic_exterior_L(r),
        F_fn=lambda r: vs.analytic_exterior_F(r),
        K_fn=lambda r: vs.analytic_exterior_K(r),
        r_min=r_min,
        r_max=r_max,
        n_grid=n_grid,
    )


def harness_time_loop(
    window: TimeMachineWindow, target_dt_per_rev: float, n_revolutions: int
) -> dict:
    """Compute the closed-loop schedule of an N-revolution time-travel orbit.

    Parameters
    ----------
    window : TimeMachineWindow
        The CTC band to use.
    target_dt_per_rev : float
        Desired coordinate-time advance per revolution (negative -> backwards).
    n_revolutions : int
        Number of revolutions.

    Returns
    -------
    dict with the orbit data and accumulated quantities.
    """
    orbit = window.orbit_for_target_dt(target_dt_per_rev)
    if not orbit.is_timelike:
        raise ValueError("Constructed orbit is not timelike; check inputs.")
    dt_per = orbit.coord_dt_per_revolution
    dtau_per = orbit.proper_dtau_per_revolution
    return {
        "r": orbit.r,
        "Omega": orbit.Omega,
        "F": orbit.F,
        "K": orbit.K,
        "L": orbit.L,
        "is_timelike": orbit.is_timelike,
        "is_in_ctc_region": orbit.is_in_ctc_region,
        "n_revolutions": int(n_revolutions),
        "dt_per_revolution": dt_per,
        "dtau_per_revolution": dtau_per,
        "total_coord_time_advance": n_revolutions * dt_per,
        "total_proper_time_advance": n_revolutions * dtau_per,
        "net_coord_minus_proper": n_revolutions * (dt_per - dtau_per),
    }
