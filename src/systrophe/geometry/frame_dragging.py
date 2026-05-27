"""Frame-dragging (Lense-Thirring) signatures on LP background.

A rotating cylinder generates spacetime "frame dragging": local
inertial frames are dragged around with the cylinder. For a test
particle (or gyroscope) on a free-fall trajectory, this manifests
as a precession of the particle's spin axis.

The Lense-Thirring precession frequency for a stationary gyroscope
at radius r in a rotating-cylinder spacetime is given (to leading
order) by

    Omega_LT = -d/dr [K(r) / r^2] / 2

evaluated at the gyroscope's location. For the LP exterior, K(r)
oscillates log-periodically, so Omega_LT oscillates correspondingly.

This module:

- `lense_thirring_frequency(vs, r)`: precession frequency at r
- `gyroscope_precession_angle(vs, r, time)`: cumulative angle
- `frame_dragging_pattern(vs, r_min, r_max)`: spatial profile
- `compare_to_kerr_LT(vs, r, a_Kerr, M_Kerr)`: side-by-side comparison
- `inertial_drag_zero_locus(vs)`: where Omega_LT = 0 (frame-isolation surfaces)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameDragging:
    """Frame-dragging descriptor at a single radius."""

    r: float
    Omega_LT: float  # Lense-Thirring frequency
    K: float
    K_over_r2: float
    is_dragged: bool  # True iff |Omega_LT| > tol


def lense_thirring_frequency(vs, r: float, eps: float = 1e-5) -> float:
    """Lense-Thirring precession frequency at r.

    Omega_LT = -d/dr [K(r) / r^2] / 2.
    """
    K_plus = float(vs.analytic_exterior_K(r + eps))
    K_minus = float(vs.analytic_exterior_K(r - eps))
    K_at = float(vs.analytic_exterior_K(r))
    # Compute K/r^2 and its derivative
    f_plus = K_plus / ((r + eps) ** 2)
    f_minus = K_minus / ((r - eps) ** 2)
    df_dr = (f_plus - f_minus) / (2 * eps)
    return float(-df_dr / 2)


def gyroscope_precession_angle(vs, r: float, time: float = 1.0) -> float:
    """Cumulative gyroscope precession angle over given time."""
    omega_lt = lense_thirring_frequency(vs, r)
    return float(omega_lt * time)


def frame_dragging_pattern(
    vs, r_min: float = 1.05, r_max: float = 20.0, n_grid: int = 200,
) -> dict:
    """Spatial profile of Lense-Thirring precession in the exterior.

    Returns dict with:
      - r_grid
      - Omega_LT array
      - max_|Omega_LT|
      - zero crossings (frame-isolation surfaces)
    """
    rs = np.linspace(r_min, r_max, n_grid)
    omegas = np.array([lense_thirring_frequency(vs, float(r)) for r in rs])
    max_abs = float(np.max(np.abs(omegas)))
    # Zero crossings
    signs = np.sign(omegas)
    flips = np.where(np.diff(signs) != 0)[0]
    zero_radii = []
    for i in flips:
        r1, r2 = rs[i], rs[i + 1]
        o1, o2 = omegas[i], omegas[i + 1]
        if abs(o2 - o1) > 1e-30:
            zero_radii.append(float(r1 - o1 * (r2 - r1) / (o2 - o1)))
    return {
        "r_grid": rs.tolist(),
        "Omega_LT_array": omegas.tolist(),
        "max_abs_Omega_LT": max_abs,
        "n_zero_crossings": len(zero_radii),
        "zero_crossing_radii": zero_radii,
    }


def kerr_lense_thirring(a: float, M: float, r: float) -> float:
    """Lense-Thirring frequency for a Kerr black hole.

    Omega_LT_Kerr = 2 a M / (r^3 + a^2 r + 2 M a^2).
    """
    if r <= 0:
        return 0.0
    return float(2 * a * M / (r ** 3 + a ** 2 * r + 2 * M * a ** 2))


def compare_to_kerr_LT(
    vs, r: float, a_Kerr: float = 0.5, M_Kerr: float = 1.0,
) -> dict:
    """Compare LP frame-dragging at r to a Kerr value with same total angular momentum."""
    omega_lp = lense_thirring_frequency(vs, r)
    omega_kerr = kerr_lense_thirring(a_Kerr, M_Kerr, r)
    return {
        "r": r, "Omega_LT_LP": omega_lp, "Omega_LT_Kerr": omega_kerr,
        "ratio_LP_over_Kerr": omega_lp / omega_kerr if abs(omega_kerr) > 1e-30 else float("nan"),
    }


def inertial_drag_zero_locus(vs, r_min: float = 1.05, r_max: float = 20.0,
                                  n_grid: int = 500) -> list[float]:
    """Find radii where Omega_LT = 0 (frame-isolation surfaces)."""
    pattern = frame_dragging_pattern(vs, r_min, r_max, n_grid)
    return pattern["zero_crossing_radii"]


def total_drag_per_revolution(
    vs, r: float, Omega_obs: float = 1.0,
) -> float:
    """Total precession over one Keplerian revolution at radius r.

    A test particle in circular orbit at angular velocity Omega_obs
    completes a revolution in time T = 2 pi / Omega_obs. The
    Lense-Thirring precession accumulated is Omega_LT * T.

    Returns the dimensionless precession angle per revolution.
    """
    omega_lt = lense_thirring_frequency(vs, r)
    T = 2 * np.pi / max(abs(Omega_obs), 1e-30)
    return float(omega_lt * T)
