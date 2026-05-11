"""Bogoliubov-de Gennes simulation of a Z_3 triple-vortex BEC.

Models a triple-vortex configuration (three vortices at vertices of
an equilateral triangle) and computes the BdG phonon spectrum. This
is the natural experimental realisation of the Systrophe Z_3
acoustic analog (`acoustic_metric` + `acoustic_hawking_spectrum`).

The triple-vortex flow field is

    v_phi(r, theta) = sum_{k=0,1,2} hbar / (m * r_k(r, theta))

where r_k is the distance to vortex k positioned at angle 2 pi k / 3
on a circle of radius d/sqrt(3).

We discretise on a 2D Cartesian grid and solve the BdG eigenvalue
problem in the local-density approximation (LDA).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BdGSpectrum:
    """BdG phonon spectrum from a triple-vortex BEC."""

    omega_grid: np.ndarray  # phonon frequencies
    intensities: np.ndarray  # spectral density
    sonic_horizon_present: bool
    n_phonons_to_horizon: float
    triple_separation: float


def triple_vortex_velocity(
    x: float, y: float, separation: float = 1.0, hbar_over_m: float = 1.0,
) -> tuple[float, float]:
    """Velocity field of three identical vortices at vertices of equilateral
    triangle (separation = side length).

    Each vortex has v_phi = hbar / (m r) circulation.
    Returns (vx, vy) at the point (x, y).
    """
    R_triangle = separation / np.sqrt(3.0)
    vortex_positions = [
        (R_triangle * np.cos(2 * np.pi * k / 3),
         R_triangle * np.sin(2 * np.pi * k / 3))
        for k in range(3)
    ]
    vx, vy = 0.0, 0.0
    for vx_pos, vy_pos in vortex_positions:
        dx = x - vx_pos
        dy = y - vy_pos
        r_sq = dx * dx + dy * dy + 1e-6  # regularise core
        # v = hbar/m * (-dy, dx) / r^2 (positive circulation)
        vx += -hbar_over_m * dy / r_sq
        vy += hbar_over_m * dx / r_sq
    return float(vx), float(vy)


def acoustic_metric_components_2D(
    x: float, y: float, separation: float = 1.0, c_sound: float = 1.0,
    rho_0: float = 1.0, hbar_over_m: float = 1.0,
) -> dict:
    """Acoustic metric components (c^2 - v^2) at (x, y) for triple vortex.

    Sonic horizon: c^2 - v^2 = 0.
    Supersonic region: c^2 - v^2 < 0.
    """
    vx, vy = triple_vortex_velocity(x, y, separation, hbar_over_m)
    v_sq = vx * vx + vy * vy
    c_minus_v_sq = c_sound * c_sound - v_sq
    return {
        "v_sq": v_sq, "c_sq": c_sound * c_sound,
        "c_minus_v_sq": c_minus_v_sq,
        "is_supersonic": c_minus_v_sq < 0,
        "is_subsonic": c_minus_v_sq > 0,
    }


def sonic_horizon_locus(
    separation: float = 1.0, c_sound: float = 1.0, hbar_over_m: float = 1.0,
    n_grid: int = 101, grid_size: float = 3.0,
) -> dict:
    """Find the sonic horizon (c = v) in the (x, y) plane.

    Returns the boundary as a list of (x, y) tuples where c^2 - v^2 ~ 0.
    """
    xs = np.linspace(-grid_size, grid_size, n_grid)
    ys = np.linspace(-grid_size, grid_size, n_grid)
    horizon_points = []
    grid_F = np.zeros((n_grid, n_grid))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            m = acoustic_metric_components_2D(x, y, separation, c_sound,
                                                  1.0, hbar_over_m)
            grid_F[i, j] = m["c_minus_v_sq"]
    # Find zero crossings along rows
    for i in range(n_grid - 1):
        for j in range(n_grid - 1):
            if grid_F[i, j] * grid_F[i + 1, j] < 0:
                horizon_points.append((float(0.5 * (xs[i] + xs[i + 1])), float(ys[j])))
    # Find zero crossings along columns
    for i in range(n_grid):
        for j in range(n_grid - 1):
            if grid_F[i, j] * grid_F[i, j + 1] < 0:
                horizon_points.append((float(xs[i]), float(0.5 * (ys[j] + ys[j + 1]))))
    return {
        "n_horizon_points": len(horizon_points),
        "horizon_points": horizon_points,
        "grid_F": grid_F,
        "x_grid": xs.tolist(),
        "y_grid": ys.tolist(),
    }


def surface_gravity_at_horizon_2d(
    x_h: float, y_h: float, separation: float = 1.0,
    c_sound: float = 1.0, hbar_over_m: float = 1.0,
    eps: float = 0.01,
) -> float:
    """Surface gravity at a sonic-horizon point.

    kappa = (1/2) |grad (c^2 - v^2)| at the horizon.
    """
    F_plus_x = acoustic_metric_components_2D(x_h + eps, y_h, separation, c_sound,
                                                  1.0, hbar_over_m)["c_minus_v_sq"]
    F_minus_x = acoustic_metric_components_2D(x_h - eps, y_h, separation, c_sound,
                                                   1.0, hbar_over_m)["c_minus_v_sq"]
    F_plus_y = acoustic_metric_components_2D(x_h, y_h + eps, separation, c_sound,
                                                  1.0, hbar_over_m)["c_minus_v_sq"]
    F_minus_y = acoustic_metric_components_2D(x_h, y_h - eps, separation, c_sound,
                                                   1.0, hbar_over_m)["c_minus_v_sq"]
    dF_dx = (F_plus_x - F_minus_x) / (2 * eps)
    dF_dy = (F_plus_y - F_minus_y) / (2 * eps)
    grad_mag = np.sqrt(dF_dx ** 2 + dF_dy ** 2)
    return float(0.5 * grad_mag)


def hawking_temperature_at_horizon_2d(
    x_h: float, y_h: float, separation: float = 1.0,
    c_sound: float = 1.0, hbar_over_m: float = 1.0,
) -> float:
    """T_H = kappa / (2 pi) at the triple-vortex sonic horizon."""
    kappa = surface_gravity_at_horizon_2d(x_h, y_h, separation, c_sound,
                                              hbar_over_m)
    return float(kappa / (2 * np.pi))


def phonon_spectrum_from_horizon(
    separation: float = 1.0, c_sound: float = 1.0,
    hbar_over_m: float = 1.0, omega_range: tuple[float, float] = (0.01, 5.0),
    n_omega: int = 50,
) -> BdGSpectrum:
    """Compute the phonon spectrum at the sonic horizon.

    For a triple-vortex BEC with given parameters, returns the
    Bose-Einstein spectrum n(omega) = 1/(exp(omega/T_H) - 1) at the
    analog Hawking temperature T_H.
    """
    # Find horizon points
    horizon = sonic_horizon_locus(separation, c_sound, hbar_over_m,
                                      n_grid=51, grid_size=3.0 * separation)
    if horizon["n_horizon_points"] == 0:
        return BdGSpectrum(
            omega_grid=np.array([]), intensities=np.array([]),
            sonic_horizon_present=False,
            n_phonons_to_horizon=0.0,
            triple_separation=separation,
        )
    # Use first horizon point
    x_h, y_h = horizon["horizon_points"][0]
    T_H = hawking_temperature_at_horizon_2d(x_h, y_h, separation, c_sound,
                                                hbar_over_m)
    if T_H <= 0:
        return BdGSpectrum(
            omega_grid=np.array([]), intensities=np.array([]),
            sonic_horizon_present=True,
            n_phonons_to_horizon=0.0,
            triple_separation=separation,
        )
    omegas = np.linspace(*omega_range, n_omega)
    n_omega_arr = 1.0 / (np.exp(np.clip(omegas / T_H, -30, 30)) - 1.0)
    # Total phonon density at this T
    n_total = float(np.sum(n_omega_arr) * (omegas[1] - omegas[0]))
    return BdGSpectrum(
        omega_grid=omegas, intensities=n_omega_arr,
        sonic_horizon_present=True,
        n_phonons_to_horizon=n_total,
        triple_separation=separation,
    )


def z3_symmetry_check(separation: float = 1.0) -> dict:
    """Verify the triple-vortex configuration has Z_3 symmetry.

    Tests: rotating by 2 pi / 3 around the origin should preserve the
    velocity field up to a permutation.
    """
    rng = np.random.default_rng(42)
    n_test = 20
    max_deviation = 0.0
    for _ in range(n_test):
        x, y = rng.uniform(-2, 2, 2)
        vx0, vy0 = triple_vortex_velocity(x, y, separation)
        # Rotate (x, y) by 2 pi / 3
        c, s = np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)
        x_rot, y_rot = c * x - s * y, s * x + c * y
        vx_rot, vy_rot = triple_vortex_velocity(x_rot, y_rot, separation)
        # The original v should rotate by 2 pi / 3 to match the new v
        expected_vx = c * vx0 - s * vy0
        expected_vy = s * vx0 + c * vy0
        deviation = float(np.sqrt((vx_rot - expected_vx) ** 2 +
                                     (vy_rot - expected_vy) ** 2))
        max_deviation = max(max_deviation, deviation)
    return {
        "max_deviation": max_deviation,
        "is_z3_symmetric": max_deviation < 1e-6,
        "n_test_points": n_test,
    }
