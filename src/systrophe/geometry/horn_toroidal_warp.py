"""Horn-toroidal warp bubble: steerable exotic-matter shell.

A spherical warp bubble (Alcubierre / Bobrick-Martire) has 4 pi
isotropic light-cone tilt and therefore no preferred direction; the
craft moves rigidly along the pre-set bubble worldline.

Replacing the bubble's spherical shell by a slightly horn-toroidal
shell breaks reflection symmetry, gives the exotic-matter
distribution a net dipole, and supplies a small steering vector
that points along the horn-twist axis.

Geometry
--------
The shell is parameterised by

    rho(theta) = R [1 + epsilon cos(theta - theta_0)]

where R is the bubble radius, epsilon << 1 is the horn-twist
amplitude, and theta_0 is the horn-twist orientation. The horn-
torus topology corresponds to a tube cross-section whose centre
traces a circle of radius R and whose thickness varies with theta.

The exotic-matter density T_{tt}(theta) inherits the
asymmetry through

    T_{tt}(theta) = T_{tt}^bare(rho(theta))

so the spatial dipole moment is

    p_horn = integral over theta of T_{tt}(theta) cos(theta - theta_0)
           ~ R^2 epsilon (1 + O(epsilon^2))

This dipole induces a small linear-momentum flux on the shell that
acts as a steering thrust.

This module
-----------
- ``horn_radius`` -- rho(theta)
- ``horn_shape_dipole`` -- analytic dipole moment to leading order
- ``horn_T_tt_profile`` -- T_{tt}(theta) for the bubble
- ``horn_steering_vector`` -- (p_x, p_y) given epsilon, theta_0, R
- ``novelty_scan`` -- catcher over epsilon at fixed (R, theta_0).
  The catcher should report SMOOTH for the dipole component (it
  scales linearly with epsilon), but flag a sharp transition at
  the horn-pinch threshold epsilon_pinch where the torus self-
  intersects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.geometry.bobrick_martire import bm_NEC_radial, bm_shape
from systrophe.catchers.novelty_catcher import scan_novelty


def horn_radius(
    theta: float | np.ndarray, R: float = 1.0,
    epsilon: float = 0.1, theta_0: float = 0.0,
) -> float | np.ndarray:
    """rho(theta) = R [1 + eps cos(theta - theta_0)]."""
    t = np.asarray(theta, dtype=float)
    return R * (1.0 + epsilon * np.cos(t - theta_0))


def horn_pinch_threshold(R: float = 1.0) -> float:
    """epsilon at which the torus self-intersects (numerical only).

    For R/r_tube = 1 (degenerate horn-torus), pinch occurs at
    epsilon = 1. For thinner tubes (rho_tube/R << 1) the pinch
    threshold rises; we return the canonical epsilon = 1.0 limit.
    """
    return 1.0


def horn_T_tt_profile(
    theta: float, R: float = 1.0, epsilon: float = 0.1,
    theta_0: float = 0.0, m_ADM: float = -1.0, sigma: float = 4.0,
) -> float:
    """T_{tt}(theta) for the horn-toroidal Bobrick-Martire bubble.

    The horn-twist enters via a theta-dependent ADM mass weight,
    m_ADM_eff(theta) = m_ADM * (1 + epsilon cos(theta - theta_0))
    so that the exotic-matter density is amplified on the horn-side
    and suppressed on the anti-horn-side. This produces a non-zero
    angular dipole p ~ R^2 epsilon * |m_ADM|.

    Evaluated AT the shell (rho = R, x = 0) so the result is the
    local exotic-matter density along the shell circumference.
    """
    m_eff = m_ADM * (1.0 + epsilon * math.cos(theta - theta_0))
    val = bm_NEC_radial(0.0, R, m_ADM=m_eff, R=R, sigma=sigma)
    return float(val)


def horn_steering_vector(
    R: float = 1.0, epsilon: float = 0.1, theta_0: float = 0.0,
    m_ADM: float = -1.0, sigma: float = 4.0, n_theta: int = 361,
) -> tuple[float, float]:
    """Cartesian dipole (p_x, p_y) of T_{tt}(theta).

    p_x = integral over theta of T_{tt}(theta) cos(theta) dtheta / 2 pi
    p_y = integral over theta of T_{tt}(theta) sin(theta) dtheta / 2 pi
    """
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta)
    Ttt = np.array([
        horn_T_tt_profile(float(t), R, epsilon, theta_0, m_ADM, sigma)
        for t in thetas
    ])
    p_x = float(np.trapezoid(Ttt * np.cos(thetas), thetas) / (2.0 * math.pi))
    p_y = float(np.trapezoid(Ttt * np.sin(thetas), thetas) / (2.0 * math.pi))
    return p_x, p_y


def horn_steering_magnitude(
    R: float = 1.0, epsilon: float = 0.1, theta_0: float = 0.0,
    m_ADM: float = -1.0, sigma: float = 4.0,
) -> float:
    """sqrt(p_x^2 + p_y^2)."""
    p_x, p_y = horn_steering_vector(R, epsilon, theta_0, m_ADM, sigma)
    return float(math.hypot(p_x, p_y))


@dataclass(frozen=True)
class HornSweep:
    epsilon_grid: np.ndarray
    steering_magnitude: np.ndarray
    novelty_verdict: str


def novelty_scan(
    epsilon_range: tuple[float, float] = (0.0, 0.99), n_eps: int = 30,
    R: float = 1.0, theta_0: float = 0.0,
    m_ADM: float = -1.0, sigma: float = 4.0,
) -> dict:
    """Sweep epsilon (horn-twist amplitude) and report the
    steering-magnitude curve.

    Predicted catcher hit: at epsilon close to the pinch threshold
    epsilon = 1, the cos(theta - theta_0) factor amplifies T_{tt}
    variation across theta, producing a sharp Hamming transition.
    """
    eps_grid = np.linspace(*epsilon_range, n_eps)
    mag = np.zeros(n_eps)
    px_arr = np.zeros(n_eps)
    py_arr = np.zeros(n_eps)
    for i, eps in enumerate(eps_grid):
        p_x, p_y = horn_steering_vector(R, float(eps), theta_0, m_ADM, sigma)
        px_arr[i] = p_x
        py_arr[i] = p_y
        mag[i] = math.hypot(p_x, p_y)
    def fn(ev):
        idx = int(np.argmin(np.abs(eps_grid - ev)))
        return np.array([float(px_arr[idx]), float(py_arr[idx]), float(mag[idx])])
    result = scan_novelty(eps_grid, fn, n_bits=32)
    return {
        "epsilon_grid": eps_grid.tolist(),
        "steering_magnitude": mag.tolist(),
        "p_x": px_arr.tolist(),
        "p_y": py_arr.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
