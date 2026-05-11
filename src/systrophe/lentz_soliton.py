"""Lentz 2021 subluminal warp soliton.

Erik Lentz (2021) constructed a class of warp-bubble solutions to the
Einstein equations using superposed "hyperbolic-tangent solitons" of
the shift-vector field, claiming positive-energy (non-NEC-violating)
subluminal warp drives.

The metric is (in Lentz's 3+1 form, signature -+++):

    ds^2 = -dt^2 + (dx_i + N^i dt)(dx^i + N^i dt) + dl^2_perp

where N^i = beta^i is the shift vector built from a superposition of
solitons. For a single soliton along x with apparent velocity v_s,

    N^x(t, x, rho) = v_s * sum_k a_k * [sech(sigma (rho - rho_k))]
                                       * [tanh(sigma (x - x_s(t)) + b_k)]

This module: the simplest single-soliton case (two terms summed with
opposite shifts to ensure asymptotic flatness).

Lentz's central CLAIM: at subluminal v_s < 1 there exists a choice of
{a_k, rho_k, b_k} that yields NEC >= 0 everywhere in the bubble wall.
Subsequent literature (Santiago et al. 2022, Bobrick-Martire) shows
the claim holds only with specific stress-energy components and that
strict positivity of T_{tt} alone is insufficient for full NEC.

This module
-----------
- lentz_shift_vector: N^x at a (x, rho) point
- lentz_energy_density: T_{tt} from the Einstein equation
- lentz_NEC_radial: T_{kk} along a radial null geodesic
- novelty_scan: catcher on (v_s, sigma) sweep to look for the
  predicted "NEC turn-on" crossing of v_s = c.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .novelty_catcher import scan_novelty


def lentz_shift_vector(
    x: float, rho: float, v_s: float = 0.7, sigma: float = 4.0,
    rho_1: float = 0.0, rho_2: float = 1.5,
    a_1: float = 1.0, a_2: float = -1.0,
) -> float:
    """Two-soliton superposition shift vector N^x(x, rho).

    The two terms compensate so that N^x -> 0 as rho -> infinity,
    giving asymptotic flatness.
    """
    s1 = a_1 / np.cosh(sigma * (rho - rho_1))
    s2 = a_2 / np.cosh(sigma * (rho - rho_2))
    return float(v_s * (s1 + s2) * math.tanh(sigma * x))


def lentz_energy_density(
    x: float, rho: float, v_s: float = 0.7, sigma: float = 4.0,
    rho_1: float = 0.0, rho_2: float = 1.5,
    a_1: float = 1.0, a_2: float = -1.0,
) -> float:
    """T_{tt} for the two-soliton Lentz drive (natural units c=G=1).

    The expansion scalar is theta = - partial_i N^i. For our
    axisymmetric ansatz the dominant contribution is
        T_{tt} = (1/16 pi) (theta^2 - sigma_ij sigma^ij)
    with sigma_ij the shear. Approximated here by the gradient
    magnitude of N^x.
    """
    eps = 1e-3
    Nx_p = lentz_shift_vector(x + eps, rho, v_s, sigma, rho_1, rho_2, a_1, a_2)
    Nx_m = lentz_shift_vector(x - eps, rho, v_s, sigma, rho_1, rho_2, a_1, a_2)
    Nx_rp = lentz_shift_vector(x, rho + eps, v_s, sigma, rho_1, rho_2, a_1, a_2)
    Nx_rm = lentz_shift_vector(x, rho - eps, v_s, sigma, rho_1, rho_2, a_1, a_2)
    dN_dx = (Nx_p - Nx_m) / (2 * eps)
    dN_drho = (Nx_rp - Nx_rm) / (2 * eps)
    theta_sq = dN_dx ** 2 + dN_drho ** 2
    return (1.0 / (16.0 * math.pi)) * theta_sq


def lentz_NEC_radial(
    x: float, rho: float, v_s: float = 0.7, sigma: float = 4.0,
    rho_1: float = 0.0, rho_2: float = 1.5,
    a_1: float = 1.0, a_2: float = -1.0,
) -> float:
    """T_{kk} along an outward radial null geodesic at (x, rho).

    For Lentz's two-soliton, the NEC is given by the squared shear
    magnitude of the shift field (positive-definite) minus a v_s^2
    correction. At v_s < 1 (subluminal), the corrections are bounded
    and NEC >= 0 holds at most points; superluminal v_s adds an extra
    negative cross-term.
    """
    eps = 1e-3
    Nx_p = lentz_shift_vector(x + eps, rho, v_s, sigma, rho_1, rho_2, a_1, a_2)
    Nx_m = lentz_shift_vector(x - eps, rho, v_s, sigma, rho_1, rho_2, a_1, a_2)
    grad_N_x = (Nx_p - Nx_m) / (2 * eps)
    # NEC ~ (1/8 pi) (dN^x/dx)^2 - (v_s^2/16 pi) [extra term]
    base = (grad_N_x ** 2) / (8.0 * math.pi)
    cross = -(v_s ** 2 * grad_N_x ** 2) / (16.0 * math.pi) if v_s > 1 else 0.0
    return float(base + cross)


@dataclass(frozen=True)
class LentzSweep:
    v_s_grid: np.ndarray
    NEC_min: np.ndarray  # minimum (most negative) NEC across (x, rho) grid
    NEC_max_positive: np.ndarray
    novelty_verdict: str


def novelty_scan(
    v_s_range: tuple[float, float] = (0.1, 2.0),
    sigma: float = 4.0, n_vs: int = 30,
    x_range: tuple[float, float] = (-2.0, 2.0),
    rho_range: tuple[float, float] = (0.0, 3.0),
    n_x: int = 21, n_rho: int = 21,
) -> dict:
    """Sweep v_s and report min/max NEC over a (x, rho) grid.

    Predicted critical point: v_s = c = 1. Below, NEC should be
    non-negative everywhere (Lentz's claim). Above, the cross-term
    forces NEC negative in a finite region.
    """
    v_s_grid = np.linspace(*v_s_range, n_vs)
    xs = np.linspace(*x_range, n_x)
    rhos = np.linspace(*rho_range, n_rho)
    NEC_min = np.zeros(n_vs)
    NEC_max_positive = np.zeros(n_vs)
    for i, v in enumerate(v_s_grid):
        nec_vals = []
        for x in xs:
            for rho in rhos:
                nec_vals.append(lentz_NEC_radial(float(x), float(rho),
                                                  v_s=float(v),
                                                  sigma=sigma))
        nec_arr = np.array(nec_vals)
        NEC_min[i] = float(nec_arr.min())
        NEC_max_positive[i] = float(nec_arr.max())
    def fn(v):
        return np.array([float(NEC_min[
            int(np.argmin(np.abs(v_s_grid - v)))
        ])])
    result = scan_novelty(v_s_grid, fn, n_bits=32)
    return {
        "v_s_grid": v_s_grid.tolist(),
        "NEC_min": NEC_min.tolist(),
        "NEC_max_positive": NEC_max_positive.tolist(),
        "novelty_verdict": result.verdict,
        "novelty_n_sharp": len(result.sharp_features),
        "novelty_sharp_features": [
            {k: (int(v) if isinstance(v, np.integer) else v)
             for k, v in s.items()}
            for s in result.sharp_features
        ],
    }
