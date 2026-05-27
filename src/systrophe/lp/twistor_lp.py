"""Penrose twistor transform on the LP rotating-cylinder background.

In Penrose twistor theory, points in (3+1)D Minkowski spacetime are
mapped to lines in a 4D twistor space CP^3. Curved spacetimes generally
do not have a global twistor structure, but locally one can construct
"twistor fibers" associated with null geodesics through each point.

For the LP exterior, null geodesics include radial null rays and
phi-loops at chronology horizons. The local twistor at a point r
encodes the directions of null geodesics emanating from r.

Module:
- alpha_plane_at_r: principal null direction in LP frame
- twistor_norm: alpha-plane norm |Z|^2
- twistor_inner_product: <Z, W>
- self_dual_test: alpha-plane self-duality
- pair_extinction_twistor: twistor collapse at delta=pi
- novelty_scan: catcher on twistor-norm profile
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from systrophe.catchers.novelty_catcher import scan_novelty
from systrophe.geometry.vanstockum import VanStockumInterior


@dataclass(frozen=True)
class TwistorAtPoint:
    r: float
    spinor_omega: complex
    spinor_pi: complex
    norm_squared: float


def alpha_plane_at_r(vs: VanStockumInterior, r: float) -> TwistorAtPoint:
    """Local twistor (omega^A, pi_{A'}) at radius r.

    omega^A = (1, F(r))^T (positive helicity);
    pi_{A'} = (sqrt(K(r)), sqrt(L(r)))^T.
    Returns complex amplitudes (heuristic).
    """
    F = float(vs.analytic_exterior_F(np.array([r]))[0])
    K = float(vs.analytic_exterior_K(np.array([r]))[0])
    L = float(vs.analytic_exterior_L(np.array([r]))[0])
    omega = complex(1.0, F)
    pi_spinor = complex(math.sqrt(abs(K)) if K >= 0 else 0,
                         math.sqrt(abs(L)) if L >= 0 else 0)
    norm = abs(omega) ** 2 + abs(pi_spinor) ** 2
    return TwistorAtPoint(
        r=r, spinor_omega=omega, spinor_pi=pi_spinor,
        norm_squared=float(norm),
    )


def twistor_norm(vs: VanStockumInterior, r: float) -> float:
    """|Z|^2 for the local twistor."""
    Z = alpha_plane_at_r(vs, r)
    return float(Z.norm_squared)


def twistor_inner_product(
    vs: VanStockumInterior, r1: float, r2: float,
) -> complex:
    """Inner product <Z(r1), Z(r2)>."""
    Z1 = alpha_plane_at_r(vs, r1)
    Z2 = alpha_plane_at_r(vs, r2)
    return Z1.spinor_omega.conjugate() * Z2.spinor_omega + \
            Z1.spinor_pi.conjugate() * Z2.spinor_pi


def self_dual_test(vs: VanStockumInterior, r: float) -> dict:
    """Self-duality criterion: |omega|^2 = |pi|^2."""
    Z = alpha_plane_at_r(vs, r)
    is_self_dual = abs(abs(Z.spinor_omega) ** 2 - abs(Z.spinor_pi) ** 2) < 0.1
    return {
        "r": r,
        "|omega|^2": float(abs(Z.spinor_omega) ** 2),
        "|pi|^2": float(abs(Z.spinor_pi) ** 2),
        "is_self_dual": bool(is_self_dual),
    }


def pair_extinction_twistor(
    vs: VanStockumInterior, r: float, delta: float,
) -> dict:
    """Twistor norm scales by (1+cos delta)/2 in pair geometry."""
    base = twistor_norm(vs, r)
    extinction = 0.5 * (1.0 + math.cos(delta))
    return {
        "r": r,
        "delta": delta,
        "extinction_factor": extinction,
        "norm_single": base,
        "norm_pair": float(base * extinction),
        "twistor_collapsed": bool(extinction < 1e-9),
    }


def novelty_scan(n_radii: int = 30) -> dict:
    """Catcher on twistor-norm + inner-product profile."""
    vs = VanStockumInterior(omega=1.0, R=1.0)
    r_grid = np.linspace(1.05 * vs.R, 10 * vs.R, n_radii)
    r_ref = 1.5
    def fn(r: float) -> np.ndarray:
        Z = alpha_plane_at_r(vs, r)
        ip = twistor_inner_product(vs, r_ref, r)
        return np.array([
            Z.norm_squared,
            float(abs(Z.spinor_omega) ** 2),
            float(abs(Z.spinor_pi) ** 2),
            float(abs(ip)),
        ])
    result = scan_novelty(r_grid, fn, n_bits=32, parameter_label="r")
    return {
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }
